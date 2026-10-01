# DeepSeek-V3 in depth

This topic follows the DeepSeek-V3 technical report to see how a real large model combines many algorithms. In structure it uses [[multi-head latent attention]] and a [[mixture of experts]]; in training it adds [[multi-token prediction]] and [[FP8]], together with custom parallelism and communication. All figures come from the report.

## Overview

DeepSeek-V3 has 671 billion parameters, 37 billion of them [[active parameters|active]] per token, and was trained on 14.8 trillion tokens on 2048 H800 GPUs. Its main design choices:

| Part | Setting | Purpose |
|---|---|---|
| Layers and width | 61 layers, hidden dimension 7168 | |
| Attention | Multi-head latent attention, 128 heads of 128 dimensions | Keeps the KV cache very small |
| Feedforward | Dense feedforward networks in the first 3 layers, mixture of experts in the rest | Many parameters, little computation |
| Experts | 1 shared and 256 routed experts per layer, 8 selected per token | |
| Training objective | Next token, plus multi-token prediction of depth 1 | Better representations; drafts at inference |
| Training precision | FP8 for the main matrix multiplications | Saves compute and memory |
| Context | Trained at 4K, then extended to 128K with [[YaRN]] | |

The report states that the whole training run had no irrecoverable loss spikes and no rollbacks.

## Multi-head latent attention

Each layer compresses keys and values into a 512-dimensional vector, with a separate 64-dimensional shared key carrying the rotary position; queries are also compressed first, to 1536 dimensions, to save activation memory in training. Each token caches only $512 + 64 = 576$ numbers per layer.

Example: Over 61 layers, that is $576 \times 61 = 35136$ numbers, about 69 KiB in BF16. With a context of 128K, or 131072 tokens, the cache of one sequence takes about 8.6 GiB. Multi-head attention of the same shape would cache $2 \times 128 \times 128$ numbers per layer, about 488 GiB in all.

A small cache lets more requests run concurrently on one GPU and reduces the bytes read in each decode step, which matters most for bandwidth-bound [[prefill and decode|decoding]].

## Experts and load balancing

Each mixture-of-experts layer has 1 shared expert that every token passes through, and 256 routed experts with an inner dimension of 2048 each, of which every token selects 8. The router computes the affinity of a token for each expert with a sigmoid, picks the 8 largest, and normalizes among those 8 to obtain the weighting coefficients.

Load is balanced not by an auxiliary loss but by a set of biases:

- Each expert has a bias $b_i$, added to its affinity only when choosing the top 8 and never used in the weighting.
- After each step, an overloaded expert lowers its bias by $\gamma$ and an underloaded one raises it by $\gamma$. For the first 14.3 trillion tokens $\gamma = 0.001$, and for the last 500 billion it is 0.
- A sequence-wise balance loss with a tiny coefficient ($\alpha = 0.0001$) guards only against extreme imbalance within a single sequence.

An auxiliary loss mixes the balancing goal into the gradient, pulling against the language-modeling objective; a bias changes only the selection, not the gradient. No tokens are dropped in training or inference.

Under [[expert parallelism]], communication across machines is the most expensive, so routing also limits each token to at most 4 nodes. In the report, each token selects on average 3.2 experts per node.

## Multi-token prediction

Beyond the main model, DeepSeek-V3 chains 1 multi-token prediction module. It takes the main model's final representation and the embedding of the token after next, and predicts one token further still. The module shares the embedding layer and output head with the main model and adds only about 14 billion parameters. In training, its loss is multiplied by a coefficient $\lambda$ and added to the total: 0.3 for the first 10 trillion tokens and 0.1 for the rest.

At inference the module can be dropped, or kept as a draft for [[speculative decoding]]. The next token it guesses is accepted 85% to 90% of the time, and generation runs 1.8 times as fast.

## FP8 training

All three matrix multiplications of a linear layer are computed in FP8: the forward pass, the gradient with respect to the input and the gradient with respect to the weights. All tensors use the E4M3 format, with fine-grained scaling making up for its range:

- activations take one scale factor per token per 128 channels;
- weights take one scale factor per $128 \times 128$ block.

The fine granularity isolates outliers: a very large number affects only its own small group instead of stretching the grid points of the whole tensor.

The precision of accumulation is handled separately. The report notes that the H800's Tensor Cores keep only about 14 bits of precision when accumulating FP8 products, so errors build up in large matrices. After every 128 elements, the partial sums are therefore moved to FP32 registers on the ordinary cores before accumulation continues.

What stays in BF16 or FP32: the embedding layer, the output head, the mixture-of-experts gating, normalization and the attention operators, along with the master weights, weight gradients and optimizer state. This follows the principle of [[mixed-precision training]]: compute in low precision, keep accumulating state in high precision.

## Parallelism and communication

Training uses 16-way [[pipeline parallelism]], 64-way [[expert parallelism]] across 8 nodes and [[data parallelism]] with stage 1 of [[ZeRO]]. Memory is saved well enough to do without costly [[tensor parallelism]]. Cross-node expert parallelism brings the ratio of computation to communication to about 1 to 1, so unless communication is hidden, half the time is spent waiting.

DualPipe is the pipeline schedule designed for this:

- The forward and backward computation of each chunk is split into four parts, attention, expert dispatch, feedforward and expert combine, so that one chunk's computation overlaps another's communication.
- Micro-batches enter from both ends of the pipeline at once, shrinking the pipeline bubbles; the price is that each GPU keeps two copies of its stage's parameters.

Nodes are connected by InfiniBand at about 50 GB/s per GPU, and GPUs within a node by NVLink at about 160 GB/s. A token first travels over InfiniBand to the GPU with the same in-node index on the target node, then over NVLink to the GPU holding its target expert. Just 20 SMs suffice to saturate both kinds of bandwidth, leaving all other SMs for computation.

## Training and cost

Example: Each trillion tokens of training takes 180,000 H800 hours, so 14.8 trillion tokens take $14.8 \times 0.18 = 2.664$ million hours. Adding 0.119 million for context extension and 0.005 million for post-training gives 2.788 million hours; at 2 dollars per hour, that is about 5.58 million dollars.

This is only the rental cost of the final training run, excluding earlier research and ablations. By the $6ND$ estimate, pre-training takes about $6 \times 3.7 \times 10^{10} \times 1.48 \times 10^{13} \approx 3.3 \times 10^{24}$ floating-point operations, about $3.4 \times 10^{14}$ per GPU per second on average.

The context was extended in two stages, from 4K to 32K and then to 128K, with 1000 steps each. YaRN acts only on the shared key that carries position, with scale factor $s = 40$.

## Inference deployment

At inference, [[prefill and decode|prefill and decode]] are deployed separately, each with its own parallelism:

| | Prefill | Decode |
|---|---|---|
| Minimum deployment unit | 4 nodes, 32 GPUs | 40 nodes, 320 GPUs |
| Attention | Tensor parallelism 4, data parallelism 8 | Tensor parallelism 4, data parallelism 80 |
| Mixture of experts | Expert parallelism 32 | Expert parallelism 320 |

Prefill processes many tokens at once and is compute-heavy, so a smaller unit suffices; decoding has only one token per sequence per step and needs very large batches to use the compute fully, so its unit is much larger. In decoding each GPU holds only one expert, and the shared expert is treated as a routed one, so each token selects 9.

With uneven load, GPUs holding popular experts slow down the whole layer. Redundant experts therefore duplicate the busiest ones, adjusted from statistics about every 10 minutes. Prefill has 32 redundant experts, each GPU holding one more beside its original 8; in decoding, 64 GPUs are dedicated to redundant and shared experts.

## Post-training

Post-training has two steps:

1. [[instruction tuning|Instruction tuning]]: about 1.5 million examples across many domains. The reasoning data among them was generated by an internal DeepSeek-R1 model, in effect [[knowledge distillation|distilling]] R1's reasoning ability into V3.
2. Reinforcement learning with [[GRPO]], which trains no value model and estimates the baseline from scores within a group.
