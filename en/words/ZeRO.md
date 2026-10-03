A method that shards the optimizer state, gradients and parameters that [[data parallelism]] duplicates on every GPU. It has three stages: stage 1 shards only the optimizer state, stage 2 also the gradients, and stage 3 the parameters as well. Each GPU keeps only its own shard and gathers the full parameters from the others temporarily when it needs them.

Note: With [[Adam]] in [[mixed-precision training]], each parameter takes 16 bytes: 2 bytes each for the 16-bit parameter and gradient, and 12 bytes for the 32-bit master weight and the first and second moments. Under data parallelism every GPU stores all 16 bytes; with stage 3 sharding over $N$ GPUs, each stores only $16 / N$ bytes. PyTorch's FSDP takes exactly the stage-3 approach. The price is communication: the first two stages communicate as much as data parallelism, while stage 3 must gather parameters before every use, about 1.5 times as much.

Example: A 7.5-billion-parameter model needs $16 \times 7.5 = 120$ GB, more than an 80 GB GPU holds. Sharded at stage 3 over 64 GPUs, each needs only about $120 / 64 \approx 1.9$ GB, plus activations.

Paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2019)
