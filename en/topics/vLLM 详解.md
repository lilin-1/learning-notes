# vLLM in depth

This topic is based on the source code of vLLM v0.30.0. All paths are relative to the root of the source tree and have been checked against the source. The general principles are covered by the entries [[PagedAttention]], [[continuous batching]], [[prefix caching]], [[chunked prefill]] and [[speculative decoding]]. Two computational optimizations are covered separately by [[FlashAttention]] and [[CUDA graph]]. This topic describes only how [[vLLM]] implements them.

## Overview

vLLM splits serving into two processes. The API process handles the network and text: it receives HTTP requests, applies the chat template, splits the text into [[token|tokens]], converts generated tokens back into text and streams it back. The engine process handles computation: at every step it schedules a batch of requests, has the executor run one forward pass on the GPU, and sends the results back. The two processes communicate over ZMQ sockets, with messages encoded in msgpack.

- API process: `vllm/v1/engine/async_llm.py:AsyncLLM` holds the input processor `vllm/v1/engine/input_processor.py:InputProcessor` and the output processor `vllm/v1/engine/output_processor.py:OutputProcessor`.
- Engine process: `vllm/v1/engine/core.py:EngineCore` holds the scheduler `vllm/v1/core/sched/scheduler.py:Scheduler` and the executor `vllm/v1/executor/abstract.py:Executor`.
- Below the executor, each GPU has a worker, `vllm/v1/worker/gpu_worker.py:Worker`, whose model runner performs the forward pass. With a single GPU, the worker runs inside the engine process.

This release has only the V1 engine. The old V0 engine has been removed, and `vllm/engine/llm_engine.py` is merely an alias pointing to V1. The default model runner has also been replaced by a rewritten one, `vllm/v1/worker/gpu/model_runner.py:GPUModelRunner`.

## The life of a request

A streaming chat request passes through the following steps in order:

1. `vllm serve` is implemented by `vllm/entrypoints/cli/serve.py:ServeSubcommand`. It starts `vllm/entrypoints/launchers/api_server/entry.py:run_server`, which sets up the FastAPI application and the engine client AsyncLLM.
2. The request arrives at `vllm/entrypoints/openai/chat_completion/api_router.py:create_chat_completion` and is handed to `vllm/entrypoints/openai/chat_completion/serving.py:OpenAIServingChat`, which applies the chat template and tokenizes the text.
3. AsyncLLM turns the request into an `EngineCoreRequest` through `vllm/v1/engine/input_processor.py:process_inputs`, registers it with the output processor, and sends it to the engine process through `vllm/v1/engine/core_client.py:AsyncMPClient`.
4. The input thread of the engine process, `vllm/v1/engine/core.py:process_input_sockets`, receives the request and computes the hashes of its blocks for prefix-cache lookup; the main loop `vllm/v1/engine/core.py:run_busy_loop` puts it in the scheduler's waiting queue.
5. In each `vllm/v1/engine/core.py:step`, the scheduler decides how many tokens each request processes in this step and allocates cache blocks for them; the executor then runs the forward pass on the GPU.
6. The model runner's `vllm/v1/worker/gpu/model_runner.py:sample_tokens` samples the new tokens; with speculative decoding enabled, it also verifies the drafts.
7. The scheduler's `vllm/v1/core/sched/scheduler.py:update_from_output` updates the state of each request and packages the new tokens, which `vllm/v1/engine/core.py:process_output_sockets` sends back to the API process.
8. The output processor's `vllm/v1/engine/output_processor.py:process_outputs` uses `vllm/v1/engine/detokenizer.py:FastIncrementalDetokenizer` to convert tokens back into text incrementally and to check stop conditions, and the results stream back as SSE.

## The scheduler

The scheduler `vllm/v1/core/sched/scheduler.py:Scheduler` makes no distinction between a “prefill phase” and a “decode phase”, as its source comments state plainly. Each request records only the number of tokens already computed, `num_computed_tokens`, and every step decides how many more to compute; prefill is merely the case where many tokens are computed at once. [[chunked prefill|Chunked prefill]], [[prefix caching]] and [[speculative decoding]] are all handled by this single mechanism.

The total number of tokens per step stays within the budget `max_num_batched_tokens`, and the number of requests running at once within `max_num_seqs`. Scheduling serves running requests first; most of them are decoding and need only 1 token each. The remaining budget goes to waiting requests. When a long prompt cannot get enough budget, only part of it is computed in this step and the rest in the next, which is where chunked prefill comes from.

When a waiting request is scheduled for the first time, the scheduler first checks the prefix cache: `vllm/v1/core/kv_cache_manager.py:get_computed_blocks` finds the longest cached prefix, which is recorded directly as computed.

When cache blocks run out, the scheduler preempts. If `vllm/v1/core/kv_cache_manager.py:allocate_slots` cannot obtain blocks, the scheduler picks a request from the running queue and hands it to `vllm/v1/core/sched/scheduler.py:_preempt_request`. That function frees all its blocks, resets its computed-token count to zero and returns it to the waiting queue, to be recomputed from scratch later. This release supports only recomputation as a preemption method; swapping out to CPU memory via `--swap-space` has been removed. Under first-come-first-served scheduling, the request preempted is the one that entered the running queue last; under priority scheduling, it is the one with the lowest priority that arrived latest.

When conditions allow, asynchronous scheduling, `vllm/v1/core/sched/async_scheduler.py:AsyncScheduler`, is enabled by default, overlapping the scheduling of the next step with the execution of the current one.

## KV cache management

Management of the [[KV cache]] has three layers: `vllm/v1/core/kv_cache_manager.py:KVCacheManager` provides the interface to the scheduler, `vllm/v1/core/kv_cache_coordinator.py:KVCacheCoordinator` coordinates different types of attention layers (such as full attention and sliding windows), and `vllm/v1/core/single_type_kv_cache_manager.py:SingleTypeKVCacheManager` manages one type. Physical blocks are allocated and reclaimed centrally by `vllm/v1/core/block_pool.py:BlockPool`.

- The block size defaults to 16 tokens (`vllm/config/cache.py:DEFAULT_BLOCK_SIZE`); an attention backend may specify another.
- Free blocks form a doubly linked list, `vllm/v1/core/kv_cache_utils.py:FreeKVCacheBlockQueue`, and allocation takes from the front. When freed, blocks with a hash go to the back and blocks without one to the front. The least recently used cached blocks are thus evicted first, and until they are evicted they can still be hit again.
- Prefix caching works in units of full blocks. A block's hash is computed from its parent block's hash, the token IDs of the block and extra keys (such as LoRA or multimodal inputs), using SHA-256 by default (`vllm/v1/core/kv_cache_utils.py:hash_block_tokens`). The parent hashes chain the whole prefix together, so equal hashes mean that all tokens from the start are equal.
- A hit covers at most the prompt length minus 1: the last token must always be recomputed to obtain the logits needed to predict the next token.

## Execution

The executor is chosen by `vllm/v1/executor/abstract.py:get_class` according to the degree of parallelism. With one GPU it is `vllm/v1/executor/uniproc_executor.py:UniProcExecutor`, and the worker lives in the engine process; with several GPUs it is `vllm/v1/executor/multiproc_executor.py:MultiprocExecutor`, with one worker process per GPU and the calls of each step broadcast through a message queue in shared memory. Across machines, there is also a Ray-based executor.

The worker `vllm/v1/worker/gpu_worker.py:Worker` initializes the device, measures memory, loads the model and warms it up, then hands each step to the model runner. There are two model runners: the new default `vllm/v1/worker/gpu/model_runner.py:GPUModelRunner` and the old `vllm/v1/worker/gpu_model_runner.py:GPUModelRunner`. When the new one meets a feature it does not yet support, such as some speculative decoding methods or custom logits processors, it falls back to the old one automatically.

## Parallelism

When a model does not fit on one GPU, or one GPU cannot carry the load, vLLM spreads the computation over several GPUs. There are five ways to do so, and they can be combined:

| Kind | Flag | What is split | Communication |
|---|---|---|---|
| [[tensor parallelism]] | `-tp` | The weights and attention heads of every layer | Two all-reduces per layer |
| [[pipeline parallelism]] | `-pp` | Layers | Hidden states passed between adjacent stages |
| [[data parallelism]] | `-dp` | Requests | None for dense models |
| [[expert parallelism]] | `-ep` | The experts of MoE layers | Two all-to-alls per MoE layer |
| decode context parallelism | `-dcp` | The tokens in the KV cache | Gathering and merging inside attention |

One data-parallel replica occupies TP × PP GPUs, and the total is that times DP (`vllm/config/parallel.py:ParallelConfig`). The process groups number the GPUs in the order DP, PP, TP (`vllm/distributed/parallel_state.py:initialize_model_parallel`). Tensor parallelism is innermost, so consecutively numbered GPUs form a tensor-parallel group, usually within one machine where NVLink is available.

Tensor parallelism follows the Megatron-LM split. `vllm/model_executor/layers/linear.py:ColumnParallelLinear` splits a weight by columns, giving each GPU a slice of the output; `RowParallelLinear` splits by rows, and the partial sums on the GPUs are added by an all-reduce. Attention is split by heads: `vllm/model_executor/layers/linear.py:QKVParallelLinear` divides the query heads evenly among the GPUs, and when there are fewer key-value heads than GPUs, each key-value head is replicated on several GPUs. Each GPU caches keys and values only for its own heads. The all-reduce is done by `vllm/distributed/device_communicators/cuda_communicator.py:all_reduce`, which first tries several implementations written for small tensors and NVLink and falls back to NCCL only when none applies.

Example: Llama 3 70B has 64 query heads and 8 key-value heads. With TP = 8, each GPU computes 8 query heads and 1 key-value head; with TP = 16, each key-value head lives on 2 GPUs, so the KV cache is stored twice.

For pipeline parallelism, `vllm/distributed/utils.py:get_pp_indices` divides the layers evenly among the stages. When they do not divide evenly, the leftover layers go one each to the stages starting from the second-to-last and moving forward, because the last stage also holds the final normalization and the output layer. Layers outside a stage are replaced by the placeholder `PPMissingLayer`, and stages pass hidden states and residuals as `vllm/sequence.py:IntermediateTensors`; the model must implement the `SupportsPP` interface. A batch flows through the stages in turn, so to keep the earlier stages from idling, the engine keeps PP batches in flight at once, one more under asynchronous scheduling (`vllm/config/vllm.py:max_concurrent_batches`). `vllm/v1/engine/core.py:step_with_batch_queue` schedules and collects them in rotation. Example: 80 layers split into 3 stages give 27, 27 and 26 layers.

## Data, expert and context parallelism

Each data-parallel replica is a separate engine process with its own scheduler and KV cache. The API process sends each new request to the least loaded replica: `vllm/v1/engine/core_client.py:DPLBAsyncMPClient` scores replicas by their waiting and running requests, and once KV cache usage passes one half, it weighs waiting requests more heavily. The load of each replica is collected and published by `vllm/v1/engine/coordinator.py:DPCoordinator`.

For dense models the replicas are fully independent (`vllm/v1/engine/core.py:run_engine_core`). [[mixture of experts|Mixture-of-experts]] models differ: by default their MoE layers use tensor parallelism across all TP × DP GPUs, so every forward pass needs every replica. The replicas must therefore run their forward passes in lockstep, and a replica with no requests still runs an empty batch (`vllm/v1/engine/core.py:DPEngineCoreProc`).

With `--enable-expert-parallel`, MoE layers switch to expert parallelism: each GPU holds a number of whole experts, and tokens are sent by an all-to-all to the GPUs holding their experts and sent back afterwards (`vllm/model_executor/layers/fused_moe/config.py:FusedMoEParallelConfig`). Attention layers are unaffected and are still split by TP; with TP = 1, every replica keeps a full copy of the attention weights. The all-to-all implementation is chosen by `--all2all-backend`; the default is built from an all-gather and a reduce-scatter, and across machines dedicated kernels such as DeepEP can be used instead. Example: serving DeepSeek-V3 on 8 GPUs with TP = 1 and DP = 8 puts one copy of the attention weights on every GPU and 32 of the 256 routed experts on each.

Decode context parallelism removes duplicated KV cache. Tensor parallelism splits the cache by key-value heads, but their number is fixed by the model; when TP exceeds it, the same cache is stored on several GPUs. The `-dcp` option lets those GPUs share it along the token dimension instead. By default, token $i$ is stored on GPU $i \bmod \mathrm{DCP}$ of the group (`vllm/config/parallel.py:cp_kv_cache_interleave_size`). It adds no GPUs, and its upper limit is TP divided by the number of key-value heads (`vllm/config/model.py:verify_with_parallel_config`). In attention, each GPU computes a partial result over the tokens it stores, and the partial results are merged as in [[context parallelism]] (`vllm/v1/attention/ops/dcp.py:cp_lse_ag_out_rs`), at the cost of a few more collective operations within the group per layer.

Example: DeepSeek-R1 uses [[multi-head latent attention]], which amounts to a single key-value head. With TP = 8, all 8 GPUs hold identical KV caches; adding `-dcp 8` leaves each GPU an eighth, so about 8 times as many tokens fit.

There is also prefill context parallelism, `-pcp`, which splits a long prompt across GPUs to shorten the [[time to first token]]; `docs/serving/context_parallel_deployment.md` describes it as under active development.

## Parallelism across requests

The previous two sections spread one model over several GPUs; on one copy of the model, the many requests run in parallel by being batched together. At each step, every request the scheduler selects, whether in prefill or decode, has the [[token|tokens]] it needs this step laid end to end into one $N \times d$ input, where $N$ is the total number of tokens in the step. Apart from a few positions padded for CUDA graphs, there is no padding.

- Linear layers and the [[feedforward network]] treat every token independently, so the whole batch takes one matrix multiplication. In decoding each request contributes 1 token, so batching lets one read of the weights serve every request, and the [[arithmetic intensity]] rises with the batch size.
- Attention must be computed separately, since each request may see only its own preceding text. `vllm/v1/attention/backend.py:CommonAttentionMetadata` records where each request starts among the $N$ tokens, `query_start_loc`, its length, `seq_lens`, and the block table, `block_table_tensor`. From these the attention kernel handles each request separately within one launch, reading each request's KV cache through its block table; `slot_mapping` says where in the cache the keys and values of each new token are written.
- The output layer computes [[logits]] only at the last position of each request, given by `logits_indices` of `vllm/v1/worker/gpu/input_batch.py:InputBatch`. The positions inside a prefill need none; with speculative decoding the draft positions are added. Sampling, too, runs over the whole batch at once.

Example: a step holding, in order, 3 decode requests and a 100-token stretch of prefill has $N = 103$ and `query_start_loc` equal to $(0, 1, 2, 3, 103)$. The output layer computes only positions 0, 1, 2 and 102.

Between steps there are two further overlaps. The first is asynchronous scheduling, which lets the CPU prepare the next step while the GPU runs this one; see the section on the scheduler. The second is dual batch overlap: an MoE model served with data and expert parallelism can split a batch into two microbatches with `--enable-dbo` (`vllm/v1/worker/ubatching.py:UBatchContext`). While one microbatch is in its all-to-all communication, the other computes, so the communication time is hidden (`docs/design/dbo.md`).

On the network side, the API processes split text into tokens and turn tokens back into text. With data parallelism there are by default as many API processes as replicas (`vllm/entrypoints/cli/serve.py:cmd`), adjustable with `--api-server-count`.

## Attention backends

There is no longer a dedicated PagedAttention kernel; the old CUDA implementation has been removed from the source. Attention is now computed by one of several backends: the block table is passed into the kernel as an argument, and the kernel reads the scattered keys and values through it. On CUDA GPUs there are four main backends:

- FlashAttention, an implementation of the [[FlashAttention]] algorithm. vLLM compiles in a fork that it maintains itself; the interface is in `vllm/vllm_flash_attn/flash_attn_interface.py` and the backend in `vllm/v1/attention/backends/flash_attn.py:FlashAttentionImpl`.
- FlashInfer, an attention kernel library written specifically for serving large models, with native support for paged KV caches; SGLang uses it too. The backend is in `vllm/v1/attention/backends/flashinfer.py:FlashInferImpl`.
- Triton, kernels written in [[Triton]]. vLLM uses it for a single kernel that handles both prefill and decode, `vllm/v1/attention/ops/triton_unified_attention.py:unified_attention`, which needs no extra library, runs on both NVIDIA and AMD GPUs and is the most widely applicable.
- FlexAttention, the attention interface built into PyTorch. A mask or a modification of the scores is written as a Python function, which PyTorch compiles into a kernel.

When no backend is specified, vLLM ranks the backends for the GPU at hand (`vllm/platforms/cuda.py:_get_backend_priorities`), checks in order whether each supports the model's data type, head size and so on, and takes the first that qualifies. On most GPUs the order is FlashAttention, FlashInfer, Triton, FlexAttention. Data-center Blackwell GPUs of compute capability 10, such as the B200, put FlashInfer first. All this concerns standard multi-head attention, grouped-query attention included; the MLA used by DeepSeek and other models has a separate set of backends.

To choose a backend, use `--attention-backend`, for example `--attention-backend FLASHINFER`. If the chosen backend does not support the current configuration, startup fails with an error giving the reason.

## CUDA graphs

[[CUDA graph|CUDA graphs]] are recorded during warmup, triggered by `vllm/v1/worker/gpu_worker.py:compile_or_warm_up_model`. The default mode is `FULL_AND_PIECEWISE` (`vllm/config/compilation.py:CUDAGraphMode`): a pure-decode batch is recorded whole as one graph, while prefill and mixed batches record only the parts outside attention. Most attention backends can be recorded only on pure-decode batches, so the attention of the latter still launches kernel by kernel.

Which sizes are recorded is decided by `vllm/config/vllm.py:_set_cudagraph_sizes`, counted in tokens per step: first 1, 2 and 4, then every 8 from 8 to 248, then every 16 from 256 up to a ceiling. Without speculative decoding, the ceiling defaults to twice `max_num_seqs`, but at most 512 (1024 on data-center Blackwell GPUs) and at most `max_num_batched_tokens`. At run time, a step's token count is padded up to the nearest recorded size not smaller than it; steps above the ceiling run without CUDA graphs.

Example: On an H100, `max_num_seqs` defaults to 1024, so the ceiling is 512 and $3 + 31 + 17 = 51$ sizes are recorded. A decode step with 13 requests is padded to 16, computing 3 empty slots.

## How the KV cache is sized at startup

vLLM gives all the GPU memory left over by weights and activations to the KV cache, computed in three steps:

1. The memory available to this instance is set by `--gpu-memory-utilization`, 0.92 by default (`vllm/config/cache.py:gpu_memory_utilization`).
2. A forward pass on dummy inputs of maximum size measures the peak taken by weights, activations and CUDA graphs (`vllm/v1/worker/gpu_worker.py:determine_available_memory`).
3. What remains after subtracting these from the available memory all goes to the KV cache and is converted into a number of blocks by the bytes per block (`vllm/v1/core/kv_cache_utils.py:get_kv_cache_configs`).

The log line `Available KV cache memory` reports the remainder from step 3, and the line `GPU KV cache size` reports how many tokens fit, along with the maximum concurrency at the maximum length.

Example: An 80 GiB GPU at 0.92 has 73.6 GiB available. If the weights take 14 GiB and activations and CUDA graphs 4 GiB, the KV cache gets 55.6 GiB. For a 7-billion-parameter model with 32 layers, $d = 4096$ and no grouped-query attention, the cache per token is $2 \times 32 \times 4096 \times 2$ bytes, or 0.5 MiB, so about 114,000 tokens fit.

## Deployment parameters

The most frequently adjusted startup parameters are listed below, with defaults all taken from the v0.30.0 source:

| Parameter | Default | Effect |
|---|---|---|
| `--gpu-memory-utilization` | 0.92 | Fraction of GPU memory available to this instance; higher means a larger KV cache and fewer preemptions |
| `--max-model-len` | From the model | Maximum length of a single request; lower saves memory |
| `--max-num-seqs` | Hardware-dependent | Maximum number of requests running at once |
| `--max-num-batched-tokens` | Hardware-dependent | Token budget per step |
| `--enable-prefix-caching` | On | Prefix caching; `--no-enable-prefix-caching` turns it off |
| `-tp`, `-pp`, `-dp` | 1 | Degree of tensor, pipeline and data parallelism |
| `-ep` | Off | Expert parallelism for MoE layers |
| `-dcp` | 1 | Degree of decode context parallelism |
| `--kv-cache-dtype` | auto | Data type of the KV cache; fp8 halves it |
| `-q`, `--quantization` | Model-dependent | Quantization method |
| `--enforce-eager` | Off | No CUDA graphs: faster startup and less memory, slower decoding |
| `-O0` to `-O3` | O2 | Optimization level for compilation and CUDA graphs |

The defaults of `--max-num-seqs` and `--max-num-batched-tokens` are set per GPU by `vllm/engine/arg_utils.py:get_batch_defaults`. With the defaults of `vllm serve`, GPUs with at least 160 GiB of memory get 1024 and 16384. GPUs above 70 GiB such as the H100 (the A100 excepted) get 1024 and 8192, and the rest 256 and 2048.

```bash
vllm serve facebook/opt-13b --tensor-parallel-size 4 --max-model-len 4096
```

## Tuning and monitoring

The tuning advice in `docs/configuration/optimization.md`:

- Preemption means the KV cache is too small. Raise `--gpu-memory-utilization`, lower `--max-num-seqs` or `--max-num-batched-tokens`, or increase tensor parallelism, leaving more memory for the cache.
- `--max-num-batched-tokens` trades one latency against another: small values (such as 2048) give lower [[inter-token latency]], large ones lower [[time to first token]]. For throughput, the documentation recommends 8192 or more.
- When a model does not fit on one GPU but fits on one machine, use [[tensor parallelism]]. Across machines, set the degree of tensor parallelism to the GPUs per machine and that of [[pipeline parallelism]] to the number of machines; pipeline parallelism also suits GPUs without NVLink between them. When there are enough GPUs for several full copies of the model, use data parallelism to scale throughput.

Monitoring uses Prometheus metrics, defined in `vllm/v1/metrics/loggers.py:PrometheusStatLogger`:

| Metric | Meaning |
|---|---|
| `vllm:time_to_first_token_seconds` | Time to first token |
| `vllm:inter_token_latency_seconds` | Inter-token latency |
| `vllm:e2e_request_latency_seconds` | End-to-end latency |
| `vllm:kv_cache_usage_perc` | Fraction of the KV cache in use |
| `vllm:num_requests_running`, `vllm:num_requests_waiting` | Numbers of running and waiting requests |
| `vllm:num_preemptions` | Number of preemptions |
| `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries` | Prefix cache hits and queries; their ratio is the hit rate |

## Comparison with other frameworks

This section does not come from the vLLM source. The vLLM documentation contains no direct comparison with other frameworks; `docs/design/prefix_caching.md` only mentions that prefix caching has been adopted by most open-source inference frameworks, such as SGLang. What follows draws on each project's published design and compares only long-standing orientations; performance changes from release to release and should be measured.

- SGLang likewise uses a paged KV cache and continuous batching. It organizes cached prefixes in a radix tree, called RadixAttention, sharing common prefixes on the tree; vLLM uses chains of block hashes instead. Both hit under the same condition: the prefixes match token by token. SGLang also provides a frontend language for structured generation.
- TensorRT-LLM is NVIDIA's inference library, with kernels heavily optimized for NVIDIA GPUs. It too implements continuous batching (called in-flight batching in its documentation) and a paged KV cache, and supports only NVIDIA GPUs.
- vLLM loads models in Hugging Face format directly and supports a wide range of hardware. Besides CUDA, its source has implementations for ROCm, TPU, CPU and other platforms.

When choosing, three things usually come first: whether the hardware is limited to NVIDIA, whether all the required models and features are supported, and the throughput and latency measured on one's own workload.
