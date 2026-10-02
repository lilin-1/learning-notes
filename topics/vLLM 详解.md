版本：vLLM v0.30.0
源码：~/Documents/vllm-v0.30.0

本篇依据 vLLM v0.30.0 的源码写成。文中的路径都相对于源码根目录，均已对照源码核对。通用的原理见各词条：[[PagedAttention]]、[[连续批处理]]、[[前缀缓存]]、[[分块预填充]]、[[投机解码]]。计算上的两项优化另见 [[FlashAttention]] 与 [[CUDA 图]]。本篇只写 [[vLLM]] 如何实现它们。

## 总览

vLLM 把一次服务拆成两个进程。API 进程负责网络与文本：接收 HTTP 请求，套用对话模板并切分[[词元]]，再把生成的词元还原成文本，流式返回。引擎进程负责计算：每一步调度一批请求，交给执行器在显卡上做一次前向，再把结果送回。两个进程之间用 ZMQ 套接字通信，消息以 msgpack 编码。

- API 进程：`vllm/v1/engine/async_llm.py:AsyncLLM` 持有输入处理器 `vllm/v1/engine/input_processor.py:InputProcessor` 与输出处理器 `vllm/v1/engine/output_processor.py:OutputProcessor`。
- 引擎进程：`vllm/v1/engine/core.py:EngineCore` 持有调度器 `vllm/v1/core/sched/scheduler.py:Scheduler` 与执行器 `vllm/v1/executor/abstract.py:Executor`。
- 执行器之下，每张卡有一个工作者 `vllm/v1/worker/gpu_worker.py:Worker`，其中的模型运行器执行前向。只用一张卡时，工作者直接运行在引擎进程之内。

这一版只剩 V1 引擎。旧的 V0 已经删除，`vllm/engine/llm_engine.py` 只是指向 V1 的别名。默认的模型运行器也换成了重写的一版，即 `vllm/v1/worker/gpu/model_runner.py:GPUModelRunner`。

## 一个请求的生命周期

以一个流式的对话请求为例，它依次经过：

1. `vllm serve` 由 `vllm/entrypoints/cli/serve.py:ServeSubcommand` 实现。它启动 `vllm/entrypoints/launchers/api_server/entry.py:run_server`，建立 FastAPI 应用与引擎客户端 AsyncLLM。
2. 请求到达 `vllm/entrypoints/openai/chat_completion/api_router.py:create_chat_completion`，交给 `vllm/entrypoints/openai/chat_completion/serving.py:OpenAIServingChat` 套用对话模板、切分词元。
3. AsyncLLM 经 `vllm/v1/engine/input_processor.py:process_inputs` 把请求变成 `EngineCoreRequest`，在输出处理器中登记，再由 `vllm/v1/engine/core_client.py:AsyncMPClient` 发往引擎进程。
4. 引擎进程的输入线程 `vllm/v1/engine/core.py:process_input_sockets` 收下请求，并算好各块的哈希，供前缀缓存查找；主循环 `vllm/v1/engine/core.py:run_busy_loop` 把它放进调度器的等待队列。
5. 每一步 `vllm/v1/engine/core.py:step` 中，调度器决定各请求本步处理多少词元，并为它们申请缓存块；执行器随即在显卡上完成前向。
6. 模型运行器 `vllm/v1/worker/gpu/model_runner.py:sample_tokens` 采样出新词元；启用投机解码时，同时验证草稿。
7. 调度器的 `vllm/v1/core/sched/scheduler.py:update_from_output` 更新各请求的状态，打包新词元，由 `vllm/v1/engine/core.py:process_output_sockets` 送回 API 进程。
8. 输出处理器 `vllm/v1/engine/output_processor.py:process_outputs` 用 `vllm/v1/engine/detokenizer.py:FastIncrementalDetokenizer` 增量还原文本、检查停止条件，结果以 SSE 的形式流式返回。

## 调度器

调度器 `vllm/v1/core/sched/scheduler.py:Scheduler` 没有「预填充阶段」与「解码阶段」之分，源码的注释写得很明白。每个请求只记录已算过的词元数 `num_computed_tokens`，每一步决定它再算多少个；预填充不过是一次要算的词元很多的情形。[[分块预填充]]、[[前缀缓存]]与[[投机解码]]都由这一个机制统一处理。

每一步的词元总数不超过预算 `max_num_batched_tokens`，同时运行的请求数不超过 `max_num_seqs`。调度先照顾运行中的请求，它们多在解码，每条只需 1 个词元；剩下的预算再分给等待中的请求。长提示分不到足够的预算时，本步只算其中一段，下一步接着算，分块预填充即由此而来。

等待中的请求第一次被调度时，调度器先查前缀缓存：`vllm/v1/core/kv_cache_manager.py:get_computed_blocks` 找出已缓存的最长前缀，这一段直接记为已算过。

缓存块不够时就抢占。`vllm/v1/core/kv_cache_manager.py:allocate_slots` 分不到块，调度器便从运行队列中挑出一个请求，交给 `vllm/v1/core/sched/scheduler.py:_preempt_request`：释放它的全部块，把已算的词元数清零，放回等待队列，日后从头重算。这一版只有重算一种抢占方式，换出到内存的 `--swap-space` 已经移除。按先来先服务调度时，被抢占的是最后进入运行队列的请求；按优先级调度时，是优先级最低、到达最晚的那个。

条件允许时，默认启用异步调度 `vllm/v1/core/sched/async_scheduler.py:AsyncScheduler`，让调度下一步与执行这一步重叠进行。

## KV 缓存管理

[[KV 缓存]]的管理分三层：`vllm/v1/core/kv_cache_manager.py:KVCacheManager` 对调度器提供接口，`vllm/v1/core/kv_cache_coordinator.py:KVCacheCoordinator` 协调不同类型的注意力层（如全注意力与滑动窗口），`vllm/v1/core/single_type_kv_cache_manager.py:SingleTypeKVCacheManager` 管理其中一种。物理块由 `vllm/v1/core/block_pool.py:BlockPool` 统一分配与回收。

- 块大小默认为 16 个词元（`vllm/config/cache.py:DEFAULT_BLOCK_SIZE`），注意力后端可以另行指定。
- 空闲块排成一条双向链表 `vllm/v1/core/kv_cache_utils.py:FreeKVCacheBlockQueue`，分配时从队首取。释放时，带哈希的块排到队尾，不带哈希的块插到队首。于是最久未用的缓存块最先被淘汰，而在被淘汰之前，它们都还可能再次命中。
- 前缀缓存以满块为单位。一块的哈希由父块的哈希、本块的词元编号和额外的键（如 LoRA、多模态输入）共同算出，默认用 SHA-256（`vllm/v1/core/kv_cache_utils.py:hash_block_tokens`）。父块的哈希串起了整个前缀，所以哈希相同，就意味着从开头起的全部词元相同。
- 命中的长度至多为提示长度减 1：最后一个词元总要重算，才能得到预测下一个词元所需的得分。

## 执行

执行器由 `vllm/v1/executor/abstract.py:get_class` 按并行规模选定。只用一张卡时是 `vllm/v1/executor/uniproc_executor.py:UniProcExecutor`，工作者就在引擎进程里；多张卡时是 `vllm/v1/executor/multiproc_executor.py:MultiprocExecutor`，每张卡一个工作者进程，各步的调用经共享内存中的消息队列广播。跨机器时另有基于 Ray 的执行器。

工作者 `vllm/v1/worker/gpu_worker.py:Worker` 负责初始化设备、测算显存、加载模型与预热，再把每一步交给模型运行器。模型运行器有新旧两版：默认的新版 `vllm/v1/worker/gpu/model_runner.py:GPUModelRunner`，与旧版 `vllm/v1/worker/gpu_model_runner.py:GPUModelRunner`。新版遇到尚不支持的功能，如部分投机解码方法或自定义的得分处理器，会自动退回旧版。

## 注意力后端

注意力已没有专门的 PagedAttention 内核，旧的 CUDA 实现已从源码中删除。现在由若干后端计算：块表作为参数传进内核，由内核按块表读取分散存放的键和值。CUDA 显卡上的后端主要有四种：

- FlashAttention，即 [[FlashAttention]] 算法的实现。vLLM 在构建时编入自己维护的分支，接口见 `vllm/vllm_flash_attn/flash_attn_interface.py`，后端见 `vllm/v1/attention/backends/flash_attn.py:FlashAttentionImpl`。
- FlashInfer，一个专为大模型服务编写的注意力内核库，原生支持分页的 KV 缓存，SGLang 也采用它。后端见 `vllm/v1/attention/backends/flashinfer.py:FlashInferImpl`。
- Triton，即用 [[Triton]] 写成的内核。vLLM 用它写了一个同时处理预填充与解码的内核 `vllm/v1/attention/ops/triton_unified_attention.py:unified_attention`，不依赖额外的库，NVIDIA 与 AMD 的显卡都能运行，适用面最广。
- FlexAttention，PyTorch 自带的注意力接口。掩码或对评分的修改写成一个 Python 函数，由 PyTorch 编译成内核。

未指定后端时，vLLM 按显卡排出一个尝试的次序（`vllm/platforms/cuda.py:_get_backend_priorities`），依次核对模型的数据类型、头的维数等是否受支持，取第一个合格的。多数显卡依次尝试 FlashAttention、FlashInfer、Triton、FlexAttention。计算能力为 10 的数据中心级 Blackwell 显卡（如 B200）把 FlashInfer 排在首位。以上针对标准的多头注意力，分组查询注意力也在其中；DeepSeek 等模型所用的 MLA 另有一组后端。

要指定后端，用 `--attention-backend`，例如 `--attention-backend FLASHINFER`。指定的后端不支持当前配置时，启动即报错并给出原因。

## CUDA 图

[[CUDA 图]]在预热时录制，由 `vllm/v1/worker/gpu_worker.py:compile_or_warm_up_model` 触发。默认模式为 `FULL_AND_PIECEWISE`（`vllm/config/compilation.py:CUDAGraphMode`）：纯解码的批整个录成一张图，预填充与混合的批只录注意力以外的部分。多数注意力后端只能在纯解码的批上录制，所以后者的注意力照常逐个启动。

录制哪些大小由 `vllm/config/vllm.py:_set_cudagraph_sizes` 决定，按每步的词元数计：先是 1、2、4，再从 8 到 248 每隔 8 取一个，从 256 起每隔 16 取一个，直到上限。不用投机解码时，上限默认为 `max_num_seqs` 的 2 倍，但不超过 512（数据中心级 Blackwell 为 1024），也不超过 `max_num_batched_tokens`。运行时，一步的词元数补齐到不小于它的最近一个大小；超过上限的步骤不用 CUDA 图。

例：H100 上 `max_num_seqs` 默认为 1024，上限便取 512，共录 $3 + 31 + 17 = 51$ 种大小。一步解码若有 13 个请求，就补齐到 16，多算 3 个空位。

## 启动时如何确定 KV 缓存的大小

vLLM 把显存中权重与激活用剩的部分全部分给 KV 缓存，分三步算出：

1. 按 `--gpu-memory-utilization` 算出本实例可用的显存，默认为 0.92（`vllm/config/cache.py:gpu_memory_utilization`）。
2. 用最大规模的假输入跑一遍前向，测出权重、激活与 CUDA 图占用的峰值（`vllm/v1/worker/gpu_worker.py:determine_available_memory`）。
3. 可用显存减去这些占用，余下的都给 KV 缓存，再按每块的字节数折成块数（`vllm/v1/core/kv_cache_utils.py:get_kv_cache_configs`）。

日志中的 `Available KV cache memory` 一行给出第 3 步的余量，`GPU KV cache size` 一行给出可以容纳的词元数，以及按最大长度计算的最大并发数。

例：80 GiB 的卡按 0.92 可用 73.6 GiB。设权重占 14 GiB，激活与 CUDA 图占 4 GiB，KV 缓存便得 55.6 GiB。32 层、$d = 4096$、不用分组查询注意力的 70 亿参数模型，每个词元的缓存为 $2 \times 32 \times 4096 \times 2$ 字节，即 0.5 MiB，于是约可容纳 11.4 万个词元。

## 部署参数

最常调整的启动参数如下，默认值均出自 v0.30.0 的源码：

| 参数 | 默认 | 作用 |
|---|---|---|
| `--gpu-memory-utilization` | 0.92 | 本实例可用的显存比例；调高则 KV 缓存更大，抢占更少 |
| `--max-model-len` | 取自模型 | 单个请求的最大长度；调低可省显存 |
| `--max-num-seqs` | 随硬件 | 同时运行的请求数上限 |
| `--max-num-batched-tokens` | 随硬件 | 每步的词元预算 |
| `--enable-prefix-caching` | 开 | 前缀缓存；`--no-enable-prefix-caching` 关闭 |
| `-tp`、`-pp`、`-dp` | 1 | 张量并行、流水线并行、数据并行的规模 |
| `--kv-cache-dtype` | auto | KV 缓存的数据类型；取 fp8 可减半 |
| `-q`、`--quantization` | 随模型 | 量化方法 |
| `--enforce-eager` | 关 | 不用 CUDA 图：启动快、省显存，解码变慢 |
| `-O0` 至 `-O3` | O2 | 编译与 CUDA 图的优化级别 |

`--max-num-seqs` 与 `--max-num-batched-tokens` 的默认值由 `vllm/engine/arg_utils.py:get_batch_defaults` 按显卡决定。按 `vllm serve` 的默认，显存不小于 160 GiB 的卡取 1024 与 16384。H100 这类 70 GiB 以上的卡（A100 除外）取 1024 与 8192，其余取 256 与 2048。

```bash
vllm serve facebook/opt-13b --tensor-parallel-size 4 --max-model-len 4096
```

## 调优与监控

`docs/configuration/optimization.md` 给出的调优要点：

- 出现抢占说明 KV 缓存不够。可以调高 `--gpu-memory-utilization`，调低 `--max-num-seqs` 或 `--max-num-batched-tokens`，或加大张量并行，把更多显存留给缓存。
- `--max-num-batched-tokens` 权衡两种延迟：取小（如 2048），[[词元间延迟]]更低；取大，[[首词元延迟]]更低。追求吞吐时，文档建议取 8192 以上。
- 模型放不进一张卡、却放得进一台机器时，用[[张量并行]]。跨机器时，张量并行的规模取每台机器的卡数，[[流水线并行]]的规模取机器数；卡间没有 NVLink 时，也宜用流水线并行。卡足够放下多份完整模型时，用数据并行扩展吞吐。

监控用 Prometheus 指标，定义在 `vllm/v1/metrics/loggers.py:PrometheusStatLogger`：

| 指标 | 含义 |
|---|---|
| `vllm:time_to_first_token_seconds` | 首词元延迟 |
| `vllm:inter_token_latency_seconds` | 词元间延迟 |
| `vllm:e2e_request_latency_seconds` | 端到端延迟 |
| `vllm:kv_cache_usage_perc` | KV 缓存的占用比例 |
| `vllm:num_requests_running`、`vllm:num_requests_waiting` | 运行中与等待中的请求数 |
| `vllm:num_preemptions` | 抢占次数 |
| `vllm:prefix_cache_hits`、`vllm:prefix_cache_queries` | 前缀缓存的命中数与查询数，二者之比即命中率 |

## 与其他框架的比较

本节不出自 vLLM 的源码。vLLM 的文档中没有与其他框架的正面比较，只在 `docs/design/prefix_caching.md` 中提到，前缀缓存已为多数开源推理框架（如 SGLang）采用。以下依据各项目公开的设计，只比较长期稳定的取向；性能随版本变化，应以实测为准。

- SGLang 同样采用分页的 KV 缓存与连续批处理。它用基数树组织已缓存的前缀，称为 RadixAttention，共享树上的公共前缀；vLLM 则用块哈希链实现。两者命中的条件相同：前缀逐个词元相同。SGLang 还提供一门面向结构化生成的前端语言。
- TensorRT-LLM 是 NVIDIA 的推理库，内核针对 NVIDIA 显卡深度优化。它同样实现了连续批处理（其文档称 in-flight batching）与分页的 KV 缓存，只支持 NVIDIA 显卡。
- vLLM 直接加载 Hugging Face 格式的模型，支持的硬件较广。除 CUDA 外，源码中还有 ROCm、TPU、CPU 等平台的实现。

选择时，通常先看三点：硬件是否限于 NVIDIA，所需的模型与功能是否都已支持，以及在自己的负载上实测的吞吐与延迟。
