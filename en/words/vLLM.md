An open-source inference and serving engine for [[language model|language models]]. It manages the [[KV cache]] in blocks in the manner of [[PagedAttention]], schedules requests with [[continuous batching]], and offers an OpenAI-compatible HTTP interface. It also supports [[prefix caching]], [[chunked prefill]], [[speculative decoding]], [[tensor parallelism]], [[pipeline parallelism]] and [[quantization]].

Note: Its throughput advantage comes mainly from scheduling and memory management rather than from any faster operator. The block-based cache lets one GPU hold more concurrent sequences, and continuous batching keeps batches full; attention and other operators are left to ready-made kernels such as [[FlashAttention]].

Example: `vllm serve facebook/opt-13b --tensor-parallel-size 4` starts a server on 4 GPUs with tensor parallelism, after which any OpenAI client can send requests to `/v1/chat/completions`.
