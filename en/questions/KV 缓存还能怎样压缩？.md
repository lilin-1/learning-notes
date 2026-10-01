# How else can the KV cache be compressed?

From three directions: store fewer heads, fewer bits and fewer positions. First, heads. [[grouped-query attention|Grouped-query attention]] lets several query heads share one set of keys and values, shrinking the cache to $g / h$ of its size. [[multi-head latent attention|Multi-head latent attention]] goes further and compresses the keys and values of all heads into one low-dimensional vector.

Next, bits and positions. [[KV cache quantization]] cuts each number from 16 bits to 8 or even 2. [[sliding window attention|Sliding window attention]] keeps only the latest stretch, and [[attention sink|attention sinks]] show that when middle positions are discarded, the first few tokens must stay. [[PagedAttention]] compresses no content but removes the waste of reserving the cache for the maximum length.

Example: DeepSeek-V3 caches 576 numbers per token per layer, where multi-head attention of the same shape would need 32768. Stored in 8 bits, each layer then needs only 576 bytes.
