# How do today's large models change the original Transformer?

The skeleton is unchanged; what has changed are structural choices and the internals of the layers, all aimed at more stable training, cheaper inference or greater capacity. Structurally, most models keep only the decoder, as [[GPT]] does.

Inside the layers, several changes are common. [[layer normalization|Layer normalization]] moves before each sublayer, as in [[Pre-LN]], which stabilizes training, and is itself often replaced by the cheaper [[RMSNorm]]. Position is encoded with [[rotary position embedding]], which expresses relative position directly. [[grouped-query attention|Grouped-query attention]] lets several query heads share keys and values, shrinking the [[KV cache]]. The [[feedforward network]] switches to [[SwiGLU]], and some models further replace it with a [[mixture of experts]], gaining parameters at unchanged computation.

What remains is the skeleton of the [[Transformer]]: [[self-attention]] and feedforward networks stacked in alternation, each with a [[residual connection]].

Example: Llama 2 70B uses pre-normalization with RMSNorm, SwiGLU, rotary position embedding and grouped-query attention, with 64 query heads sharing 8 groups of keys and values. Mixtral 8×7B also replaces the feedforward network with 8 experts.
