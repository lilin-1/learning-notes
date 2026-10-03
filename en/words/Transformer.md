A [[neural network]] architecture built around the [[attention mechanism]], with no recurrence or convolution, consisting of an [[encoder]] and a [[decoder]]. Input [[token|tokens]] pass through the [[embedding]], receive a [[positional encoding]], and enter the network.

The encoder stacks $N$ identical layers, each with two sublayers: [[multi-head attention|multi-head]] [[self-attention]] and a position-wise [[feedforward network]]. Each decoder layer has an additional [[cross-attention]] sublayer over the encoder output, and its self-attention carries a [[causal mask]]. In the original design, each sublayer's output is $\mathrm{LayerNorm}\big(x + \mathrm{Sublayer}(x)\big)$: a [[residual connection]] first, then [[layer normalization]].

Example: The original base model takes $N = 6$, $d = 512$, $h = 8$ and a feedforward inner dimension of 2048, about 65 million parameters in all. For English-to-German translation, the encoder reads the English sentence and the decoder generates German tokens one by one.

Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
