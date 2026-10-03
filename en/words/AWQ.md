A weight quantization method that protects the important weights according to the size of the [[activation|activations]]; the name stands for activation-aware weight quantization. A few input channels have especially large activations, and the weights multiplied by them affect the output the most. Instead of keeping these weights in high precision, AWQ multiplies them by a factor $s > 1$ before quantization and divides the corresponding inputs by $s$. The product is unchanged, and the relative error of these weights shrinks to about $1/s$.

Note: The important weights are not simply kept in 16 bits because mixed-precision matrices are hard to compute efficiently in hardware; after scaling, all weights remain in one low-bit format. The important channels must be chosen by activation, not by weight. In the original paper, protecting 1% of the weights chosen by activation lowered perplexity from 23.54 to 11.39, while choosing by weight magnitude only reached 22.37.

Example: A weight of 0.13 with grid spacing 0.1 quantizes directly to 0.1, an error of 0.03. With $s = 2$ it first becomes 0.26 and quantizes to 0.3; multiplied by the input divided by 2, the effective weight is 0.15, an error of 0.02.

Paper: [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)
