A structure that replaces the first layer and the [[activation function]] of a [[feedforward network]] with a gated linear unit:

$$\mathrm{FFN}(x) = \big(\mathrm{Swish}(xW) \odot xV\big)\, W_2, \quad \mathrm{Swish}(z) = \frac{z}{1 + e^{-z}},$$

where $\odot$ is elementwise multiplication. Two matrices split the input into two paths; one passes through Swish and is multiplied elementwise with the other, so each path scales the other, and $W_2$ projects the result back.

Note: The extra matrix $V$ adds parameters. To keep the parameter count unchanged, the inner dimension is cut to $2/3$: two $d \times 4d$ matrices become three $d \times \frac{8}{3}d$ matrices. In the original paper's experiments, its perplexity was lower than that of feedforward networks with ReLU or GELU, for reasons that still lack an accepted explanation. Today's large models mostly use it.

Example: For $d = 4096$, the two matrices of the original feedforward network hold $2 \times 4096 \times 16384 \approx 1.34 \times 10^8$ parameters. SwiGLU takes an inner dimension of $\frac{8}{3} \times 4096 \approx 10923$, and its three matrices hold $3 \times 4096 \times 10923 \approx 1.34 \times 10^8$, about the same; Llama 2 7B uses 11008.
