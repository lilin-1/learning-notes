A method that migrates outliers in the [[activation|activations]] to the weights so that both weights and activations can be quantized to 8 bits. For each input channel $j$ it takes the factor

$$s_j = \frac{\max|X_j|^{\alpha}}{\max|W_j|^{1 - \alpha}},$$

where $X_j$ is channel $j$ of the activations and $W_j$ the row of weights multiplied by it. It divides the activations by $s_j$ and multiplies the corresponding weights by $s_j$, leaving the product $XW$ unchanged. $\alpha$ controls the degree of migration; OPT and BLOOM use 0.5.

Note: The difficulty lies in the activations. In large models, a few channels of the activations are about 100 times larger than the rest; quantized with one shared scale factor, the other channels almost all become 0. The weights, by contrast, are evenly spread and easy to quantize. After migration, the two sides are equally hard, so the whole matrix multiplication can run on 8-bit integers, which saves memory and also uses the integer [[Tensor Core|Tensor Cores]]. The original paper reports up to 1.56 times the speed and half the memory.

Example: If a channel's activations reach at most 64 and the corresponding weights at most 1, then with $\alpha = 0.5$, $s = \sqrt{64} / 1 = 8$. After dividing by 8 the activations reach at most 8, and after multiplying by 8 so do the weights: the two sides are balanced.

Paper: [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) (Xiao et al., 2022)
