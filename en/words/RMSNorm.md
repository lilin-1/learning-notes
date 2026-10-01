[[layer normalization|Layer normalization]] that only rescales by the root mean square, without subtracting the mean:

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}},$$

where $\gamma$ is a learned parameter and $\epsilon$ a small positive number that prevents division by zero. Compared with layer normalization, it drops the subtraction of the mean and the bias $\beta$.

Note: The mean is dropped because the benefit of normalization comes mainly from rescaling rather than shifting. In the original paper's experiments, it performs on par with layer normalization while computing one mean and one subtraction fewer per position. Today's large models mostly use it, together with [[Pre-LN]].

Example: The root mean square of $x = (2, 4, 6, 8)$ is $\sqrt{(4 + 16 + 36 + 64) / 4} = \sqrt{30} \approx 5.48$. With $\gamma = 1$ and $\epsilon$ ignored, the output is about $(0.37, 0.73, 1.10, 1.46)$; unlike layer normalization, the output does not have mean 0.
