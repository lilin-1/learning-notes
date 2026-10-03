[[Adam]] with weight decay handled separately from the gradient. At every step, besides the Adam update, the parameters shrink by a common proportion:

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_{t-1} \right),$$

where $\lambda$ is the decay coefficient.

Note: If $\frac{\lambda}{2}\|\theta\|^2$ were added to the loss instead, the gradient of this term, $\lambda\theta$, would also be divided by $\sqrt{\hat{v}_t}$ in Adam: parameters with consistently large gradients would decay less, the opposite of the intent. Written separately, every parameter decays in the same proportion. Large models today are trained almost exclusively with AdamW; Llama 2 uses $\lambda = 0.1$ and $\beta_2 = 0.95$.

Example: With $\eta = 10^{-3}$ and $\lambda = 0.1$, even with zero gradient the parameters are multiplied by $1 - 10^{-4}$ at every step, shrinking to about $e^{-1} \approx 0.37$ of their size after ten thousand steps.

Paper: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov and Hutter, 2017)
