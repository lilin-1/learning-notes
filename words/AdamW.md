把权重衰减与梯度分开处理的 [[Adam]]。每步在 Adam 的更新之外，再让参数按同一比例缩小一点：

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_{t-1} \right),$$

其中 $\lambda$ 是衰减系数。

注：若改在损失中加上 $\frac{\lambda}{2}\|\theta\|^2$，这一项的梯度 $\lambda\theta$ 也会被 Adam 除以 $\sqrt{\hat{v}_t}$：梯度一贯大的参数，衰减反被压小，与本意相反。分开写，每个参数都按同一比例衰减。如今训练大模型几乎都用 AdamW，Llama 2 取 $\lambda = 0.1$、$\beta_2 = 0.95$。

例：$\eta = 10^{-3}$、$\lambda = 0.1$ 时，即使梯度为 0，参数每步也乘以 $1 - 10^{-4}$，一万步后约缩为原来的 $e^{-1} \approx 0.37$。
