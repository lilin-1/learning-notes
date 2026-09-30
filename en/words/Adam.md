A variant of [[gradient descent]] that adapts the step size of each parameter separately. With $g_t$ the gradient at step $t$, it keeps exponential moving averages of the first and second moments of the gradient:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2,$$

and after the bias corrections $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$, it updates $\theta_t = \theta_{t-1} - \eta\, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.

Note: Dividing by $\sqrt{\hat{v}_t}$ normalizes by the gradient's historical magnitude: parameters with consistently large gradients take smaller steps, rarely updated ones relatively larger steps, which makes the method less sensitive to the learning rate. The price is two extra state values, $m$ and $v$, per parameter, so the optimizer state takes twice the memory of the parameters themselves.

Example: Let $\beta_1 = 0.9$, $\beta_2 = 0.999$ and the first gradient $g_1 = 2$. Then $m_1 = 0.2$ and $v_1 = 0.004$; after correction $\hat{m}_1 = 2$ and $\hat{v}_1 = 4$, so the step is about $\eta \cdot 2 / 2 = \eta$. With a gradient of $200$ instead, the step is still about $\eta$: the first step does not depend on the scale of the gradient.
