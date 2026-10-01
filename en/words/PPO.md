A reinforcement learning algorithm that limits the size of each update; the name stands for proximal policy optimization. A policy $\pi_\theta$ chooses an action $a$ in a state $s$; for a [[language model]], the state is the prompt with the text generated so far and the action is the next token. The advantage $\hat{A}$ measures how much better an action is than average and is estimated by a value model. PPO maximizes

$$\mathbb{E}\Big[\min\big(\rho \hat{A},\ \mathrm{clip}(\rho, 1 - \epsilon, 1 + \epsilon)\, \hat{A}\big)\Big], \quad \rho = \frac{\pi_\theta(a \mid s)}{\pi_{\mathrm{old}}(a \mid s)},$$

usually with $\epsilon = 0.2$.

Note: The clipping keeps the new policy from moving too far from the old one: once the ratio $\rho$ leaves $[1 - \epsilon, 1 + \epsilon]$, further change in the same direction brings no further gain, so the update stops nearby. The same batch of samples can therefore be used safely for several updates. The price is a value model trained to estimate the advantage, usually as large as the policy, which adds considerably to memory and computation.

Example: With $\hat{A} = 1$ and $\epsilon = 0.2$, if the new policy raises this action's probability to 1.5 times the old, the objective counts only 1.2; raised to 1.1 times, it counts 1.1.
