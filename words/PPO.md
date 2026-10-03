一种限制每次更新幅度的强化学习算法，全称近端策略优化。策略是参数为 $\theta$ 的模型，$\pi_\theta(a \mid s)$ 是它在状态 $s$ 下选择动作 $a$ 的概率；用于[[语言模型]]时，状态是提示与已生成的部分，动作是下一个词元。优势 $\hat{A}$ 衡量一个动作比平均好多少，由一个价值模型估计。PPO 先用更新前的策略 $\pi_{\mathrm{old}}$ 采样，再最大化

$$\mathbb{E}\Big[\min\big(\rho \hat{A},\ \mathrm{clip}(\rho, 1 - \epsilon, 1 + \epsilon)\, \hat{A}\big)\Big], \quad \rho = \frac{\pi_\theta(a \mid s)}{\pi_{\mathrm{old}}(a \mid s)},$$

其中 $\mathbb{E}$ 对采样到的状态与动作取平均；$\mathrm{clip}$ 把比值截断到 $[1 - \epsilon, 1 + \epsilon]$ 之内，小于下限取下限，大于上限取上限。常取 $\epsilon = 0.2$。

注：裁剪不让新策略离旧策略太远：比值 $\rho$ 超出 $[1 - \epsilon, 1 + \epsilon]$ 之后，再往同一方向改变不带来更多收益，更新便停在附近。同一批采样因此可以安全地用来更新好几次。代价是要训练一个价值模型来估计优势，它通常与策略同样大，显存与计算都多出不少。

例：$\hat{A} = 1$、$\epsilon = 0.2$ 时，新策略若把这个动作的概率提高到原来的 1.5 倍，目标只按 1.2 计；提高到 1.1 倍时，按 1.1 计。

论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)（Schulman 等，2017）
