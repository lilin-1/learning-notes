A method that fine-tunes a language model directly on preference data without training a reward model; the name stands for direct preference optimization. For comparison data $(x, y_w, y_l)$ it minimizes

$$\mathcal{L} = \log\Big(1 + e^{-\beta (\Delta_w - \Delta_l)}\Big), \quad \Delta = \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)},$$

where $\pi_{\mathrm{ref}}$ is the reference model, usually the instruction-tuned model, and $\beta$ controls how far the policy may deviate from it. The loss makes the log-probability of the better response rise more, relative to the reference model, than that of the worse one.

Note: It rests on an identity: under the [[KL divergence|KL-penalized]] objective of [[reinforcement learning from human feedback|RLHF]], the optimal policy and the reward correspond one to one, $r = \beta \log \frac{\pi}{\pi_{\mathrm{ref}}}$ plus a term that depends only on the prompt. Substituting this relation into the loss of the [[reward model]] gives a loss in terms of the policy alone. It needs only ordinary supervised training, with no sampling during training and no value model.

Example: Take $\beta = 0.1$. If the better response's log-probability is 3 above the reference model's and the worse one's is 2 below, then $\beta(\Delta_w - \Delta_l) = 0.5$ and the loss is $\ln(1 + e^{-0.5}) \approx 0.47$. When the gap widens to $\beta(\Delta_w - \Delta_l) = 2$, the loss falls to about 0.13.

Paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
