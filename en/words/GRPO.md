A variant of [[PPO]] that estimates advantages by comparison within a group instead of with a value model; the name stands for group relative policy optimization, introduced by DeepSeek. For each question it samples a group of $G$ responses with rewards $r_1, \dots, r_G$ and uses the reward standardized within the group as each response's advantage:

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)},$$

then updates with PPO's clipped objective, adding the [[KL divergence|KL penalty]] directly to the loss.

Note: The value model is dropped because it is usually as large as the policy and heavy to carry, while comparing several responses to the same question already shows which is better. It suits especially well tasks whose answers can be checked automatically: math problems by their final answers, code by running tests. The reward then comes from rules rather than from a [[reward model]], which makes it hard to exploit.

Example: A group of 4 responses scores 1 when correct and 0 otherwise, with rewards $(1, 0, 0, 1)$. The mean is 0.5 and the population standard deviation 0.5, so the advantages are $(1, -1, -1, 1)$: correct responses become more likely and wrong ones less.

Paper: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) (Shao et al., 2024)
