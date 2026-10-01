# What does it take to train a Transformer?

Besides the model itself: an objective, a way to compute gradients, an optimizer and a few measures that keep training stable. The objective is a [[loss function]], for a [[language model]] the per-position [[cross-entropy]]; the gradients are computed by [[backpropagation]]. The parameters are updated by [[Adam]], a variant of [[gradient descent]], with the learning rate first rising and then falling according to a [[learning rate warmup]].

To reduce overfitting and overconfidence, the original design also uses [[dropout]] and [[label smoothing]]; [[residual connection|residual connections]] and [[layer normalization]] make very deep networks trainable. Large models today mostly switch the optimizer to [[AdamW]], add [[gradient clipping]] against occasional large gradients, and lower the learning rate by [[cosine annealing]] after the warmup.

Example: The base model of the original [[Transformer]] uses Adam with $\beta_1 = 0.9$ and $\beta_2 = 0.98$, 4000 warmup steps, a dropout rate of $0.1$ and label smoothing $\varepsilon = 0.1$. It was trained for 100,000 steps on 8 P100 GPUs, about 12 hours.
