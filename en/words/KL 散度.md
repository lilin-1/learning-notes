A measure of how much a probability distribution $q$ differs from $p$, in full the Kullback–Leibler divergence:

$$D_{\mathrm{KL}}(q \,\|\, p) = \sum_i q_i \ln \frac{q_i}{p_i}.$$

It is nonnegative and zero exactly when the two distributions are equal, but it is not symmetric: $D_{\mathrm{KL}}(q \| p)$ generally differs from $D_{\mathrm{KL}}(p \| q)$, so it is not a distance. It differs from the [[cross-entropy]] by the entropy of $q$ itself: $H(q, p) = H(q) + D_{\mathrm{KL}}(q \| p)$, where $H(q) = -\sum_i q_i \ln q_i$.

Note: In training models, it limits how far one distribution strays from another, under three common names. A KL constraint requires it to stay below some value; a KL penalty multiplies it by a coefficient $\beta$ and subtracts it from the reward or objective; a KL loss adds the same term to the loss being minimized. The last two are the same thing written two ways, and can be seen as a softened constraint: the larger $\beta$, the more the deviation is restricted.

Example: With $q = (0.5, 0.5)$ and $p = (0.9, 0.1)$, $D_{\mathrm{KL}}(q \| p) = 0.5 \ln \frac{0.5}{0.9} + 0.5 \ln \frac{0.5}{0.1} \approx 0.51$, while in the other direction $D_{\mathrm{KL}}(p \| q) \approx 0.37$. The cross-entropy $H(q, p) \approx 1.20$ equals $H(q) = \ln 2 \approx 0.69$ plus 0.51.

Paper: [On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694) (Kullback and Leibler, 1951)
