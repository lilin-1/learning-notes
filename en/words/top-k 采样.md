A sampling method that draws the next [[token]] only from the $k$ most probable tokens at each step. The other tokens get probability 0, the remaining $k$ are renormalized in their original proportions, and one is drawn at random; with $k = 1$ it is [[greedy decoding]]. It is often combined with [[temperature sampling]].

Note: The truncation removes the long tail. A vocabulary has tens of thousands of tokens; the unlikely ones each have tiny probability but a sizable total, so without truncation they are often drawn and the text drifts off topic. The drawback is that $k$ is fixed: when the distribution is sharp, unreasonable tokens slip into the top $k$; when it is flat, reasonable candidates are cut off.

Example: With probabilities $(0.5, 0.3, 0.1, 0.06, 0.04)$ and $k = 2$, only the first two remain, renormalized to $(0.625, 0.375)$.

Paper: [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) (Fan et al., 2018)
