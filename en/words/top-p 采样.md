A sampling method that draws the next [[token]] at each step from the smallest set of tokens whose cumulative probability reaches $p$, also called nucleus sampling. The tokens are sorted by probability in decreasing order and summed until the total is at least $p$; tokens outside the set get probability 0, and one is drawn at random from the renormalized set.

Note: Unlike [[top-k sampling]], the number of candidates adapts to the distribution: when the model is confident, one or two tokens already exceed $p$; when it is not, there are more candidates. In practice $p = 0.9$ or $0.95$ is common, together with a [[temperature sampling|temperature]].

Example: With probabilities $(0.5, 0.3, 0.1, 0.06, 0.04)$ and $p = 0.85$, the sum reaches $0.9$ at the third token, which is at least $0.85$, so the candidates are the first three, renormalized to $(0.56, 0.33, 0.11)$. With probabilities $(0.9, 0.05, 0.05)$, the first token alone reaches $p$, leaving a single candidate.

Paper: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (Holtzman et al., 2019)
