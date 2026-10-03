A [[language model]] built solely from stacked [[Transformer]] [[decoder]] layers with a [[causal mask]], pretrained to predict the next [[token]]. Compared with the original decoder, it drops the [[cross-attention]] sublayer. In generation, each step picks one token from the next-token distribution and appends it to the input, and the process repeats.

Example: GPT-1 has 12 layers, $d = 768$ and $h = 12$, about 117 million parameters.

Paper: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018)
