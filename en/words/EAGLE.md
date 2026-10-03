A [[speculative decoding]] method that runs autoregression at the feature level. Its draft model is tiny, a single decoder layer whose input is the large model's second-to-top-layer features together with the token sequence shifted by one. The draft model predicts the feature of the next position step by step, and the large model's output layer turns it into a draft token.

Note: Predicting features is easier than predicting tokens directly, because features change smoothly while tokens are discrete choices. Feeding in the next token as well removes the uncertainty about which token the next step takes. The later EAGLE-2 adjusts the candidate tree dynamically by the draft's confidence, and EAGLE-3 switches to predicting tokens directly while fusing features from several layers.

Example: In the original paper, LLaMA2-Chat 70B was sped up 2.7 to 3.5 times on three tasks: HumanEval, GSM8K and Alpaca. On MT-Bench it was sped up 3.0 times.

Paper: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) (Li et al., 2024)
