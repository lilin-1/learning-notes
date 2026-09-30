# Why can a Transformer train in parallel but generate only one token at a time?

Because in training the token each position must predict is already known, whereas in generation the next token cannot be computed until the previous one has been chosen. In training the whole sentence is given, and [[self-attention]] computes the outputs at all positions at once with matrix multiplications. The [[causal mask]] ensures that position $i$ uses only positions $1, \dots, i$, so the predictions of a [[language model]] at all $n$ positions are obtained simultaneously, together with the [[cross-entropy]] loss.

In generation, token $t + 1$ depends on the choice of token $t$, so each step can produce only one [[token]]. The [[KV cache]] spares each step from recomputing the keys and values of earlier positions, but the number of forward passes still equals the length generated.

Example: For a sentence of length 100, training obtains the losses at all 100 positions in one forward pass; generating 100 tokens takes 100 forward passes in turn.
