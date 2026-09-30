A [[Transformer]] variant that moves [[layer normalization]] to the input of each sublayer: the sublayer output is $x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$, whereas the original design (Post-LN) gives $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. One more layer normalization is added after the last layer; GPT-2 uses this design.

Note: In Post-LN, every normalization rescales the signal on the [[residual connection]]. The more layers there are, the more the gradient reaching shallow layers depends on a product of per-layer normalizations, so early training is prone to instability and relies on [[learning rate warmup]]. In Pre-LN the identity path runs through all layers and gradients return without rescaling, so warmup can be shortened or dropped. The price is a residual signal that grows with depth and has to be normalized once more at the end.

Example: Stacking two sublayers $F_1$ and $F_2$, Pre-LN outputs

$$x + F_1(\mathrm{LN}(x)) + F_2\big(\mathrm{LN}(x + F_1(\mathrm{LN}(x)))\big),$$

in which the input $x$ appears unchanged in the sum; Post-LN outputs $\mathrm{LN}\big(y + F_2(y)\big)$ with $y = \mathrm{LN}(x + F_1(x))$, so $x$ passes through two normalizations in turn.
