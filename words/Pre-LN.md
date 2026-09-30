把[[层归一化]]移到子层输入处的 [[Transformer]] 变体：子层输出为 $x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$，原始设计（Post-LN）则为 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$。最后一层之后另加一次层归一化；GPT-2 即采用此设计。

注：Post-LN 的每次归一化都会重新缩放[[残差连接]]上的信号。层数越多，传回浅层的梯度越依赖各层归一化的连乘，训练初期容易失稳，须靠[[学习率预热]]缓解。Pre-LN 的恒等路径贯穿全部层，梯度不经缩放即可传回，因此可以不用或少用预热。代价是残差上的信号随层数累积变大，最后须再归一化一次。

例：堆叠两个子层 $F_1$、$F_2$ 时，Pre-LN 的输出为

$$x + F_1(\mathrm{LN}(x)) + F_2\big(\mathrm{LN}(x + F_1(\mathrm{LN}(x)))\big),$$

输入 $x$ 原样出现在和式中；Post-LN 的输出为 $\mathrm{LN}\big(y + F_2(y)\big)$，其中 $y = \mathrm{LN}(x + F_1(x))$，$x$ 先后经过两次归一化。
