把实数向量映为概率分布的函数：

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}, \quad i = 1, \dots, n.$$

输出的各分量为正且和为 $1$，并保持原分量的大小次序；各分量加上同一常数，输出不变。

例：$\mathrm{softmax}(1, 2, 3) \approx (0.09, 0.24, 0.67)$；$\mathrm{softmax}(11, 12, 13)$ 的结果与之相同。
