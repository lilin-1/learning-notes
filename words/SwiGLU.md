以门控线性单元代替[[前馈网络]]第一层与[[激活函数]]的结构：

$$\mathrm{FFN}(x) = \big(\mathrm{Swish}(xW) \odot xV\big)\, W_2, \quad \mathrm{Swish}(z) = \frac{z}{1 + e^{-z}},$$

其中 $\odot$ 是逐元素相乘。输入经两个矩阵分成两路，一路经过 Swish，与另一路逐元素相乘，每一路都调节着另一路的大小，再由 $W_2$ 投影回来。

注：多了一个矩阵 $V$，参数随之变多。为保持参数量不变，中间维数取原来的 $2/3$：两个 $d \times 4d$ 的矩阵，换成三个 $d \times \frac{8}{3}d$ 的矩阵。原论文的实验中，它的困惑度低于用 ReLU 或 GELU 的前馈网络，原因尚无公认的解释。如今的大模型多用它。

例：$d = 4096$ 时，原始前馈网络的两个矩阵共 $2 \times 4096 \times 16384 \approx 1.34 \times 10^8$ 个参数。SwiGLU 取中间维数 $\frac{8}{3} \times 4096 \approx 10923$，三个矩阵共 $3 \times 4096 \times 10923 \approx 1.34 \times 10^8$ 个，与之相当；Llama 2 7B 实取 11008。
