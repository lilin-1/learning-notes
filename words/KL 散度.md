衡量概率分布 $q$ 与 $p$ 相差多少的量，全称 Kullback–Leibler 散度：

$$D_{\mathrm{KL}}(q \,\|\, p) = \sum_i q_i \ln \frac{q_i}{p_i}.$$

它非负，当且仅当两个分布相同时为 0；但不对称，$D_{\mathrm{KL}}(q \| p)$ 一般不等于 $D_{\mathrm{KL}}(p \| q)$，所以不是距离。它与[[交叉熵]]相差 $q$ 自身的熵：$H(q, p) = H(q) + D_{\mathrm{KL}}(q \| p)$，其中 $H(q) = -\sum_i q_i \ln q_i$。

注：训练模型时，它用来限制一个分布偏离另一个分布的程度，常见三种说法。KL 约束要求它不超过某个值；KL 惩罚把它乘以系数 $\beta$ 后从奖励或目标中减去；KL 损失把同一项加到要最小化的损失上。后两者是同一件事的两种写法，也可看作约束的软化：$\beta$ 越大，偏离越受限制。

例：$q = (0.5, 0.5)$，$p = (0.9, 0.1)$，则 $D_{\mathrm{KL}}(q \| p) = 0.5 \ln \frac{0.5}{0.9} + 0.5 \ln \frac{0.5}{0.1} \approx 0.51$，反过来 $D_{\mathrm{KL}}(p \| q) \approx 0.37$。交叉熵 $H(q, p) \approx 1.20$，等于 $H(q) = \ln 2 \approx 0.69$ 加上 0.51。

论文：[On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694)（Kullback 与 Leibler，1951）
