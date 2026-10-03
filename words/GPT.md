仅由带[[因果掩码]]的 [[Transformer]] [[解码器]]层堆叠而成、以预测下一个[[词元]]为目标预训练的[[语言模型]]。与原始解码器相比，它去掉了[[交叉注意力]]子层。生成时，每步从下一个词元的分布中选取一个，接到输入末尾，如此循环。

例：GPT-1 有 12 层，$d = 768$，$h = 12$，约 1.17 亿个参数。

论文：[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)（Radford 等，2018）
