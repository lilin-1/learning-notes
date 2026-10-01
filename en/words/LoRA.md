A fine-tuning method that freezes the pretrained weights $W_0$ and trains only a low-rank increment:

$$W = W_0 + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d \times r}, \ A \in \mathbb{R}^{r \times k}, \ r \ll \min(d, k).$$

$A$ is initialized randomly and $B$ to zero, so the increment is zero at the start of training and the model is exactly the original.

Note: It works because the changes fine-tuning needs are themselves nearly low-rank: in the original paper, even ranks of 1 to 4 for GPT-3's attention matrices came close to full fine-tuning. After training the increment can be added into $W_0$, so inference costs nothing extra. The main benefit is memory, since frozen parameters need no gradients or optimizer state. For GPT-3 175B, training memory fell from 1.2 TB to 350 GB, and each task needs only an extra increment of tens of megabytes.

Example: With $d = k = 4096$ and $r = 8$, the increment has $2 \times 4096 \times 8 = 65536$ parameters, only $0.39\%$ of the original matrix's $4096^2 \approx 16.78$ million.
