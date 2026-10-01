# Why is training so few parameters enough for LoRA?

Because the changes fine-tuning needs are themselves nearly low-rank. Pretraining has already learned most of the knowledge, and fine-tuning only adapts the model to a task or format, a change concentrated in a few directions. [[LoRA]] writes the increment as the product $BA$ of two narrow matrices whose rank $r$ is far below the matrix dimensions, which is exactly enough to express such changes. In the original paper, ranks of 1 to 4 already came close to full fine-tuning.

What is saved is mainly memory. Full fine-tuning stores a gradient and the two moments of [[Adam]] for every parameter, none of which frozen weights need. [[QLoRA]] further [[quantization|quantizes]] the frozen weights to 4 bits, so a 65-billion-parameter model can be fine-tuned on one 48 GB GPU.

Example: A $4096 \times 4096$ matrix with $r = 8$ needs only 65536 trained parameters, 0.39% of the original. For GPT-3 175B, LoRA reduced the trained parameters about ten-thousandfold and training memory from 1.2 TB to 350 GB.
