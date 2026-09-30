# Why is FlashAttention fast?

Because standard attention is slowed by memory traffic, not by computation. Computing [[scaled dot-product attention]] by its definition means writing the $n \times n$ scores to [[memory hierarchy|GPU memory]], reading them back for [[softmax]], writing the result, and reading it once more to multiply by the values. These steps all have very low [[arithmetic intensity]], and almost all their time goes into round trips to GPU memory.

[[FlashAttention]] [[kernel fusion|fuses]] these steps into one kernel and loads the queries, keys and values into shared memory [[tiling|tile by tile]], so the score matrix is never written back to GPU memory. The difficulty is that softmax needs the maximum and the sum over a whole row; FlashAttention updates these two numbers tile by tile and corrects the accumulated result, so it gets the right answer without ever seeing a whole row.

Example: For a sequence of length 4096, one head's score matrix has $4096^2 \approx 16.78$ million numbers, 32 MiB in 16-bit precision. A standard implementation writes it twice and reads it twice, 128 MiB of traffic in all; this is exactly what FlashAttention saves.
