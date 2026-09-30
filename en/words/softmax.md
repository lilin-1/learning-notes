The function that maps a real vector to a probability distribution:

$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}, \quad i = 1, \dots, n.$$

The outputs are positive, sum to $1$ and preserve the order of the inputs; adding the same constant to every input leaves the output unchanged.

Example: $\mathrm{softmax}(1, 2, 3) \approx (0.09, 0.24, 0.67)$, and $\mathrm{softmax}(11, 12, 13)$ gives the same result.
