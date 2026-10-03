8-bit floating-point formats, of which there are two: E4M3 has 4 exponent bits and 3 mantissa bits, with a maximum of 448; E5M2 has 5 exponent bits and 2 mantissa bits, with a maximum of 57344. The former is more precise and the latter has a wider range; usually weights and activations use E4M3 and gradients E5M2. From Hopper on, [[Tensor Core|Tensor Cores]] support FP8 matrix multiplication directly, at twice the speed of 16 bits.

Note: 8 bits represent too few numbers to cover a tensor's range by the format alone, so FP8 is always used with scale factors. The tensor is first multiplied by a factor that brings its maximum near the top of the format, then converted to FP8. The finer the scaling, the higher the precision. In training DeepSeek-V3, activations take one factor per 128 elements and weights one per $128 \times 128$ block, with E4M3 for all tensors.

Example: The maximum of E4M3 is $1.75 \times 2^8 = 448$. For a tensor whose largest absolute value is 3.5, multiplying by $448 / 3.5 = 128$ before conversion puts the largest element exactly at 448.

Paper: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022)
