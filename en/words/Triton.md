A language and compiler for writing [[kernel|kernels]] one thread block at a time, open-sourced by OpenAI and embedded in Python. A program describes only how one thread block processes one block of data; how the work is divided among [[warp|warps]], how to achieve [[memory coalescing]] and how to use shared memory are left to the compiler.

Note: It strikes a balance between ease and efficiency: programs are far shorter than in CUDA C++, performance often approaches that of hand-written kernels, and the size of each [[tiling|tile]] is still chosen by the programmer. On GPUs, PyTorch's `torch.compile` generates Triton kernels by default, and the same program can be compiled for both NVIDIA and AMD GPUs.

Example: To add two vectors of length $2^{20}$ with a block size of 1024, launch 1024 program instances. Instance $p$ obtains $p$ from `tl.program_id(0)`, computes its indices $1024p, \dots, 1024p + 1023$ with `tl.arange(0, 1024)`, loads that block of `x` and `y` in one go, adds them and writes the result back.

Paper: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973) (Tillet et al., 2019)
