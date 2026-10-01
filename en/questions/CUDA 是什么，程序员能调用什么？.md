# What is CUDA, and what can a programmer call?

[[CUDA]] is NVIDIA's platform for general-purpose computation on [[graphics processing unit|GPUs]]. What it offers the programmer comes in three layers: a language for writing kernels, interfaces for managing the device, and ready-made libraries.

The language is an extension of C++. A function marked `__global__` is a [[kernel]] and runs on the GPU; inside it, `blockIdx` and `threadIdx` tell a thread which [[thread block]] it belongs to and which thread it is, `__shared__` declares shared memory, and `__syncthreads()` waits for the other threads of the block.

The interfaces are called on the [[host and device|host]]: `cudaMalloc` and `cudaFree` allocate and free GPU memory, `cudaMemcpy` copies between host and GPU, `<<<…>>>` launches a kernel, `cudaStreamCreate` creates a [[stream]], and `cudaDeviceSynchronize` waits for all work to finish. All of these calls are eventually translated into commands by the [[GPU driver]].

The libraries package the most common operators: cuBLAS for matrix multiplication, cuDNN for neural network operators and NCCL for communication among GPUs. Deep learning frameworks mostly call the libraries directly rather than writing their own kernels.
