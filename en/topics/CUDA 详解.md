# CUDA in depth

This topic follows one kernel from source code to execution. The general concepts are covered by the entries [[host and device]], [[GPU driver]], [[command buffer]], [[thread block]], [[warp]], [[streaming multiprocessor]], [[PTX]], [[SASS]] and [[stream]]. Hardware figures come from NVIDIA's published A100 specifications. Details of command submission come from NVIDIA's published Ampere hardware manuals (open-gpu-doc), and the rest from the CUDA 13.4 documentation.

## Overview

There are seven layers from a program down to the hardware, and each layer deals only with its neighbors:

| Layer | Component | Role |
|---|---|---|
| Application | PyTorch, or a `.cu` program of one's own | Calls libraries or launches kernels directly |
| Libraries | cuBLAS, cuDNN, NCCL | Provide optimized kernels |
| Runtime | `libcudart`, functions starting with `cuda` | Manages memory, streams and launches; creates the context on first use |
| User-mode driver | `libcuda.so`, functions starting with `cu` | Translates calls into commands and writes them into the command buffer |
| Kernel-mode driver | `nvidia.ko` | Initializes the GPU, allocates memory and page tables, sets up channels |
| Bus | PCIe 4.0 ×16 | About 31.5 GB/s in each direction |
| GPU | Front end, distribution unit, 108 SMs, L2 cache, GPU memory | Fetches commands and assigns thread blocks to SMs |

All the state a process has on the GPU, including the page tables of its GPU memory, its channels and its loaded kernels, is called a context. In the words of the Ampere manuals, a context is “a virtualization of the GPU for a particular software application”. When several processes share a GPU, each has its own context and cannot see the others' memory.

## A CUDA program

The program below adds two vectors of length $2^{20}$; the sections that follow take its steps apart one by one:

```cuda
__global__ void add(const float *x, const float *y,
                    float *z, int n) {
    // this thread's index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *hx, *hy, *hz, *x, *y, *z;
    // pinned host memory
    cudaMallocHost(&hx, bytes);
    cudaMallocHost(&hy, bytes);
    cudaMallocHost(&hz, bytes);
    // GPU memory
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&z, bytes);
    /* fill hx and hy */
    cudaMemcpy(x, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, bytes, cudaMemcpyHostToDevice);
    // 4096 blocks of 256 threads
    add<<<(n + 255) / 256, 256>>>(x, y, z, n);
    // follows the kernel; returns when done
    cudaMemcpy(hz, z, bytes, cudaMemcpyDeviceToHost);
}
```

`__global__` marks the kernel. `blockIdx.x * blockDim.x + threadIdx.x` is the $i = 256b + t$ of the [[kernel]] entry. The block count is rounded up, so there may be more threads than elements, and `if (i < n)` stops the threads that would go out of bounds. The three `cudaMemcpy` calls and the launch all go into the default stream and run in order.

## Compilation: from CUDA C++ to PTX and SASS

nvcc splits a source file in two, handing the host code to an ordinary C++ compiler. The device code is first translated into [[PTX]], which the assembler ptxas then translates into the [[SASS]] of a particular GPU generation, packaged as a cubin. The PTX and the cubin are bundled into a fatbinary embedded in the executable. At run time, if no cubin matches the current GPU, the driver compiles the PTX into machine instructions just in time.

```bash
# same as -arch=compute_80 -code=sm_80,compute_80
nvcc -arch=sm_80 add.cu -o add
# output PTX only
nvcc -arch=sm_80 -ptx add.cu
# show the SASS in the executable
cuobjdump -sass add
```

The PTX of `add` follows, without the file header; register numbering and instruction order vary slightly between versions of nvcc:

```ptx
.visible .entry _Z3addPKfS0_Pfi(
    .param .u64 x, .param .u64 y, .param .u64 z, .param .u32 n)
{
    ld.param.u64        %rd1, [x];
    ld.param.u64        %rd2, [y];
    ld.param.u64        %rd3, [z];
    ld.param.u32        %r2, [n];
    mov.u32             %r3, %ctaid.x;       // block index
    mov.u32             %r4, %ntid.x;        // threads per block
    mov.u32             %r5, %tid.x;         // thread index
    mad.lo.s32          %r1, %r3, %r4, %r5;  // i
    setp.ge.s32         %p1, %r1, %r2;
    @%p1 bra            DONE;                // done if i >= n
    cvta.to.global.u64  %rd4, %rd1;
    mul.wide.s32        %rd5, %r1, 4;        // byte offset
    add.s64             %rd6, %rd4, %rd5;
    cvta.to.global.u64  %rd7, %rd2;
    add.s64             %rd8, %rd7, %rd5;
    ld.global.f32       %f1, [%rd6];
    ld.global.f32       %f2, [%rd8];
    add.f32             %f3, %f1, %f2;
    cvta.to.global.u64  %rd9, %rd3;
    add.s64             %rd10, %rd9, %rd5;
    st.global.f32       [%rd10], %f3;
DONE:
    ret;
}
```

On the A100 it typically assembles into the following SASS, shown without addresses, the scheduling information attached to each instruction, or a few auxiliary instructions:

```sass
S2R        R6, SR_CTAID.X             // block index
S2R        R3, SR_TID.X               // thread index
IMAD       R6, R6, c[0x0][0x0], R3    // i
ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT
@P0 EXIT                              // done if i >= n
MOV        R7, 0x4
IMAD.WIDE  R2, R6, R7, c[0x0][0x160]  // &x[i]
IMAD.WIDE  R4, R6, R7, c[0x0][0x168]  // &y[i]
LDG.E      R2, [R2.64]
LDG.E      R4, [R4.64]
IMAD.WIDE  R6, R6, R7, c[0x0][0x170]  // &z[i]
FADD       R9, R2, R4
STG.E      [R6.64], R9
EXIT
```

`c[0x0][…]` is bank 0 of the constant memory, which holds the block dimensions and the kernel arguments, written as part of the launch.

## Programming interfaces

Host code mainly calls the following runtime functions:

| Function | Effect |
|---|---|
| `cudaMalloc`, `cudaFree` | Allocate and free GPU memory, returning a virtual address on the GPU |
| `cudaMallocHost`, `cudaFreeHost` | Allocate and free pinned host memory |
| `cudaMemcpy`, `cudaMemcpyAsync` | Copy between host and GPU or within the GPU; the latter is placed in a stream and runs asynchronously |
| `cudaMemset` | Set every byte of a range of GPU memory to one value |
| `kernel<<<grid, block, smem, stream>>>` | Launch a kernel: blocks, threads per block, bytes of dynamic shared memory, stream |
| `cudaStreamCreate`, `cudaStreamSynchronize` | Create a stream; wait for the work in a stream to finish |
| `cudaEventRecord`, `cudaStreamWaitEvent` | Place a marker in a stream; make another stream wait for that marker |
| `cudaEventElapsedTime` | Time elapsed between two markers |
| `cudaDeviceSynchronize` | Wait for all work on the device to finish |
| `cudaGraphLaunch` | Replay a recorded CUDA graph |
| `cudaGetDeviceProperties` | Query the SM count, memory size, compute capability and more |

The runtime is built on the driver interface. The lower-level driver interface requires calling `cuInit` and obtaining a context, then loading a cubin or PTX with `cuModuleLoad`, getting the kernel with `cuModuleGetFunction` and launching it with `cuLaunchKernel`.

Every function returns an error code. A kernel launch returns nothing and must be checked with `cudaGetLastError`. Errors on the GPU are reported asynchronously and often surface only at some later call. For debugging, setting the environment variable `CUDA_LAUNCH_BLOCKING=1` makes every launch synchronous, so an error stops at the call that caused it.

## How the driver hands commands to the GPU

The terms in this section come from the Ampere manuals. The process has two stages: setting up a channel goes through the operating system, and every submission after that happens entirely in user mode.

The kernel-mode driver sets up a channel once:

- It allocates the memory a channel needs: the ring buffer GPFIFO, the pushbuffer that holds commands, the control area USERD and an instance block.
- The instance block records the channel's state, including the root address of the GPU page tables; the channels of one context point to the same page tables.
- It registers the channel in the runlist, the list of channels the GPU serves in turn.
- It maps USERD and the doorbell register into the process's address space, so the user-mode driver can read and write them directly from then on.

The user-mode driver performs each submission:

1. It writes commands into a segment of the pushbuffer. Each command is called a method and consists of an address word and a data word: the address says what to do, and the data is the operand.
2. It writes one 8-byte entry into the GPFIFO, recording the address and length of that segment.
3. It updates the tail pointer GP_PUT in USERD.
4. It writes the channel number into the doorbell register NV_USERMODE_NOTIFY_CHANNEL_PENDING.

On the GPU side, a unit called the PBDMA reads the GPFIFO and the pushbuffer by DMA and unpacks the commands into address–data pairs. Methods that belong to it, such as semaphores, it executes on the spot; the rest it forwards by subchannel to the engines, the compute engine running kernels and the copy engines moving data.

Completion is signaled by semaphores. The driver appends a semaphore-release method after a series of commands. When the GPU reaches it, it writes a value into memory. `cudaStreamSynchronize` waits for exactly this value; when needed, the GPU can also raise an interrupt at the same time.

## Kernel launch: from the launch descriptor to the SMs

To launch a kernel, the driver first writes two things into GPU memory: the kernel arguments, placed in a constant buffer, and a launch descriptor, which the manuals call a QMD. The Ampere QMD includes these fields:

| QMD field | Meaning | Counterpart in CUDA |
|---|---|---|
| `CTA_RASTER_WIDTH`, `_HEIGHT`, `_DEPTH` | The three dimensions of the grid | `gridDim` |
| `CTA_THREAD_DIMENSION0` to `2` | The three dimensions of a thread block | `blockDim` |
| `SHARED_MEMORY_SIZE` | Shared memory per block | Static plus dynamic shared memory |
| `REGISTER_COUNT_V` | Registers per thread | Decided by ptxas |
| `BARRIER_COUNT` | Barriers used per block | Decided by the compiler |
| `PROGRAM_ADDRESS_LOWER`, `_UPPER` | Address of the kernel's machine code in GPU memory | Fixed when the module is loaded |
| `CONSTANT_BUFFER_ADDR_LOWER(i)` and others | Address of constant buffer $i$ | Where the arguments are |

The driver then issues launch methods to the compute class AMPERE_COMPUTE_A. The data field of `SEND_PCAS_A` is named `QMD_ADDRESS_SHIFTED8`, the QMD address shifted right by 8 bits; the action of `SEND_SIGNALING_PCAS2_B` can be `SCHEDULE`, which queues the work for scheduling.

In the manuals, the compute engine contains three parts: the front end FE, the scheduler SKED and the work distributor CWD. The front end receives the methods, and in the end the CWD sends the thread blocks one by one to the SMs. The A100's SMs are grouped at two levels: 7 GPCs with 54 TPCs in total, each TPC holding 2 SMs, 108 in all. A thread block is sent to an SM only when that SM has enough registers, shared memory, warp slots and block slots. When a block finishes, the freed resources immediately take the next block.

## Inside the SM: issuing and executing warps

Once a thread block is resident, its threads are grouped 32 at a time into warps, which are spread over the SM's 4 partitions. Each A100 partition has:

- one warp scheduler and one dispatch unit, plus an L0 instruction cache;
- a 64 KB register file, that is, 16384 registers of 32 bits, shared by the threads resident in the partition;
- 16 single-precision units, 16 integer units, 8 double-precision units and one Tensor Core.

The 4 partitions share 192 KB of combined L1 cache and shared memory, of which up to 164 KB can be configured as shared memory.

Every cycle, the scheduler picks from its partition a warp that can issue: the operands of its next instruction are ready and it is not waiting at a barrier. It then dispatches that instruction to the appropriate units. A partition has only 16 single-precision units, so an `FFMA` for 32 threads occupies them for two cycles. Memory instructions go to the load–store units, which check L1 and shared memory first, then the L2 cache, and finally GPU memory, a round trip of hundreds of cycles. Meanwhile the scheduler issues other warps, and as long as enough warps are resident, the arithmetic units never sit idle.

These figures give the peak directly. Single precision: $108 \times 4 \times 16 \times 2 \times 1.41 \times 10^9 \approx 1.95 \times 10^{13}$, or 19.5 TFLOPS. Each Tensor Core completes 256 16-bit multiply–adds per cycle: $108 \times 4 \times 256 \times 2 \times 1.41 \times 10^9 \approx 3.12 \times 10^{14}$, or 312 TFLOPS, matching the specification.

## Common instructions

The opcodes of [[SASS]] and their meanings are listed in CUDA's “CUDA Binary Utilities” documentation. The common classes and their correspondence to [[PTX]] are as follows:

| SASS | PTX | Effect |
|---|---|---|
| `FFMA`, `FADD`, `FMUL` | `fma.rn.f32`, `add.f32`, `mul.f32` | Single-precision multiply–add, add, multiply |
| `IMAD`, `ISETP` | `mad.lo.s32`, `setp` | Integer multiply–add; compare and set a predicate |
| `MUFU` | `ex2.approx.f32`, `rsqrt.approx.f32` and others | Special functions such as exponentials and reciprocal square roots |
| `HMMA` | `mma.sync` | Matrix multiply–accumulate on the Tensor Cores |
| `LDG`, `STG` | `ld.global`, `st.global` | Load from and store to GPU memory |
| `LDS`, `STS` | `ld.shared`, `st.shared` | Load from and store to shared memory |
| `LDGSTS` | `cp.async` | Asynchronous copy from GPU memory into shared memory |
| `LDC` | `ld.const`, `ld.param` | Load from constant memory, such as kernel arguments |
| `ATOM`, `RED` | `atom`, `red` | Atomic operations; reductions that write back without returning a result |
| `BAR` | `bar.sync` | Barrier within a thread block, that is, `__syncthreads()` |
| `SHFL` | `shfl.sync` | Register exchange within a warp |
| `S2R` | Reads of special registers such as `%tid` and `%ctaid` | Read the thread and block indices |
| `BRA`, `EXIT` | `bra`, `ret`, `exit` | Branch; end the thread |

The predicate registers `P0` to `P6` hold the results of comparisons, and prefixing any instruction with `@P0` makes it take effect only in threads where the predicate is true. Short branches therefore need no jump: the compiler turns them into predicated instructions.

Hopper adds further instructions: `HGMMA` lets 4 warps jointly perform one larger matrix multiply–accumulate, and `UTMALDG` has a dedicated unit move a whole tile of a tensor from GPU memory into shared memory.

## Data between host and GPU

In a 64-bit process, the host and the GPU share one virtual address space, which is called unified virtual addressing. The addresses returned by `cudaMalloc` lie in the GPU's page tables; the CPU cannot read or write them directly, but the runtime can tell the direction of a copy from the addresses alone.

Copies are performed by the GPU's copy engines, without occupying the SMs:

- When the source is in pinned memory, the copy engine reads it directly by DMA over PCIe, and `cudaMemcpyAsync` can overlap with computation.
- When the source is in ordinary pageable memory, the driver first copies it into a pinned staging buffer and then hands it to the copy engine, and the copy falls back to synchronous behavior.
- Managed memory allocated with `cudaMallocManaged` is migrated page by page between the two sides by the driver: when one side touches a page that is not local, a page fault occurs and the driver moves the page over.

Example: The program above copies 8 MiB in and 4 MiB out. At 31.5 GB/s in each direction over PCIe, this takes about 0.4 milliseconds in all, while the kernel itself reads and writes only 12 MiB of GPU memory, about 6 microseconds at 2039 GB/s. Moving the data is over sixty times slower than computing on it, so data that has reached GPU memory should stay there as long as possible.

Between GPUs, the A100 also has direct NVLink connections, with much higher bandwidth than PCIe; NCCL uses NVLink whenever it is available.

## Synchronization: waiting at which level

Synchronization comes at four levels, each more expensive than the one inside it:

- Within a warp: the 32 threads execute together, and data can be exchanged directly with `SHFL`, without going through memory.
- Within a thread block: `__syncthreads()` compiles to `BAR`, and the threads of the block continue together only after all have arrived. Threads exchange data through shared memory, with a barrier before and after.
- Between thread blocks: there is no general barrier within one kernel, and the order in which blocks run is not determined. To combine the results of the blocks, one can accumulate atomically with `ATOM`, or end the kernel and launch the next. In the same stream, a later kernel always starts after every block of the earlier one has finished.
- Between host and GPU: through the order of a [[stream]], events and the semaphores described above. `cudaStreamSynchronize` makes the host thread wait until all the work in a stream has finished.

## Tiled matrix multiplication

The kernel below computes $C = AB$ with [[tiling]], drawing on most of the preceding sections: two-dimensional block indices, shared memory, two barriers and coalesced access.

```cuda
#define T 16

__global__ void matmul(const float *A, const float *B,
                       float *C, int n) {
    __shared__ float As[T][T], Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float acc = 0;
    for (int k0 = 0; k0 < n; k0 += T) {
        As[ty][tx] = A[row * n + k0 + tx];  // each loads one
        Bs[ty][tx] = B[(k0 + ty) * n + col];
        __syncthreads();                    // tile loaded
        for (int k = 0; k < T; k++)
            acc += As[ty][k] * Bs[k][tx];   // reuse
        __syncthreads();                    // tile used
    }
    C[row * n + col] = acc;
}

// launch, assuming n is a multiple of T
dim3 block(T, T), grid(n / T, n / T);
matmul<<<grid, block>>>(A, B, C, n);
```

Each thread block computes one $16 \times 16$ tile of $C$ with 256 threads, one element per thread. Each pass of the outer loop handles a stretch of length 16 along $k$:

1. Each of the 256 threads moves one number of $A$ and one of $B$ from GPU memory into shared memory, so two $16 \times 16$ tiles arrive at once. Threads of a warp with adjacent `tx` read adjacent addresses, so the accesses are coalesced.
2. After the first barrier, each thread performs 16 multiply–adds with row `ty` of `As` and column `tx` of `Bs`. Each `As[ty][k]` is read by the 16 threads of its row and `Bs[k][tx]` by the 16 threads of its column, so each number read from GPU memory is used 16 times in shared memory.
3. The second barrier ensures that the next pass does not overwrite the tile while some thread is still reading it.

Each thread block uses only $2 \times 16 \times 16 \times 4 = 2048$ bytes, or 2 KB, of shared memory.

Example: For $n = 4096$, the untiled version reads $2n^3 \approx 1.37 \times 10^{11}$ numbers from GPU memory and the tiled one $2n^3 / 16 \approx 8.6 \times 10^9$. If all of these reads went to GPU memory, the arithmetic intensity would be $2n^3$ operations over $4 \times 2n^3 / 16$ bytes, that is, 4 operations per byte. This is still below the single-precision critical value of about 10, so real matrix multiplication goes further:

- each thread computes a small patch of $C$, such as $8 \times 8$, keeping fragments of $A$ and $B$ in registers for one more level of reuse;
- `cp.async` prefetches the next tile while the current one is computed;
- the Tensor Cores' `mma` instruction replaces individual multiply–adds, completing a small matrix multiplication in one instruction.

The kernels in cuBLAS and CUTLASS are built from exactly such layered tiling.

## Tools

- `nvidia-smi`: shows each GPU's memory use, utilization, temperature and driver version.
- Nsight Systems (command `nsys`): draws a timeline of every API call, copy and kernel, for finding gaps and serialization.
- Nsight Compute (command `ncu`): profiles a single kernel, reporting occupancy, memory throughput, the reasons warps stall and other metrics.
- `cuobjdump` and `nvdisasm`: show the PTX and SASS in an executable or a cubin.
- `compute-sanitizer`: checks for out-of-bounds accesses, races and reads of uninitialized memory.
