版本：CUDA 13.4，硬件以 A100（计算能力 8.0）为例

本篇把一个内核从源码到执行的全过程串起来，深度约相当于计算机组成之于处理器。通用的概念见各词条：[[主机与设备]]、[[显卡驱动]]、[[命令缓冲区]]、[[线程块]]、[[线程束]]、[[流多处理器]]、[[PTX]]、[[SASS]] 与[[流]]。硬件的数字取自 NVIDIA 公布的 A100 规格。命令提交的细节取自 NVIDIA 公开的 Ampere 硬件手册（open-gpu-doc），其余取自 CUDA 13.4 的文档。

## 总览

从程序到硬件共有七层，每一层只和相邻的层打交道：

| 层 | 组件 | 作用 |
|---|---|---|
| 应用 | PyTorch，或自己写的 `.cu` 程序 | 调用库，或直接启动内核 |
| 库 | cuBLAS、cuDNN、NCCL | 提供优化好的内核 |
| 运行时 | `libcudart`，接口以 `cuda` 开头 | 管理显存、流与启动，首次调用时自动建立上下文 |
| 用户态驱动 | `libcuda.so`，接口以 `cu` 开头 | 把调用翻译成命令，写进命令缓冲区 |
| 内核态驱动 | `nvidia.ko` | 初始化显卡，分配显存与页表，建立通道 |
| 总线 | PCIe 4.0 ×16 | 每个方向约 31.5 GB/s |
| 显卡 | 前端、分配单元、108 个 SM、二级缓存、显存 | 取回命令，把线程块分给 SM 执行 |

一个进程在显卡上的全部状态，包括显存的页表、各条通道与载入的内核，合称上下文。Ampere 手册的说法是：上下文是「为某个应用程序虚拟出的一块显卡」。多个进程共用显卡时，各有各的上下文，互相看不到对方的显存。

## 一段 CUDA 程序

下面的程序把两个长 $2^{20}$ 的向量相加，后文各节依次拆解其中的每一步：

```cuda
__global__ void add(const float *x, const float *y,
                    float *z, int n) {
    // 本线程的下标
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *hx, *hy, *hz, *x, *y, *z;
    // 锁页的主机内存
    cudaMallocHost(&hx, bytes);
    cudaMallocHost(&hy, bytes);
    cudaMallocHost(&hz, bytes);
    // 显存
    cudaMalloc(&x, bytes);
    cudaMalloc(&y, bytes);
    cudaMalloc(&z, bytes);
    /* 填入 hx 与 hy */
    cudaMemcpy(x, hx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(y, hy, bytes, cudaMemcpyHostToDevice);
    // 4096 个线程块，每块 256 个线程
    add<<<(n + 255) / 256, 256>>>(x, y, z, n);
    // 排在内核之后，拷回完成才返回
    cudaMemcpy(hz, z, bytes, cudaMemcpyDeviceToHost);
}
```

`__global__` 标明内核。`blockIdx.x * blockDim.x + threadIdx.x` 即[[内核]]词条中的 $i = 256b + t$。块数向上取整，线程数可能多于元素数，所以要用 `if (i < n)` 挡住越界的线程。三次 `cudaMemcpy` 与一次启动都放进默认流，按次序执行。

## 编译：从 CUDA C++ 到 PTX 与 SASS

nvcc 把源文件拆成两半，主机代码交给普通的 C++ 编译器。设备代码先译成 [[PTX]]，再由汇编器 ptxas 译成某一代显卡的 [[SASS]]，装进 cubin。PTX 与 cubin 一起打包成 fatbinary，嵌进可执行文件。运行时，若找不到与当前显卡相符的 cubin，驱动就把 PTX 即时编译成机器指令。

```bash
# 相当于 -arch=compute_80 -code=sm_80,compute_80
nvcc -arch=sm_80 add.cu -o add
# 只输出 PTX
nvcc -arch=sm_80 -ptx add.cu
# 查看可执行文件中的 SASS
cuobjdump -sass add
```

`add` 的 PTX 如下，省去了文件头，不同版本的 nvcc 在寄存器编号与指令次序上会略有出入：

```ptx
.visible .entry _Z3addPKfS0_Pfi(
    .param .u64 x, .param .u64 y, .param .u64 z, .param .u32 n)
{
    ld.param.u64        %rd1, [x];
    ld.param.u64        %rd2, [y];
    ld.param.u64        %rd3, [z];
    ld.param.u32        %r2, [n];
    mov.u32             %r3, %ctaid.x;       // 块号
    mov.u32             %r4, %ntid.x;        // 每块的线程数
    mov.u32             %r5, %tid.x;         // 块内的线程号
    mad.lo.s32          %r1, %r3, %r4, %r5;  // i
    setp.ge.s32         %p1, %r1, %r2;
    @%p1 bra            DONE;                // i >= n 即结束
    cvta.to.global.u64  %rd4, %rd1;
    mul.wide.s32        %rd5, %r1, 4;        // 字节偏移
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

在 A100 上，它汇编成的 SASS 典型如下，省去了地址、每条指令附带的调度信息和个别辅助指令：

```sass
S2R        R6, SR_CTAID.X             // 块号
S2R        R3, SR_TID.X               // 线程号
IMAD       R6, R6, c[0x0][0x0], R3    // i
ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT
@P0 EXIT                              // i >= n 即结束
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

`c[0x0][…]` 是常量区的第 0 块，存放块的尺寸与内核的参数，由启动命令一并写入。

## 编程接口

主机代码能调用的，主要是下面这些运行时接口：

| 接口 | 作用 |
|---|---|
| `cudaMalloc`、`cudaFree` | 分配、释放显存，返回显卡上的虚拟地址 |
| `cudaMallocHost`、`cudaFreeHost` | 分配、释放锁页的主机内存 |
| `cudaMemcpy`、`cudaMemcpyAsync` | 在主机与显卡之间或显卡之内拷贝；后者放进流中，异步执行 |
| `cudaMemset` | 把一段显存逐字节置为同一个值 |
| `kernel<<<grid, block, smem, stream>>>` | 启动内核：块数、每块线程数、动态共享内存的字节数、流 |
| `cudaStreamCreate`、`cudaStreamSynchronize` | 建立流；等待流中的工作完成 |
| `cudaEventRecord`、`cudaStreamWaitEvent` | 在流中打下标记；让另一个流等待这个标记 |
| `cudaEventElapsedTime` | 两个标记之间的耗时 |
| `cudaDeviceSynchronize` | 等待设备上的全部工作完成 |
| `cudaGraphLaunch` | 重放录制好的 CUDA 图 |
| `cudaGetDeviceProperties` | 查询 SM 数、显存大小、计算能力等 |

运行时建立在驱动接口之上。驱动接口更底层，要先调用 `cuInit` 并取得上下文，再以 `cuModuleLoad` 载入 cubin 或 PTX，以 `cuModuleGetFunction` 取得内核，以 `cuLaunchKernel` 启动。

每个接口都返回一个错误码。内核启动没有返回值，要用 `cudaGetLastError` 检查。显卡上的错误是异步报告的，往往在其后的某次调用中才出现。调试时设环境变量 `CUDA_LAUNCH_BLOCKING=1`，可以让每次启动都同步执行，错误就停在出错的那次调用上。

## 驱动怎样把命令交给显卡

这一节的名词取自 Ampere 手册。过程分两段：建立通道时要经过操作系统，此后的提交全在用户态完成。

建立通道由内核态驱动一次做完：

- 分配一条通道所需的几块内存：环形队列 GPFIFO、存放命令的 pushbuffer、控制区 USERD 与实例块。
- 实例块记下这条通道的状态，包括显卡页表的根地址；同一上下文的各条通道指向同一张页表。
- 把这条通道登记进 runlist，即显卡轮流服务的通道清单。
- 把 USERD 与门铃寄存器映射进进程的地址空间，此后用户态驱动可以直接读写它们。

每次提交由用户态驱动完成：

1. 把命令写进 pushbuffer 中的一段。每条命令称为方法，由一个地址字和一个数据字组成：地址说明做什么，数据是操作数。
2. 在 GPFIFO 中写入一项，共 8 字节，记下这段命令的地址与长度。
3. 更新 USERD 中的队尾指针 GP_PUT。
4. 把通道号写进门铃寄存器 NV_USERMODE_NOTIFY_CHANNEL_PENDING。

显卡一侧，名为 PBDMA 的单元以 DMA 方式读取 GPFIFO 与 pushbuffer，把命令解开成一对对地址与数据。其中属于它自己的方法（如信号量）当场执行，其余按子通道转给各个引擎：计算引擎执行内核，拷贝引擎搬运数据。

得知完成靠信号量。驱动在一串命令之后加一条释放信号量的方法，显卡执行到它时，把一个值写进内存。`cudaStreamSynchronize` 等的正是这个值；需要时，还可以让显卡同时发一个中断。

## 内核启动：从启动描述到 SM

启动内核时，驱动先在显存中写好两样东西：内核的参数（放进一块常量缓冲区），以及一份启动描述，手册称之为 QMD。Ampere 的 QMD 中有这些字段：

| QMD 字段 | 含义 | 在 CUDA 中的对应 |
|---|---|---|
| `CTA_RASTER_WIDTH`、`_HEIGHT`、`_DEPTH` | 网格的三个维度 | `gridDim` |
| `CTA_THREAD_DIMENSION0` 至 `2` | 线程块的三个维度 | `blockDim` |
| `SHARED_MEMORY_SIZE` | 每块的共享内存 | 静态与动态共享内存之和 |
| `REGISTER_COUNT_V` | 每个线程的寄存器数 | 由 ptxas 决定 |
| `BARRIER_COUNT` | 每块用到的屏障数 | 由编译器决定 |
| `PROGRAM_ADDRESS_LOWER`、`_UPPER` | 内核机器码在显存中的地址 | 载入模块时确定 |
| `CONSTANT_BUFFER_ADDR_LOWER(i)` 等 | 第 $i$ 块常量缓冲区的地址 | 参数所在之处 |

接着驱动向计算类 AMPERE_COMPUTE_A 发出启动方法。`SEND_PCAS_A` 的数据字段名为 `QMD_ADDRESS_SHIFTED8`，即 QMD 的地址右移 8 位；`SEND_SIGNALING_PCAS2_B` 的动作可以取 `SCHEDULE`，即排入调度。

手册中，计算引擎内部有前端 FE、调度单元 SKED 与工作分配单元 CWD 三个部分。前端接收方法，最后由 CWD 把线程块逐个发给 SM。A100 的 SM 按两级分组：7 个 GPC，共 54 个 TPC，每个 TPC 含 2 个 SM，合计 108 个。一个线程块只有在某个 SM 上的寄存器、共享内存、线程束名额与块名额都够用时，才会发过去。一块结束，腾出的资源立刻用来接收下一块。

## SM 内部：线程束的发射与执行

线程块驻留下来，它的线程每 32 个组成一个线程束，分到 SM 的 4 个分区。A100 的每个分区有：

- 一个线程束调度器与一个派发单元，以及一块 L0 指令缓存；
- 64 KB 的寄存器文件，即 16384 个 32 位寄存器，由驻留在本分区的线程分用；
- 16 个单精度单元、16 个整数单元、8 个双精度单元与一个张量核心。

4 个分区共用 192 KB 的 L1 缓存兼共享内存，其中至多 164 KB 可配作共享内存。

每个周期，调度器从本分区的线程束中挑一个能发射的：下一条指令的操作数已经就绪，也没有停在屏障上。然后把这条指令发给相应的单元。本分区只有 16 个单精度单元，一条 32 个线程的 `FFMA` 要占用它们两个周期。访存指令交给访存单元，先查 L1 与共享内存，不命中再查二级缓存，最后到显存，往返数百个周期。这期间调度器改发别的线程束，只要驻留的线程束够多，运算单元就不会闲着。

由这些数字可以直接算出峰值。单精度：$108 \times 4 \times 16 \times 2 \times 1.41 \times 10^9 \approx 1.95 \times 10^{13}$，即 19.5 TFLOPS。张量核心每个周期完成 256 次 16 位乘加：$108 \times 4 \times 256 \times 2 \times 1.41 \times 10^9 \approx 3.12 \times 10^{14}$，即 312 TFLOPS，与规格表相同。

## 常用指令

[[SASS]] 的操作码及其含义见 CUDA 的「CUDA Binary Utilities」文档。常用的几类及其与 [[PTX]] 的对应如下：

| SASS | PTX | 作用 |
|---|---|---|
| `FFMA`、`FADD`、`FMUL` | `fma.rn.f32`、`add.f32`、`mul.f32` | 单精度乘加、加、乘 |
| `IMAD`、`ISETP` | `mad.lo.s32`、`setp` | 整数乘加；比较并设置谓词 |
| `MUFU` | `ex2.approx.f32`、`rsqrt.approx.f32` 等 | 指数、倒数平方根等特殊函数 |
| `HMMA` | `mma.sync` | 在张量核心上做矩阵乘加 |
| `LDG`、`STG` | `ld.global`、`st.global` | 读写显存 |
| `LDS`、`STS` | `ld.shared`、`st.shared` | 读写共享内存 |
| `LDGSTS` | `cp.async` | 从显存异步拷入共享内存 |
| `LDC` | `ld.const`、`ld.param` | 读常量区，如内核参数 |
| `ATOM`、`RED` | `atom`、`red` | 原子操作；只写回、不取回结果的归约 |
| `BAR` | `bar.sync` | 线程块内的屏障，即 `__syncthreads()` |
| `SHFL` | `shfl.sync` | 线程束内交换寄存器 |
| `S2R` | 读 `%tid`、`%ctaid` 等特殊寄存器 | 读出线程号、块号 |
| `BRA`、`EXIT` | `bra`、`ret`、`exit` | 跳转；结束线程 |

谓词寄存器 `P0` 至 `P6` 保存比较的结果，任何指令前加 `@P0` 就只在谓词为真的线程上生效。短小的分支因此不必跳转，编译器把它变成带谓词的指令。

Hopper 又加了几类指令：`HGMMA` 让 4 个线程束共同完成一次更大的矩阵乘加，`UTMALDG` 由专门的单元把一整块张量从显存搬进共享内存。

## 主机与显卡之间的数据

64 位进程中，主机与显卡共用一个虚拟地址空间，称为统一虚拟寻址。`cudaMalloc` 返回的地址落在显卡的页表中，处理器不能直接读写它，但运行时凭地址就能判断一次拷贝的方向。

拷贝由显卡上的拷贝引擎完成，不占用 SM：

- 源在锁页内存中时，拷贝引擎直接以 DMA 方式经 PCIe 读取，`cudaMemcpyAsync` 可以与计算同时进行。
- 源在普通的可换页内存中时，驱动先把它复制到一块锁页的中转区，再交给拷贝引擎，拷贝退化为同步。
- `cudaMallocManaged` 分配的托管内存由驱动按页在两边之间迁移：一方访问不在本地的页时触发缺页，驱动把页搬过来。

例：上面的程序要把 8 MiB 拷入、4 MiB 拷出。按 PCIe 每个方向 31.5 GB/s 计，共约 0.4 毫秒；内核本身只读写 12 MiB 的显存，按 2039 GB/s 计约 6 微秒。搬运比计算慢六十多倍，所以数据一旦进了显存，就应尽量留在那里。

多卡之间，A100 另有 NVLink 直连，带宽比 PCIe 高得多；NCCL 在有 NVLink 时优先用它。

## 同步：在哪一层等待

同步分四层，越往外，代价越大：

- 线程束之内：32 个线程一同执行，交换数据可以直接用 `SHFL`，不必经过内存。
- 线程块之内：`__syncthreads()` 编译成 `BAR`，块内的线程都到达后才一齐继续。线程之间借共享内存交换数据，前后要各有一道屏障。
- 线程块之间：同一个内核里没有通用的屏障，块的执行次序也不确定。要合并各块的结果，可以用 `ATOM` 做原子累加，也可以结束这个内核、再启动下一个。同一个流中，后一个内核总在前一个的所有块结束之后才开始。
- 主机与显卡之间：靠[[流]]的次序、事件，以及上文所说的信号量。`cudaStreamSynchronize` 让主机线程等到流中的工作全部完成。

## 分块的矩阵乘法

下面的内核用[[分块]]计算 $C = AB$，用到前几节的大部分内容：线程块的二维编号、共享内存、两道屏障与合并访存。

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
        As[ty][tx] = A[row * n + k0 + tx];  // 每个线程搬一个数
        Bs[ty][tx] = B[(k0 + ty) * n + col];
        __syncthreads();                    // 等全块搬完
        for (int k = 0; k < T; k++)
            acc += As[ty][k] * Bs[k][tx];   // 反复取用
        __syncthreads();                    // 等全块用完
    }
    C[row * n + col] = acc;
}

// 启动：设 n 是 T 的整数倍
dim3 block(T, T), grid(n / T, n / T);
matmul<<<grid, block>>>(A, B, C, n);
```

每个线程块算 $C$ 的一个 $16 \times 16$ 块，共 256 个线程，每个线程算其中一个元素。外层循环每转一圈，处理 $k$ 方向上长 16 的一段：

1. 256 个线程各从显存搬 $A$ 的一个数与 $B$ 的一个数进共享内存，两个 $16 \times 16$ 的小块就一次搬齐了。同一个线程束中 `tx` 相邻的线程读相邻的地址，访存是合并的。
2. 第一道屏障之后，每个线程用 `As` 的第 `ty` 行与 `Bs` 的第 `tx` 列做 16 次乘加。`As[ty][k]` 被同一行的 16 个线程读取，`Bs[k][tx]` 被同一列的 16 个线程读取，所以从显存读进来的每个数，在共享内存中被用了 16 次。
3. 第二道屏障保证没有线程还在读这一块时，下一圈就把它覆盖了。

每个线程块只用 $2 \times 16 \times 16 \times 4 = 2048$ 字节，即 2 KB 共享内存。

例：$n = 4096$ 时，不分块要从显存读 $2n^3 \approx 1.37 \times 10^{11}$ 个数，分块后为 $2n^3 / 16 \approx 8.6 \times 10^9$ 个。若这些读取全都落到显存，算术强度为 $2n^3$ 次运算除以 $4 \times 2n^3 / 16$ 字节，即 4 次每字节。这仍低于单精度的临界值约 10，所以实际的矩阵乘法还要更进一步：

- 让每个线程算 $C$ 的一小片，如 $8 \times 8$，把 $A$、$B$ 的片段放进寄存器，再复用一次；
- 用 `cp.async` 在计算这一块的同时预取下一块；
- 改用张量核心的 `mma` 指令，一条指令完成一个小矩阵乘法。

cuBLAS 与 CUTLASS 中的内核正是这样层层分块写成的。

## 工具

- `nvidia-smi`：查看各卡的显存占用、利用率、温度与驱动版本。
- Nsight Systems（命令 `nsys`）：画出时间线，列出每次接口调用、拷贝与内核的起止，用来找空隙与串行的地方。
- Nsight Compute（命令 `ncu`）：剖析单个内核，给出占用率、显存吞吐、线程束停顿的原因等指标。
- `cuobjdump` 与 `nvdisasm`：查看可执行文件或 cubin 中的 PTX 与 SASS。
- `compute-sanitizer`：检查越界访问、竞争与未初始化的读。
