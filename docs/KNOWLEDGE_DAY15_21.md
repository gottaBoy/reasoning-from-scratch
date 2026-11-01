# 第三周关键知识：Day 15–21

返回 [索引](KNOWLEDGE_INDEX.md)。GPU 参考实现：
[triton_ops.py](../learning/kernels/triton_ops.py)；
[CUDA C++](../learning/kernels/vector_reduce.cu)。以下 kernel 均是教学 forward，
不是替代生产库的完整实现。

## Day 15：CUDA 线程层级与访存

**核心概念**：grid、block、thread、warp、global/shared/register、coalescing。
CUDA 的一个 warp 通常有 32 个线程；block 可由多个 warps 组成。
寄存器通常是线程私有，shared memory 在 block 内共享，global memory 跨 kernel 可见。

**公式/直觉**：一维索引 `i=blockIdx.x*blockDim.x+threadIdx.x`，
`grid=ceil(N/block)`。N=257、block=256 需要 2 blocks，尾块仅 1 个有效线程，
但实际启动了整块，其他线程必须正确屏蔽。

**实验**：独立编译 6 种大小的 Vector Add。入口：`day15_course.ipynb`。
观察 N=255/256/257 的边界，不只测整齐长度。

**面试主问**：什么是 coalesced memory access？
**答案要点**：同 warp 相邻线程访问相邻/合适对齐地址，可用较少内存事务服务，
反例是大跨度散读；具体事务取决于架构、访问大小和对齐。
**追问**：Shared memory 一定比 global 快吗？答：取决于复用、bank conflicts、
同步和容量占用，搬运没有复用可能得不偿失。

**误区/验收**：CPU 有 nvcc 不等于容器内有；`nvidia-smi` 也不证明 kernel
包含适用架构。先实际编译运行，编译目标以本机 compute capability 为准。

## Day 16：Reduction、同步与数值

**核心概念**：block-local reduction、树形归约、__syncthreads、warp shuffle。
每轮把活动数减半；block 间不能靠普通 `__syncthreads` 同步。

**直觉**：先各小组汇总，再合并小组总和。尾部无效元素对 sum 填 0，
对 max 填 -inf，补充值由操作决定。

**实验**：当前 CUDA 示例在 shared memory 做 block sums，CPU 合并部分和；
清楚记录这个边界。改变输入幅值，观察浮点加法次序的误差。
入口：`day16_course.ipynb`。

**面试主问**：为什么把 barrier 放在仅部分线程进入的 if 中可能挂起？
**答案要点**：block 级 barrier 需要相应线程一致到达，条件不一致可能死锁；
用无效线程补零但仍参与同步，不让部分线程提前返回。
**追问**：warp shuffle 就不需要边界考虑吗？答：仍需有效 lane/mask 和 warp 范围，
不能在不完整活跃集合里假定所有 lane 有值。

**误区/验收**：数学 sum 相等不保证 FP32 bitwise 相等。写 reference、容差，
说明实现的全局合并发生在 CPU 还是 GPU。

## Day 17：Triton 编程模型与 RMSNorm

**核心概念**：program_id、向量化 offsets、tl.load/store Mask、constexpr、JIT。
一个 program 操作一个 tile，不直接等同于“一个 CUDA thread”。
编译知识补充：[IR（Intermediate Representation）全称与多级编译](TRITON_IR.md)，
配套 `learning/day17_triton_ir.ipynb` 读取本机真实 TTIR/TTGIR/LLVM IR/PTX/cubin。

**公式**：row RMSNorm 的 sum 用有效宽度 N 作除数，padding 不计入均方；
用 FP32 累积，再按输出张量 dtype 存储。

**实验**：Vector Add 尾部与 row width=1/7/127/1003、FP32/FP16/BF16。
入口：`day17_course.ipynb`。wrapper 拒绝非连续布局/梯度请求，避免静默错误。

**面试主问**：为什么 JIT 的第一次调用不能当稳态耗时？
**答案要点**：包含编译、加载、初始化；warm-up 后同步再采样，若 shape/constexpr
变化还可能重新编译。
**追问**：调 BLOCK 一定只影响速度吗？答：若 Mask/reduction 写错也影响正确性；
tile 大小还影响寄存器和占用，必须重新对齐。

**误区/验收**：forward-only 原始 kernel 不会自动生成正确 backward；
要训练需单独 autograd 封装与梯度实现/检查。

## Day 18：稳定 Softmax 与 LayerNorm

**核心概念**：exp 溢出、shift invariance、reduction padding、居中方差。
`softmax(x)=exp(x-max(x))/Σexp(x-max(x))`；减常数不改变数学概率。

**直觉**：先统一分数基准再指数化，避免大数直接爆炸。
LayerNorm 使用 `(x-mean)/sqrt(var+eps)`；RMSNorm 不减 mean。
本课程 LayerNorm kernel 非 affine，不含 gamma/beta。

**实验**：非常数长行、常数行、零行和幅值较大输入。softmax padding 设 -inf，
LayerNorm padding 不计入均值/方差的有效宽度。

**面试主问**：先求 E[x²] 再减 E[x]² 有什么风险？
**答案要点**：高均值低方差输入可能严重消去误差；中心化或更稳定算法更可靠，
不因为简单公式在纸上等价就忽略浮点。
**追问**：mask 一整行如何处理？答：先定义无效行输出约定，再选择算法，不能期待
softmax(-inf,...,-inf) 正常产生概率。

**误区/验收**：不能只看概率和等于 1，还要对齐每个元素和被 Mask 的位置。

## Day 19：MatMul 分块与算术强度

**核心概念**：M/N/K、BM/BN/BK、累积 dtype、Tensor Cores、复用和 compute/IO。
矩阵乘 FLOPs 约 `2MNK`，输入读取量与 tile 重用相关，不等同于逐元素算子。

**直觉**：每次搬来一小片 A 和 B，尽量多次使用，再累加到输出 tile。
K 分块是累加，不是把各块最后覆盖到同一输出。

**实验**：17/33/19 等边界尺寸，tile=16/32/64，数值先对齐再记录时间。
入口：`day19_course.ipynb`。本实现 FP32 使用 IEEE dot 精度，不默认 TF32 近似。

**面试主问**：为什么更大 tile 可能反而慢？
**答案要点**：寄存器/共享内存压力、occupancy、尾部浪费、并行块数和硬件 tile
约束共同影响，不能只看减少循环次数。
**追问**：FP32 与 TF32 可以无条件比较误差吗？答：要写清 precision mode 和容差，
不同模式可能调用不同指令/算法。

**误区/验收**：小矩阵 launch-bound 的结果不能代表大 GEMM 算力；
不要声称本教学 kernel 优于 cuBLAS 全部工作负载。

## Day 20：Attention 正确性与基线

**核心概念**：QK/softmax/PV、prefill vs decode、GQA expansion、mask 与 shape。
本日同时有 PyTorch 分块数学参考和 GPU forward，对照 SDPA 而非自比自。

**公式**：prefill Q length=T，score 为 T×T；单 token decode query length=1，
score 为 1×S。复杂度随不同维度变化，不应都叫同一个“大 T”。

**实验**：T=1/17/65，D=16/32/64，检查 causal weights 与输出误差。
入口：`day20_course.ipynb`。当前 Triton Attention 不支持 Cache、GQA、padding/backward。

**面试主问**：为何先写朴素参考而不是直接优化？
**答案要点**：用清晰 reference 定位布局/Mask/数值问题；否则优化与算法错误混在一起。
**追问**：SDPA 调用了哪个具体 kernel？答：由版本、shape、dtype、硬件与设置决定，
需 profiler 或显式 backend 控制，不凭 API 名称断言。

**误区/验收**：自己的两个实现一致也可能共同出错，需要独立 oracle 和负例。

## Day 21：融合 Online Attention、SwiGLU 与性能证据

**核心概念**：融合避免中间张量写回、Online Softmax、数值累积、tile 资源。
Fused SwiGLU 将 SiLU 和乘法同一 kernel 完成，但不包含 FFN 的三次 GEMM。

**直觉**：中间 score 分块在片上用完即丢；维护 m/l/o 而不是保存全矩阵。
本课程 forward kernel 不覆盖训练 backward，也无生产级 autotune。

**实验**：同一 workload、warm-up、同步，比较 SDPA 与 Triton，
不强求正加速比。本机 tiny shape 中可能更慢，这也是有效结果。
入口：`day21_course.ipynb`。

**面试主问**：优化 kernel 最少需要交付什么？
**答案要点**：输入契约、支持 dtype/layout、误差阈值、边界测试、同步计时、
环境信息、性能适用区间和回退/不支持说明。
**追问**：融合越多越好吗？答：不一定，寄存器占用、调度自由度、复用与维护成本
可能抵消 launch/IO 节约。

**误区/验收**：不能用错误 fast-math/放宽容差换来“更快”；不能把教学版本
称作功能完整、最佳性能的 FlashAttention。
