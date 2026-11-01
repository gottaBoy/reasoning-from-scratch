# Triton IR：全称、编译流程与实际检查

## 1. IR 全称是什么

**IR = Intermediate Representation，中间表示。**

它是编译器用来表达程序的一种形式，既不完全等同于你写的 Python，
也不一定是 GPU 最终执行的机器码。

可以把它想成同一个任务逐步细化：

```text
我要把两列数相加
    ↓
哪些元素一起处理、怎样读取、尾部怎样屏蔽
    ↓
怎样分配到 GPU 的线程/warp，使用什么布局
    ↓
生成目标平台的低层操作和指令
```

“中间表示”是**代码/计算过程的表示**，不是模型 forward 中产生的
hidden states、Q/K/V 等激活张量；不要看到 intermediate 就把两者混在一起。

## 2. 当前 NVIDIA 后端的简化流程

本机已检查 Triton `3.6.0` NVIDIA 后端的 `add_stages`：

```text
Triton Python DSL / @triton.jit
    ↓ 解析、特化、编译
TTIR
    ↓ GPU 布局与相关变换
TTGIR
    ↓ lowering
LLVM IR
    ↓ NVIDIA target code generation
PTX
    ↓ ptxas
cubin
    ↓ 加载并执行
GPU
```

每个箭头可能包含很多 passes；这不是“每一步只有一个优化”。
阶段名和检查 API 随版本/后端可能变化。AMD 等其他后端不是同一套 PTX/cubin 产物。

| 名称 | 含义 | 可以先关注什么 |
| --- | --- | --- |
| `TTIR` | Triton IR | 张量块、load/store、算术、Mask、循环等操作 |
| `TTGIR` | TritonGPU IR | GPU 相关布局/分布信息，以及后续优化后的操作 |
| `LLVM IR`，导出键 `llir` | LLVM 的中间表示 | 更低层的控制流、指针、目标相关 intrinsics |
| `PTX` | Parallel Thread Execution，NVIDIA 的虚拟 ISA 表示 | 目标架构、加载/存储、算术、谓词等虚拟指令 |
| `cubin` | CUDA binary | 本路径由 ptxas 生成的目标二进制；不是逐行看的文本 IR |
| `SASS` | NVIDIA GPU 原生指令汇编表示 | 可用适用的反汇编工具检查二进制，不等于 PTX |

TTIR/TTGIR 使用基于 MLIR 的表示。MLIR 是 Multi-Level Intermediate Representation；
理解其多层级表达即可，现在不要求独立开发 MLIR pass。

## 3. 为什么不从 Python 一步到机器码

因为不同阶段解决的问题不同：

- 高层适合保留 tile、张量与 Mask 的语义，做相关分析和变换。
- GPU 布局阶段可以决定数据如何分布和转换，不只关心数学 shape。
- 更低层逐步明确指针、控制流、硬件相关操作。
- 最后结合目标 GPU 和工具链生成二进制。

分层有利于复用优化与后端、定位编译问题。并不是看完一份 IR 就能判断全部性能。

### Shape 与 layout 不是一回事

```text
shape：这块逻辑上有多少行、多少列
layout：这些元素怎样分布、由哪些线程/warp处理
```

同一个 `128×32` 逻辑块，可以有不同布局和访存/计算行为。
`view/transpose` 的张量逻辑操作与 GPU layout 变换也不能不加区分地混称。
学习时先跟踪一条 load 的 offset/Mask，再逐步看分布信息，不急着读全部类型注解。

## 4. 一个 Vector Add 怎么穿过这些层

课程 kernel：

```python
index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = index < N
value = tl.load(X + index, mask, 0) + tl.load(Y + index, mask, 0)
tl.store(OUT + index, value, mask)
```

设 N=1003、BLOCK=256：

```text
grid = ceil(1003/256) = 4
前三块各256个有效元素
最后一块位置768..1023，仅768..1002有效，共235个
剩余21个位置由Mask屏蔽
```

先回答“哪些元素允许读写”，再观察它在 TTIR/TTGIR/PTX 中怎样表达。
编译器可能变换或折叠代码，不保证源码每一行都对应一条固定指令。

`num_warps=4` 是编译/执行配置之一，不代表一个 program 只有4个元素；
BLOCK 元素数与硬件线程数量不是相同概念。

## 5. 在当前 Jupyter 中查看真实产物

打开：

```text
learning/day17_triton_ir.ipynb
```

流程：

1. 运行初始化，确认使用支持 CUDA 的内核。
2. 启动复用课程 kernel 的 Vector Add，并先检查输出正确性。
3. 从实际返回的 `CompiledKernel.asm` 读取可用阶段。
4. 查看 TTIR、TTGIR、LLVM IR、PTX 的前16行。
5. 在生成的日志目录查看完整文件和 `report.json`。

Notebook 每次用新目录，避免覆盖上次结果。
也可以在 Jupyter Terminal 执行：

```bash
cd /workspace
python -m learning.triton_ir --size 1003 --block 256 \
  --output-dir logs/triton-ir-my-first-run
```

目录已存在时会报错，换新名称即可；不要为重跑直接删除旧笔记。

输出文件：

```text
vector_add.ttir
vector_add.ttgir
vector_add.llir
vector_add.ptx
vector_add.cubin
report.json
```

具体阶段以报告的 `available_asm_keys` 为准。cubin 是二进制，不用文本编辑器
把它当乱码文件分析。源码：[triton_ir.py](../learning/triton_ir.py)。

### 本机这次产物中的具体例子

在 GB10 / Triton 3.6.0 上，实验已经实际得到：

```text
asm keys: cubin, llir, ptx, source, ttgir, ttir
TTGIR: ttg.target = "cuda:121", ttg.num-warps = 4
layout: sizePerThread=[2], threadsPerWarp=[32], warpsPerCTA=[4]
PTX: .version 8.8 / .target sm_121a
```

你可以先将 layout 理解为本块256个元素怎样分配给4个warp：
每warp32个线程，本例每线程负责2个元素，逻辑覆盖为 `4×32×2=256`。
这是这次编译的实际布局，不是所有Triton算子的固定配置。

数值检查通过，报告为每线程16个寄存器、0字节shared memory；Vector Add不需要
复用shared memory，所以0不表示运行失败。换tile/算子后这些结果可能变化。
PyTorch预编译wheel的架构警告与Triton本次JIT目标也不是同一个东西，
不能由一个警告断言全部Triton kernel必然不兼容。
`sm_121a` 是本次工具链产物，不能把这个cubin直接当成RTX 3090等其他架构的通用二进制。

## 6. 怎样利用 IR，而不被它淹没

初学只检查三件事：

| 问题 | 先看哪里 |
| --- | --- |
| Mask 是否正确、逻辑 tile 大小是否符合预期 | 源码和 TTIR，结合边界数值测试 |
| 数据分布/布局相关操作是否发生变化 | TTGIR |
| 生成了哪些目标虚拟指令、目标标识是什么 | PTX；需更低层再用反汇编/profiler |

改变 BLOCK 前先预测 grid、尾部元素数，再对比 IR 和结果。
文本行数更少不等于更快；寄存器更多不一定更慢；没有 spill 也不保证高吞吐。
判断性能还需要固定 workload 的 Benchmark 和 profiler 证据。

Python 断点通常观察 host 侧变量和 launch，不会自动逐行进入已经编译的
GPU 机器指令。pdb、编译 IR、kernel sanitizer、GPU profiler 是不同层次的工具。

## 7. 面试自测与参考答案

**问：IR 全称？为什么 Triton 有多级 IR？**

答：Intermediate Representation。编译器从张量语义逐步降低到 GPU layout、
低层控制流和目标指令，在合适的表示层做对应变换；不同后端产物可不同。

**问：TTIR 和 TTGIR 的区别？**

答：前者主要保留 Triton 张量/程序操作，后者加入 GPU 相关布局/分布等信息；
具体操作和优化要查看目标版本，不能把文件扩展名当成独立性能保证。

**问：PTX 是最终机器码吗？**

答：不是，它是 NVIDIA 的虚拟 ISA；当前本路径使用 ptxas 生成 cubin，
再由驱动加载。讨论底层原生指令时需区分 PTX 和 SASS。

**问：IR 看起来正确就可以交付优化了吗？**

答：不够，还要输出数值对齐、边界输入、dtype/layout 契约、真实 GPU 执行、
warm-up/同步计时，以及性能适用范围。

**问：这个实验与五个工程方向有什么关系？**

答：它属于算子优化的编译检查。不能代替模型 Cache 对齐、推理系统所有权测试、
HTTP 部署验证或生产并发压测，见 [五方向实践路线](INFERENCE_ENGINEERING_PATH.md)。

## 一手资料

- [Triton 3.6.0 NVIDIA 后端 add_stages](https://github.com/triton-lang/triton/blob/v3.6.0/third_party/nvidia/backend/compiler.py)
- [Triton Compiler 源码](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/compiler/compiler.py)
- [MLIR](https://mlir.llvm.org/)
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
