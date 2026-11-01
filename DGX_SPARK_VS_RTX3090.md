# DGX Spark vs RTX 3090：模型训练与部署选择

本文比较你当前的 **DGX Spark（GB10，128GB 统一内存）** 与
**NVIDIA RTX 3090（24GB GDDR6X）**。结论按大模型训练/推理任务划分，
不是单一的“谁绝对更强”。

70B KV Cache 的具体计算见 [KV Cache 70B 公式](docs/KV_CACHE_70B.md)。

## 1. 硬件差异

| 项目 | 当前 DGX Spark | RTX 3090 |
| --- | --- | --- |
| GPU 架构 | Grace Blackwell，GB10 | Ampere，GA102 |
| 可用 GPU/系统内存 | 128GB coherent unified memory | 24GB 独立 GDDR6X VRAM |
| 官方/规格内存带宽 | 273GB/s | 约 936GB/s（不同厂商板卡规格可能略有差异） |
| CPU | 20-core Arm | 依赖主机 CPU |
| 功耗/形态 | 约 140W GB10、SFF | 约 350W 级别独立显卡，需较强电源和散热 |
| GPU 代际 | Blackwell，5th-gen Tensor Cores，SM 12.1 | Ampere，成熟 CUDA SM 8.6 |
| 适合的核心优势 | 容量、统一内存、较大模型、低功耗本地开发 | 带宽、成熟库、适合装得下的模型和 kernel benchmark |

DGX Spark 官方规格为 128GB 统一 LPDDR5X、273GB/s 内存带宽；NVIDIA 同时宣传
其 FP4 AI 性能和大模型本地推理/微调能力。citeturn246854search0turn246854search2
RTX 3090 官方页面确认 24GB GDDR6X；936GB/s 是常见 RTX 3090 板卡规格，
应以具体板卡 BIOS/厂商规格为准。citeturn681646search3turn681646search12

**不要把统一内存当作“等于 128GB 高速显存”。** Spark 的容量大很多，
但它的内存带宽明显低于 RTX 3090 的 GDDR6X；模型能否装下和装下后的每 token
速度是两个问题。

## 2. 模型训练

### 2.1 全参数训练

两者都不适合在本地做大模型全参数训练：

- RTX 3090 的 24GB 很快被权重、梯度、optimizer states 和 activation 占满。
- DGX Spark 的 128GB 能容纳更大模型或更长上下文，但统一内存带宽、软件兼容、
  训练 kernel 和 optimizer 内存仍是限制。
- 70B 级别的“可微调/可运行”不等于可以用单机做高效全参数训练。

推荐：**不要把全参数训练作为本月目标**。先做小模型、LoRA/QLoRA、短序列、
小 batch 和可恢复 checkpoint。

### 2.2 LoRA/QLoRA 微调

| 场景 | 更适合 |
| --- | --- |
| 1B–7B，FP16/BF16，短上下文 | RTX 3090 通常更快，24GB 常够用 |
| 7B–14B，4-bit + LoRA | 两者都可，取决于 activation、batch 和 sequence length |
| 14B–70B，量化/分片/LoRA | DGX Spark 更有容量优势，但必须实测训练吞吐和内存 |
| 需要快速迭代 CUDA/Triton 资料 | RTX 3090 生态更成熟；Spark 也可，但扩展需核对 SM 12.1 |
| 同时运行桌面/数据处理/模型 | DGX Spark 统一内存和一体化更方便，但要给系统服务留余量 |

关键估算：

```text
可训练性 ≈ 权重 + 激活 + 梯度 + optimizer + 临时 workspace
```

不能只用“模型参数量 × dtype bytes”判断能否训练。

### 2.3 本项目中的训练建议

当前项目的 GRPO/蒸馏 smoke 已在 DGX Spark GPU 上通过，但只是短 sanity run：

- Spark：适合继续做本项目的 Qwen3 训练、Cache、Triton 和大容量实验。
- RTX 3090：如果有第二台机器，适合做同一小实验的带宽、kernel 和 batch 对照。
- 不要直接把 Spark 上的 70B 微调宣传为“单机高效训练”；先记录 batch、
  sequence、dtype、峰值内存、tokens/s 和 loss 曲线。

## 3. 模型部署/推理

### 3.1 模型能否装下

粗略权重大小：

```text
FP16/BF16：参数量 × 2 bytes
INT8：参数量 × 1 byte，再加 scale/metadata
INT4：参数量 × 0.5 byte，再加 scale/metadata
```

还要留出 KV Cache、runtime workspace 和系统内存：

```text
可部署 ≠ 权重刚好放入内存
```

| 模型/场景 | RTX 3090 | DGX Spark |
| --- | --- | --- |
| Qwen3 0.6B / 7B 短上下文 | 能装，通常带宽更有优势 | 能装，容量富余 |
| 13B/14B FP16 | 大多受 24GB 限制 | 更有机会装下，但速度依赖统一内存 |
| 30B–70B 量化 | 需量化/分片，24GB 常受限 | 容量明显占优，需适配 kernel |
| 100B+ 量化 | 单卡通常不现实 | Spark 官方定位支持更大模型推理，但要看具体模型/量化/运行时 |
| 长上下文、高并发 | KV Cache 很快吃满 VRAM | 容量更好，但 273GB/s 带宽仍会成为 decode 瓶颈 |

NVIDIA 对 DGX Spark 的官方定位包括本地大模型推理、最高 200B 级别的配置化
工作流和最高 70B 级别微调场景；这应理解为平台能力/特定软件路径的上限，
不是所有模型、batch 和上下文都能达到。citeturn246854search0turn246854search1

### 3.2 推理速度

**模型装得下时：**

- RTX 3090 的显存带宽约 936GB/s，decode 阶段常受权重/KV 读取限制，
  因此小模型、低并发场景可能更快。
- DGX Spark 的容量更大，能够运行 3090 放不下的模型或更长上下文；
  但对同一个装得下的模型，不应先验声称它比 3090 快。
- Prefill 偏计算，decode 偏内存/Cache 访问；两者应分开 Benchmark。

**模型装不下时：**

- RTX 3090 需要 CPU offload、量化或多卡，实际速度可能大幅下降。
- DGX Spark 可避免部分 offload/分片，整体任务可能反而更快，因为它能先运行。

### 3.3 软件生态

| 项目 | DGX Spark | RTX 3090 |
| --- | --- | --- |
| CUDA/Triton | 支持，但需注意 GB10/SM 12.1 扩展兼容性 | Ampere 生态成熟 |
| 预编译第三方 wheel | 可能缺 ARM64/SM 12.1 版本 | x86_64/CUDA wheel 选择更多 |
| vLLM/SGLang/TensorRT-LLM | 需使用针对 Spark/Blackwell 的兼容版本 | 旧版本和教程更多，但也受 24GB 限制 |
| 调试本项目 | 当前已验证 | 需另建 x86/3090 环境验证 |

本机已观察到 PyTorch wheel 的架构列表提示 `sm_120`，而 GB10 报告 capability
为 12.1；矩阵乘法、SDPA、Triton Vector Add 实测通过，但第三方 CUDA 扩展仍需
逐项验证。这是 Spark 的软件风险，不是理论规格比较。

## 4. 明确结论

### 你当前这套大模型学习路线

**选 DGX Spark 做主环境。**

原因：

1. 你已经有可复现的 Spark 镜像、Qwen3 权重、Jupyter、CUDA/Triton 自检和课程报告。
2. 128GB 统一内存更适合后续长上下文、较大模型、分页 Cache 和多请求模拟。
3. 本项目后半段重点是推理系统容量和 Cache，不只是单个小 kernel 的峰值带宽。

### 如果目标是“相同小模型谁跑得快”

**RTX 3090 更值得作为对照卡。**

它的 24GB 独立 VRAM 和高 GDDR6X 带宽更适合：

- 装得下的 1B–7B 模型；
- CUDA/Triton kernel 性能实验；
- 传统 x86_64 CUDA 软件栈；
- 低延迟小 batch 推理。

但这不是说 3090 能替代 Spark：一旦模型、KV Cache、batch 或上下文超过 24GB，
容量限制会先成为硬瓶颈。

### 训练与部署一句话

```text
小模型、低延迟、带宽和成熟生态：RTX 3090
大模型、长上下文、容量和本地部署：DGX Spark
你的当前课程主机：DGX Spark；3090 适合做第二套性能对照
```

## 5. 必须用实测回答的问题

不要只用规格表决定最终结论。对相同模型、权重、输入和输出预算分别记录：

```text
模型/量化：
prompt tokens：
output tokens：
batch/concurrency：
dtype：
TTFT：
TPOT/ITL：
total tokens/s：
峰值 GPU/统一内存：
功耗/温度：
kernel fallback/warning：
```

至少做三组：

1. 7B 或更小模型、短上下文、batch=1。
2. 同模型、长上下文，观察 KV Cache。
3. 3090 无法容纳、但 Spark 可容纳的量化模型。

## 参考规格

- [NVIDIA DGX Spark 产品规格](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [DGX Spark 硬件规格](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA RTX 3090 官方页面](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/)
