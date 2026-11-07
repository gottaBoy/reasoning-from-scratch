# 大模型推理与性能优化：30 天学习计划

本计划面向希望进入大模型推理、算子优化和 AI Infra 方向的学习者。

## 目标与边界

30 天不能达到生产级“精通”。本计划的实际目标是：

- 能读懂 Llama、Qwen、DeepSeek 的核心模型代码。
- 能手写 GQA、RoPE、KV Cache 和简化 Transformer。
- 能编写并测试基础 CUDA/Triton 算子。
- 能解释 FlashAttention、PagedAttention 和 Continuous Batching。
- 能运行、分析并修改 vLLM 相关推理流程。
- 能完成一个带 Benchmark 和技术报告的推理项目。

建议每天投入 4～6 小时，每周学习 6 天、复盘 1 天。每一天都要有
可运行的代码、测试或实验记录，不能只阅读资料。

## 当前进度与执行起点

**当前学习位置：Day 7，请求生命周期与调度地图。**
2026-11-07用户要求完善Day 6教程并继续下一天，详见
[进度记录](docs/LEARNING_PROGRESS.md)和[Day 7学习记录](docs/DAY07_STUDY_LOG.md)。
材料自动准备不改变个人完成状态；旧的 Day 2 启动清单保留作复习。
后续参考实现、每日 Notebook 和验证入口见 [Day 6–30 课程](docs/COURSE_DAY06_30.md)。
**每日知识点、概念、公式、误区、面试主问/追问与掌握标准**见
[30 天知识与面试索引](docs/KNOWLEDGE_INDEX.md)，覆盖 Day 1–30。
公式中的字母、张量轴和配置名称统一查阅
[课程符号与张量形状规范](docs/NOTATION.md)；首次阅读Attention形状时从此处查起。
五个工程方向的衔接见 [推理优化与工程实践](docs/INFERENCE_ENGINEERING_PATH.md)，
算子编译补充见 [Triton IR](docs/TRITON_IR.md)。

本节将后面的 30 天安排与当前 DGX Spark 环境对齐。Day 表示学习日，不是
自然日期；没有通过当天验收时继续补齐，不因为日历翻页就算完成。

| 内容 | 当前状态 | 对学习进度的含义 |
| --- | --- | --- |
| Spark 镜像、CUDA 前后向、Triton JIT | 已实测通过 | 环境准备完成，不必每天重建 |
| Qwen3 生成和小规模推理/训练 smoke | 已实测通过 | 有可运行的参考实现，不等于已经理解模型 |
| 普通测试（含新增课程） | 301 passed / 27 skipped | 现有Spark工作区缓存数据回归；不是个人掌握评分，历史与失败记录另行保留 |
| Day 1 独立复现与笔记 | 待自己验收 | 独立运行一次生成，能找到设备、Cache 和日志 |
| Day 2–4 模型、Shape、Cache | 已讲解并准备实验 | 个人理解仍需用自测题确认 |
| Day 5 架构、MoE、MLA | 本轮讲解结束，用户决定进入下一天 | 独立自测结果尚未登记 |
| Day 6 Online Softmax/FlashAttention | 本轮教程收尾，个人独立验收尚未登记 | 参考实验通过不代替个人掌握 |
| Day 7 请求生命周期与调度 | 当前主任务，学习中 | 从A/B/C调度轨迹和分页映射开始 |
| Day 8–30 的学习产出 | 材料自动准备，个人尚未验收 | 按每日 Notebook 和每周检查点推进 |

运行证据见 [Spark 验证记录](docs/DGX_SPARK_VALIDATION.md)。
不需要重新执行整套 smoke 才能开始学习，也不能因为 smoke 通过就跳过第二周。
剩余 Day 6–30 的统一执行包见
[剩余 25 天执行包](docs/REMAINING_25_DAYS.md)，学习产出入口见
[learning/README.md](learning/README.md)。

Day 1 已完成项、Day 2 的数据流/模块讲解及自测题见
[Day 1–2 学习笔记](docs/DAY01_DAY02_NOTES.md)。其中的参考答案不是个人验收记录。

### 四周如何衔接

| 学习日 | 主线 | 你需要独立完成的产出 |
| --- | --- | --- |
| Day 1–7 | 一周建立知识地图，以 Qwen3 为入口 | 模型数据流图、Shape 表、Cache 对齐实验、机制对比笔记 |
| Day 8–14 | 用 PyTorch 实现模型核心 | RMSNorm、RoPE、Attention、GQA、KV Cache、测试与基线 |
| Day 15–21 | CUDA/Triton 和性能分析 | 至少 3 个正确的算子、可复现计时与瓶颈分析 |
| Day 22–30 | 推理系统与综合项目 | vLLM 阅读记录、分页缓存/调度教学实现、项目报告 |

第一周是阅读和验证现有实现，第二周才是独立重写。例如 Day 4 使用仓库模型
验证 Cache 等价性，Day 12 再为自己的模型实现 Cache；两者不是重复任务。
第一周对 MoE、MLA、FlashAttention、vLLM 要建立概念图，不要求当天实现生产框架。

### Day 2 起步复习（历史清单）

本清单的任务是 **Day 2：看懂 Qwen3 的一次前向传播**。如果还没有独立运行过，
先用 20～30 分钟完成 Day 1 的生成命令并记录输出，然后进入 Day 2。

1. 阅读 `reasoning_from_scratch/qwen3.py`，顺序为
   `QWEN_CONFIG_06_B → Qwen3Model.forward → TransformerBlock.forward →
   GroupedQueryAttention.forward → FeedForward.forward`。
2. 画出 `token → embedding → 28 个 Transformer Block → final norm →
   output projection → logits` 的数据流。
3. 为每个模块写下输入、输出、作用和残差连接的位置，遇到不懂的算子先记问题。
4. 对照配置写出 `n_heads=16`、`n_kv_groups=8`、`head_dim=128`、
   `emb_dim=1024` 分别控制什么；具体 Shape 验证安排在 Day 3。
5. 写一份 Day 2 笔记，至少包含数据流图、模块表和三个尚未解决的问题。

今天不以“读完全部文件”为目标。验收是：不用照着代码，也能解释一次前向传播
经过哪些模块，以及 logits 和最终生成 token 的区别。

### Day 3–7安排与复习

| 学习日 | 具体行动 | 当天验收 |
| --- | --- | --- |
| Day 3 | 对已有 Attention 打印/断点观察 `view`、`transpose`、KV head 扩展前后的形状 | 用 `B=1,T=8` 完成 Q/K/V、scores、context、logits 表，并解释 `128` 为什么不一定等于 `1024/16` |
| Day 4 | 同一组固定输入，一次输入 8 token，对比 Cache prefill 7 token 后再输入 1 token | 比较最后位置 logits，而非采样文本；记录误差、位置偏移和每层 K/V 形状 |
| Day 5 | 完成架构差异笔记和 MoE/MLA 数据流图 | 比较时注明具体模型版本；说清各机制改变的是哪部分数据流或资源开销 |
| Day 6 | 先做Online Softmax手算和Notebook，再选读 `docker/check_triton.py` 中program、offset、mask | 能解释m/l/o合并、IO收益与边界；GPU索引作为后续补充 |
| Day 7 | 整理请求生命周期与分页缓存/调度概念图，完成第一周复盘 | 能把 GQA、KV Cache、FlashAttention、PagedAttention、Continuous Batching 的职责分开说明 |

Day 4 先使用小配置、FP32 和固定权重建立数值参考，每次独立实验使用新的 Cache
并重置模型的位置状态。再观察 BF16 的误差，不直接要求低精度输出逐元素完全相等。

Day 5 的模型组件对比表沿用 Day 2 的草稿补齐，不要求 Day 2 一天内吃透三个模型
家族。本仓库没有实现的模块要明确标为外部阅读项。

### 每天如何安排

| 时间 | 行动 |
| --- | --- |
| 45 分钟 | 阅读当天涉及的源码和原理，先自己推导 |
| 90 分钟 | 完成一个小实验或实现，只改变一个关键变量 |
| 45 分钟 | 正确性测试，保存输入、配置与结果；需要计时时另做 Benchmark |
| 30 分钟 | 写实验笔记，解释错误和未解决的问题 |
| 30–90 分钟 | 修复问题、复述原理或补充边界测试 |

可在 Jupyter 的 `/workspace/learning/` 下按学习日保存笔记和 notebook，
例如 `day02.md`、`day03_shapes.ipynb`、`day04_cache.ipynb`。它们保存在 Spark
volume 内，不会自动成为宿主机 Git 改动；阶段结束时按
[Spark 教程的备份说明](DGX_SPARK.md) 导出需要版本管理的学习产出。

例如，在 Jupyter 服务容器已创建、`/workspace/learning/` 已有笔记后，可在宿主机
仓库根目录导出：

```bash
docker compose -f docker-compose.spark.yml cp reasoning:/workspace/learning ./learning
```

导出前检查宿主机是否已有同名笔记，避免覆盖自己的旧版本；随后查看 Git diff，
只提交源码、测试和笔记，不提交模型、缓存或训练产物。

### 每周检查点

- Day 7：能解释全貌并展示 Shape/Cache 实验，才开始独立实现模型模块。
- Day 14：自己的模型通过参考对齐和 Cache 一致性测试，才进入算子优化。
- Day 21：算子正确、有计时证据和瓶颈结论，才开始组合成推理系统。
- Day 30：能从空环境重现项目，并区分实测功能、模拟功能与未实现功能。

AI 可以帮助解释、检查推导和审查测试，但“AI 写出了代码”不能直接算作当天验收。
先自己回答问题，再与源码和实验结果核对。

## 使用本仓库

**本机有 DGX Spark 时，先执行 [DGX Spark 实战教程](DGX_SPARK.md)。**
该教程使用 `docker-compose.spark.yml`，实际验证 CUDA 矩阵乘法、Attention
反向传播和逐阶段 smoke，并记录 [已验证范围](docs/DGX_SPARK_VALIDATION.md)。
下面的 CPU 命令保留作对照，不是 Spark 的首选启动方式。

工作目录：

```bash
cd /home/my/workspace/llm/reasoning-from-scratch
```

如果宿主机已经安装了兼容版本的 `uv`，可以先确认环境：

```bash
uv run python -c \
  "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

如果不希望安装本地 Python 依赖，直接使用 Docker。仓库已移除会导致不同
`uv` 版本行为不一致的 `exclude-newer = "7 days"` 配置。推荐先执行：

```bash
docker compose -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

docker compose -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml run --rm reasoning \
  python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

CPU 环境运行第一次最小生成：

```bash
docker compose -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml run --rm reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cpu \
  --max_new_tokens 16 \
  --prompt "What is attention in a Transformer?"
```

有 NVIDIA GPU 和 NVIDIA Container Toolkit 时，先检查 GPU：

```bash
docker compose \
  -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml \
  -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.gpu.yml \
  run --rm reasoning \
  python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

GPU 环境运行第一次最小生成：

```bash
docker compose \
  -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml \
  -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.gpu.yml \
  run --rm reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cuda \
  --max_new_tokens 16 \
  --prompt "What is attention in a Transformer?"
```

第一次运行会下载 Qwen3 0.6B 权重和 tokenizer。模型、源码副本和实验产物
保存在 Docker named volume `reasoning_workspace` 中，后续运行会复用。
使用 `docker compose down -v` 会删除该 volume 以及其中的学习进度。

CPU Compose 默认使用 PyPI 版 PyTorch，避免在 ARM64 CPU 环境中误下载约
505 MiB 的 CUDA wheel。需要 GPU 时，使用 GPU Compose 覆盖文件，它会恢复
项目锁定的 CUDA PyTorch 来源：

```bash
# CPU：使用 PyPI PyTorch
docker compose -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml build

# GPU：使用 CUDA PyTorch（需要 NVIDIA Container Toolkit）
docker compose \
  -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml \
  -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.gpu.yml build
```

如果构建日志在 Dockerfile 的 `pip install uv` 或 `uv sync` 步骤出现
`ReadTimeoutError`，通常是容器网络下载 PyPI 超时，不是代码错误。当前
Compose 已为构建启用 Host 网络，并提高 pip/uv 的超时和重试次数。网络受限时，
同时把 pip 和 uv 的索引指向当前网络可访问的镜像，例如：

```bash
docker compose -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

如果已经成功安装 `uv` 但后续 `uv sync` 下载 PyTorch 超时，说明锁定的
PyTorch wheel 地址仍然无法稳定访问；重试构建并检查 Docker/宿主机网络或代理。
已经完成的 pip 安装步骤会被 Docker 缓存复用。

如果运行 `docker compose run` 时出现
`all predefined address pools have been fully subnetted`，这是 Docker 主机的
默认网络地址池耗尽，不是镜像问题。不要直接执行 `docker network prune`，因为
它可能影响其他项目；可以使用 Host 网络直接运行当前镜像：

```bash
# CPU 环境检查
docker run --rm --network host \
  -v reasoning_workspace:/workspace \
  -w /workspace reasoning-from-scratch:local \
  python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"

# CPU 环境第一次生成
docker run --rm --network host \
  -v reasoning_workspace:/workspace \
  -w /workspace reasoning-from-scratch:local \
  python ch02/05_use_model/generate_simple.py \
  --device cpu --max_new_tokens 16 \
  --prompt "What is attention in a Transformer?"
```

GPU 环境必须先构建 GPU 镜像，不能只给 `:local` CPU 镜像增加 `--gpus all`。
通用 GPU 镜像为 `reasoning-from-scratch:gpu`，Spark 镜像为
`reasoning-from-scratch:spark`；Spark 的默认数据 volume 为
`reasoning_spark_workspace`。构建和启动均使用对应的 Compose 配置。

学习期间先使用短输入和较小的 `max_new_tokens`，不要一开始运行完整评测或
长时间训练。

## 固定学习闭环

每天按下面的顺序执行；第一周主要用现有实现做诊断实验，第二周起再独立重写：

1. 阅读 30～60 分钟，明确当天要回答的问题。
2. 第一周完成最小验证实验；第二周起手写最小实现，不直接复制完整框架。
3. 用随机输入或仓库测试验证正确性。
4. 用 Benchmark 记录延迟、吞吐量、显存和误差。
5. 写下失败原因、结论和下一步。

每周只选择一个主问题，避免同时修改多个变量。

## 第 1 周：建立完整知识地图

第一周的目标是“知道每项技术是什么、解决什么问题、在哪里实现”，而不是
立刻实现所有高级功能。

### Day 1：环境与 Qwen3 推理

DGX Spark：

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python docker/check_environment.py --device cuda

docker compose -f docker-compose.spark.yml run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cuda --max_new_tokens 32 --prompt "The capital of France is"
```

执行最小生成。推荐使用 Docker 命令，宿主机有兼容 `uv` 时也可以使用本地命令。

Docker CPU：

```bash
docker compose -f /home/my/workspace/llm/reasoning-from-scratch/docker-compose.yml run --rm reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cpu \
  --max_new_tokens 16 \
  --prompt "What is attention in a Transformer?"
```

本地 `uv`：

```bash
uv run python ch02/05_use_model/generate_simple.py \
  --device cpu \
  --max_new_tokens 16 \
  --prompt "What is attention in a Transformer?"
```

有 GPU 时，将 Docker 命令中的 `docker-compose.yml` 替换为基础 compose 文件
加上 `docker-compose.gpu.yml`，并将 `--device cpu` 改为 `--device cuda`。

记录：

- Git commit：`git rev-parse HEAD`
- Python、PyTorch 版本和设备
- 输入文本和 `max_new_tokens`
- 是否成功使用 KV Cache
- 总耗时和生成 token 数

阅读：

- `reasoning_from_scratch/qwen3.py`
- `ch02/05_use_model/generate_simple.py`

当天产出：成功生成一次文本，并写下从 token 到 logits 的流程。

### Day 2：Transformer 和模型架构

配套讲解：[Day 1–2 学习笔记](docs/DAY01_DAY02_NOTES.md)，先分清“整体数据流”
与“单个模块职责”，再按下面的顺序阅读源码。

学习：

- Embedding、Attention、FFN、Residual、Norm
- Llama、Qwen、DeepSeek 的主要组件
- Pre-Norm、RMSNorm、SwiGLU、RoPE

阅读：

- `QWEN_CONFIG_06_B`
- `Qwen3Model.forward`
- `TransformerBlock.forward`
- `FeedForward.forward`

当天产出：一张 Qwen3 前向数据流图和模块输入/输出表；起草模型组件对比表，
在 Day 5 再补齐具体 Llama/Qwen/DeepSeek 版本的差异。

### Day 3：Attention、MHA、MQA、GQA

可执行教程：[Day 3 调试与 Shape 验证](docs/DAY03_SHAPES_DEBUGGING.md)。
先完成 Day 2 的模块职责复述，再执行 hooks、断点和因果性验证。
已经使用 Jupyter 时，可直接打开
[Day 3 交互式 Notebook](learning/day03_debug.ipynb)，不必创建新容器。
RoPE 和 FFN 门控的补充材料见
[形象解释](docs/ROPE_SWIGLU_INTUITION.md) 与
[旋转、SiLU 和小 FFN 实验](learning/day03_rope_swiglu.ipynb)。

```bash
docker compose -f docker-compose.spark.yml run --rm reasoning \
  python learning/day03_shapes.py --config tiny --device cpu \
  --output logs/day03-tiny-shapes.json
```

小模型通过后，再按教程使用 Qwen3 本地权重和 CUDA；脚本不自动下载模型。

学习：

- Q、K、V 投影
- Causal Mask
- MHA、MQA、GQA 的差别
- Query Head 与 KV Head 的关系

阅读：

- `GroupedQueryAttention.forward`

用 `B=1、T=8` 填写并验证：

```text
input_ids     [B, T]
queries       [B, 16, T, 128]
keys/values   [B, 8, T, 128]
attention     [B, 16, T, T]
logits        [B, T, vocab_size]
```

当天产出：一页 Tensor Shape 记录，包含每次 reshape、transpose 和
`repeat_interleave` 的原因。

### Day 4：RoPE、KV Cache、Prefill 和 Decode

可执行 Notebook：[Day 4 KV Cache 实验](learning/day04_kv_cache.ipynb)；
配套说明：[Day 4 KV Cache 教程](docs/DAY04_KV_CACHE.md)。

学习：

- RoPE 的旋转方式
- Prefill 和 Decode 的计算差异
- KV Cache 保存的内容和形状
- 为什么 Decode 阶段只需要最新 token

阅读：

- `Qwen3Model.forward`
- `KVCache`
- `apply_rope`

当天产出：画出一次带 KV Cache 的推理流程，并回答：

1. Prefill 阶段计算了什么？
2. Decode 阶段复用了什么？
3. 每层 Cache 中的 K/V 形状是什么？

实践：使用相同权重、固定 8 token 输入，对比一次完整前向和 Cache 的
`7 token prefill + 1 token decode`，验证最后位置 logits 的数值一致性。
Tiny FP32 要求严格对齐；真实 Qwen3 BF16 记录最大误差和 top-1 一致性，
不要把 BF16 逐元素误差直接当成逻辑失败。每次独立运行都重置 Cache 和位置状态。
这一天调用已有模型验证，尚不要求手写 Cache。

### Day 5：MoE 和 MLA

配套说明：[Day 5 架构、MoE 和 MLA 教程](docs/DAY05_ARCHITECTURES.md)；
配套 Notebook：[Day 5 架构图和路由实验](learning/day05_architecture_map.ipynb)。
字段已经熟悉后，补做 [配置、投影参数与内存](learning/day05_config_memory.ipynb)，
分清权重、KV Cache 和单层 score 三类内存，避免只记头数和宽度。
拆头后继续 [Attention汇集到残差的手算](docs/ATTENTION_HAND_CALC.md) 和
[逐步Notebook](learning/day05_attention_handcalc.ipynb)，分清合并头与输出投影。
MLA 专项：[直观解释与手算](docs/MLA_INTUITION.md)、
[展开 KV / latent 计算对齐实验](learning/day05_mla_intuition.ipynb)。
这项实验只验证同一低秩结构的计算等价，不代表已复现完整 DeepSeek。

学习：

- Expert、Router、Top-k 路由
- Expert Capacity 和负载均衡
- MLA 的 latent KV 压缩、解压和位置编码关系

当天只完成教学级数据流，不尝试复刻完整生产实现。

当天产出：分别画出 MoE 和 MLA 的数据流图，并说明它们主要节省的是
计算、参数激活量还是 KV Cache。

### Day 6：GPU、CUDA、Triton 和 FlashAttention

先读 [Day 6 Online Softmax 图解与手算](docs/DAY06_ONLINE_SOFTMAX.md)，
打开 [两块数字实验](learning/day06_online_softmax.ipynb)，再运行
统一入口 `learning/day06_course.ipynb`。从数学参考开始，不要求今天完成生产GPU kernel。

学习：

- Thread、Warp、Block、Grid
- Global Memory、Shared Memory、寄存器
- 内存合并访问
- Tiling、Reduction
- FlashAttention 的分块和 Online Softmax

当天产出：解释为什么 FlashAttention 不需要保存完整的
`[B, H, T, T]` attention score 矩阵。

### Day 7：vLLM 机制总览

先看 [Day 7学习记录](docs/DAY07_STUDY_LOG.md)和
[请求调度、Continuous Batching与Paged KV讲义](docs/DAY07_SCHEDULER_MAP.md)，
运行 [A/B/C调度实验](learning/day07_vllm_map.ipynb)。
原 `day07_course.ipynb` 保留为统一参考报告入口，不需要部署真实vLLM才开始。
四个角色、真实存储和生命周期用 [工作台/抽屉图解](docs/DAY07_KV_LIFECYCLE.md)
及 [KV存取Notebook](learning/day07_kv_lifecycle.ipynb) 补齐；
debug先看所有权与有效长度，再看调用批次，最后做单变量性能实验。

学习：

- KV Block
- PagedAttention
- Continuous Batching
- Scheduler
- Prefill/Decode 调度

当天产出：画出一个请求从进入 vLLM 到生成 token 的流程图，并写出
静态 Batch 和 Continuous Batching 的区别。

### 第 1 周验收

- [ ] 能说明 token、embedding、attention、logits 的数据流。
- [ ] 能写出 Q、K、V 的实际 Tensor Shape。
- [ ] 能解释 MHA、MQA、GQA。
- [ ] 能解释 Prefill、Decode 和 KV Cache。
- [ ] 能解释 MoE、MLA、PagedAttention 的基本目标。
- [ ] 成功运行一次短文本生成。

## 第 2 周：从零实现模型核心

目标：把第一周的概念变成可以测试的代码。

建议目录：

```text
mini_transformer/
  model.py
  attention.py
  rope.py
  kv_cache.py
  test_model.py
  benchmark.py
```

### Day 8：RMSNorm 和 RoPE

- 实现 `RMSNorm`。
- 实现 RoPE 参数生成和旋转。
- 用固定随机输入与仓库实现对齐。

验收：先在固定输入、FP32 和小 Shape 下与参考实现对齐，可从
`atol=1e-4, rtol=1e-4` 开始，同时记录最大绝对误差；BF16/FP16 另设容差，
不要沿用 FP32 阈值判断所有低精度实现。

### Day 9：单头 Attention

- 实现 Q/K/V 投影。
- 实现缩放点积 Attention。
- 实现 Causal Mask。
- 验证未来 token 不会影响当前位置输出。

### Day 10：MHA

- 将单头 Attention 扩展到多头。
- 确认 `[B, T, H, D]` 与 `[B, H, T, D]` 的转换。
- 测试不同 batch 和 sequence length。

### Day 11：GQA

- 保留多个 Query Head。
- 减少 KV Head。
- 实现 KV 的 head 扩展。
- 对比 MHA、MQA、GQA 的 KV Cache 大小。

### Day 12：KV Cache

- 实现 Prefill。
- 实现逐 token Decode。
- 比较有 Cache 和无 Cache 的输出一致性。
- 测量首 token 延迟和平均 Decode 延迟。

### Day 13：Transformer Block

- 加入 RMSNorm、Residual、SwiGLU。
- 堆叠多个 Transformer Block。
- 输出 logits。

### Day 14：对齐和 Benchmark

- 与仓库 Qwen3 模块进行数值对齐。
- 添加单元测试。
- 添加延迟、吞吐和显存测试。

### 第 2 周验收

- [ ] 能独立运行 `mini_transformer`。
- [ ] GQA 与参考实现数值对齐。
- [ ] KV Cache 输出与无 Cache 输出一致。
- [ ] 能解释 Cache 对显存和延迟的影响。
- [ ] 项目有测试和 Benchmark。

## 第 3 周：CUDA 和 Triton 算子

目标：具备基础 Kernel 编写、正确性验证和性能分析能力。

实现顺序：

```text
Vector Add
→ RMSNorm
→ Row-wise Softmax
→ LayerNorm
→ Fused SwiGLU
→ Causal Attention
```

### Day 15：CUDA 编程模型

- 编写 Vector Add。
- 观察 Thread、Block、Grid 的索引关系。
- 理解 Global Memory 和线程访问模式。

### Day 16：CUDA Reduction

- 实现 Sum 或 Max Reduction。
- 学习 Shared Memory 和 Warp Reduction。
- 处理非整齐长度和边界条件。

### Day 17：Triton 入门

补充实验：[Triton IR讲义](docs/TRITON_IR.md)、
[实际编译产物Notebook](learning/day17_triton_ir.ipynb)。
IR全称Intermediate Representation；观察编译链不等于kernel性能验收。

- 重写 Vector Add。
- 实现 RMSNorm。
- 学习 `program_id`、块指针和 Mask。

### Day 18：Softmax 和 LayerNorm

- 实现 Row-wise Softmax。
- 实现 LayerNorm。
- 对比 PyTorch 参考结果。

### Day 19：MatMul 和分块

- 理解 Tile 大小对性能的影响。
- 测试不同输入 Shape。
- 记录访存和计算瓶颈。

### Day 20：Attention Kernel

- 实现教学版 Causal Attention。
- 不追求一次达到 FlashAttention 性能。
- 先保证数值正确和边界完整。

### Day 21：FlashAttention

- 实现分块加载 Q/K/V。
- 实现 Online Softmax。
- 避免保存完整 attention score。
- 与 `torch.nn.functional.scaled_dot_product_attention` 对比。

时间不足时先交付正确的 Online Softmax 分块参考实现，再列出 GPU Kernel
待完成项；不能把阅读教程或调用 SDPA 标记为独立实现 FlashAttention。

### 第 3 周验收

- [ ] 至少完成 3 个 Triton 算子。
- [ ] 每个算子有随机输入和边界 Shape 测试。
- [ ] 有最大误差、延迟和吞吐记录。
- [ ] 至少一个算子优于朴素 PyTorch 实现。
- [ ] 能区分计算瓶颈、访存瓶颈和 Kernel Launch 开销。

## 第 4 周：vLLM 和完整实战项目

目标：将模型、算子和推理系统知识组合成一个完整项目。

### Day 22：运行 vLLM

- 跑通单请求推理。
- 观察 Prefill 和 Decode。
- 记录输入长度、输出长度、延迟和吞吐。

在独立环境中核对 ARM64/GB10 兼容性，不覆盖当前 Spark 学习镜像的依赖。
若安装受阻，记录具体阻塞和已完成的源码分析，不能将模拟器结果当成 vLLM 实测。

### Day 23：PagedAttention

- 阅读 KV Block 的分配和回收逻辑。
- 推导教学级 Paged KV Cache 的数据结构。
- 画出逻辑位置到物理 Block 的映射，为 Day 25 的实现准备测试样例。

### Day 24：Continuous Batching

- 理解请求进入、等待、Prefill、Decode 和完成。
- 画出 Scheduler 状态转移。
- 说明为什么不同请求可以交错 Decode。

### Day 25：Paged KV Cache 模拟器

至少支持：

- Block 分配。
- Block 释放。
- 逻辑序列到物理 Block 的映射。
- 不同长度请求同时存在。

### Day 26：简化 Scheduler

实现：

- 等待队列。
- Running 队列。
- 完成请求回收。
- 每轮 Decode 的 token budget。

### Day 27：性能对比

比较：

- Sequential。
- Static Batch。
- Continuous Batching。

记录：

- 首 token 延迟。
- 平均 Decode 延迟。
- tokens/s。
- 峰值显存。
- 活跃请求数。

若只有调度模拟器，只报告模拟的排队、占用和调度指标；只有接入真实模型执行后，
才能报告实测 GPU tokens/s、首 token 延迟和峰值显存。

### Day 28：选择一个优化方向

只选一个：

- INT8/INT4 量化。
- Prefix Cache。
- Speculative Decoding。
- CUDA Graph。
- Tensor Parallel。

完成一个最小实验，不要同时实现多个方向。

### Day 29：整理项目

项目建议结构：

```text
mini_vllm/
  model.py
  attention.py
  paged_kv_cache.py
  scheduler.py
  batch_engine.py
  benchmark.py
  tests/
  README.md
```

README 至少包含：

- 项目目标。
- 系统架构图。
- 运行命令。
- 测试命令。
- Benchmark 配置。
- 结果表。
- 已知限制。
- 下一步优化方向。

### Day 30：最终验收

完成一次项目答辩式复盘：

1. 解释一次请求的完整生命周期。
2. 解释 GQA 如何减少 KV Cache。
3. 解释 PagedAttention 解决了什么问题。
4. 解释 Continuous Batching 如何提升吞吐。
5. 展示一个算子的正确性和性能数据。
6. 说明最慢的模块和下一步优化方案。

## AI 辅助开发流程

从第一天开始使用固定流程：

1. 让 AI 解释当前模块的输入、输出和调用关系。
2. 让 AI 只实现一个小功能。
3. 要求同时生成单元测试。
4. 用另一个 Subagent 或独立审查步骤检查边界条件。
5. 用 Benchmark 验证性能，不能只接受文字结论。
6. 将已经验证的事实写入项目 README 或实验记录。

推荐的任务拆分：

```text
理解 → 最小实现 → 测试 → Benchmark → 优化 → 复盘
```

Memory 只保存稳定事实，例如目录职责、启动命令、已知限制和验证过的
性能结论；不要把未经测试的猜测写入长期记忆。

## 每日实验记录模板

```text
日期：
学习主题：
要回答的问题：
代码路径：
运行命令：
输入 Shape：
输出 Shape：
设备与 dtype：
正确性结果：
延迟/吞吐/显存：
遇到的问题：
原因判断：
今天结论：
下一步：
```

## 最终能力标准

达到以下标准，可以认为具备“可实战”能力：

- 能读懂 Llama、Qwen、DeepSeek 的 Attention 代码。
- 能手写 GQA、RoPE、KV Cache。
- 能解释 MoE 和 MLA 的基本数据流。
- 能开发并测试 Triton 算子。
- 能解释 FlashAttention 的性能来源。
- 能说明 vLLM 如何管理 KV Cache。
- 能实现简化 PagedAttention 或 Continuous Batching。
- 能用 Profiler 定位性能瓶颈。
- 有完整项目、测试、Benchmark 和技术报告。

严格生产级精通还需要继续积累多 GPU、量化、分布式推理、模型并行、
硬件差异和线上故障处理经验。
