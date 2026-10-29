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

## 使用本仓库

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

GPU 环境将 `docker run` 增加 `--gpus all`，并把 `--device cpu` 改为
`--device cuda`。模型和学习产物仍保存在 `reasoning_workspace` volume 中。

学习期间先使用短输入和较小的 `max_new_tokens`，不要一开始运行完整评测或
长时间训练。

## 固定学习闭环

每天按下面的顺序执行：

1. 阅读 30～60 分钟，明确当天要回答的问题。
2. 手写最小实现，不直接复制完整框架。
3. 用随机输入或仓库测试验证正确性。
4. 用 Benchmark 记录延迟、吞吐量、显存和误差。
5. 写下失败原因、结论和下一步。

每周只选择一个主问题，避免同时修改多个变量。

## 第 1 周：建立完整知识地图

第一周的目标是“知道每项技术是什么、解决什么问题、在哪里实现”，而不是
立刻实现所有高级功能。

### Day 1：环境与 Qwen3 推理

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

学习：

- Embedding、Attention、FFN、Residual、Norm
- Llama、Qwen、DeepSeek 的主要组件
- Pre-Norm、RMSNorm、SwiGLU、RoPE

阅读：

- `QWEN_CONFIG_06_B`
- `Qwen3Model.forward`
- `TransformerBlock.forward`
- `FeedForward.forward`

当天产出：完成一张模型组件对比表，至少包含归一化、位置编码、FFN、
Attention 和权重命名。

### Day 3：Attention、MHA、MQA、GQA

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

### Day 5：MoE 和 MLA

学习：

- Expert、Router、Top-k 路由
- Expert Capacity 和负载均衡
- MLA 的 latent KV 压缩、解压和位置编码关系

当天只完成教学级数据流，不尝试复刻完整生产实现。

当天产出：分别画出 MoE 和 MLA 的数据流图，并说明它们主要节省的是
计算、参数激活量还是 KV Cache。

### Day 6：GPU、CUDA、Triton 和 FlashAttention

学习：

- Thread、Warp、Block、Grid
- Global Memory、Shared Memory、寄存器
- 内存合并访问
- Tiling、Reduction
- FlashAttention 的分块和 Online Softmax

当天产出：解释为什么 FlashAttention 不需要保存完整的
`[B, H, T, T]` attention score 矩阵。

### Day 7：vLLM 机制总览

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

验收：最大误差小于 `1e-4`。

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

### Day 23：PagedAttention

- 阅读 KV Block 的分配和回收逻辑。
- 实现一个教学级 Paged KV Cache。
- 支持逻辑位置到物理 Block 的映射。

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
