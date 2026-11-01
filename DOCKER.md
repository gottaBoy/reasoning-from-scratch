# Docker 学习环境

本机为 DGX Spark（ARM64 / NVIDIA GB10）时，优先使用
[DGX Spark 实战教程](DGX_SPARK.md) 的独立 Compose 配置；其中包括 CUDA
实际运算自检、代理、缓存复用和逐阶段验证记录。通用 CPU/GPU 命令见本文。

这个仓库的 Docker 配置默认不把源码目录 bind mount 到容器，而是把完整源码放进镜像，再复制到 Docker named volume `reasoning_workspace`。因此模型、数据集、日志、checkpoint 和 notebook 修改都不会写入宿主机仓库。

注意：named volume 第一次启动时才会从镜像初始化，重建镜像不会覆盖其中的源码
和笔记。不要用 `down -v` 更新代码，它会连同模型、checkpoint 和学习进度一起
删除；优先创建新的工作 volume，或备份后选择性同步源码。

## 1. 启动 JupyterLab

默认模式（不向容器注入 GPU）：

```bash
docker compose up --build
```

浏览器打开 <http://127.0.0.1:8888/lab?token=reasoning>。

有 NVIDIA Container Toolkit 时使用 GPU：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

验证 GPU：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reasoning \
  python docker/check_environment.py --device cuda
```

## 2. 直接运行脚本

不启动 Jupyter 也可以直接运行：

```bash
docker compose run --rm reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cpu --max_new_tokens 64
```

GPU 运行：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cuda --max_new_tokens 256
```

第一次运行会下载 Qwen3 0.6B 权重和 tokenizer，下载结果保存在 named volume 中，后续运行复用。

首次构建还会安装 PyTorch 及其平台依赖；在 ARM64 + CUDA 环境下镜像可能较大，建议预留磁盘空间并避免频繁 `down -v`。

默认镜像包含主线章节、测试和 Hugging Face 相关依赖；Appendix F/G 的可选依赖（`datasets`、`chainlit`）不默认安装。需要时可重新构建：

```bash
docker compose build --build-arg INSTALL_EXTRAS=1
```

## 3. 测试

快速测试（不下载大型模型）：

```bash
docker compose run --rm reasoning \
  sh -lc 'SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 pytest -q tests'
```

## 4. 一键跑通核心例子

项目包含很多 notebook、长时间训练脚本和需要外部 API key 的数据生成脚本，不能把“全部代码”都当作一次短测试。下面的 smoke run 会实际覆盖：模型加载与生成、MATH-500 评测、CoT、自洽采样、自我修订、best-of-N、蒸馏和快速测试套件；所有产物都写在 named volume。

CPU：

```bash
docker compose run --rm reasoning \
  sh docker/smoke_examples.sh
```

GPU：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reasoning \
  sh docker/smoke_examples.sh
```

可选地加入一次极小 GRPO 训练 sanity run：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm \
  -e RUN_TRAINING=1 reasoning sh docker/smoke_examples.sh
```

参数可覆盖：

```bash
docker compose run --rm \
  -e DEVICE=cpu -e DATASET_SIZE=1 -e MAX_NEW_TOKENS=16 \
  reasoning sh docker/smoke_examples.sh
```

Smoke run 通过后，再按下方路线逐章增加数据量；不要直接把 `DATASET_SIZE` 改成 500 或启动长训练。

### 核心例子覆盖矩阵

| 范围 | Smoke run | 后续完整运行 |
| --- | --- | --- |
| ch02 生成/聊天 | `generate_simple.py` | base/reasoning、多轮对话、`--compile` |
| ch03 评测 | `evaluate_math500.py`，1 题 | 3/10/500 题、batched、JSONL 重评 |
| ch04 推理扩展 | CoT、自洽采样各 1 题 | `num_samples=3/5/10`、batched |
| ch05 自我改进 | self-refinement、best-of-N 各 1 题 | scorer、迭代次数、完整 MATH-500 |
| ch06 GRPO | 默认跳过；`RUN_TRAINING=1` 时 1 step | 50/100 steps、batched、FSDP |
| ch07 GRPO 改进 | `RUN_ADVANCED_TRAINING=1` 时 format reward 1 step | 其他变体须分别执行，不能由共用组件推定通过 |
| ch08 蒸馏 | 自带 5 条样例、1 epoch | 50/500 条、think tokens、HF 导出 |
| Appendix C--G | 仅普通测试实际覆盖到的接口 | 未安装的 datasets/Chainlit 和外部服务不算已验证 |

## 5. 哪些例子不能无条件自动跑

下面这些例子依赖外部服务、密钥或明显更高的资源，Smoke run 不会擅自调用：

- `ch08/02_generate_distillation_data/generate_with_openrouter.py`：需要 `OPENROUTER_API_KEY`，并会产生 API 费用。
- `ch08/02_generate_distillation_data/generate_with_ollama.py`：需要宿主机或另一个容器中的 Ollama 服务和模型权重。
- Minimax 等 provider 示例：需要对应 API key。
- `ch06`/`ch07` 长时间 GRPO、batch 和 FSDP：需要 GPU、显存/内存和数小时运行时间。
- 完整 MATH-500、自洽采样和 self-refinement 大配置：运行时间按样本数、采样次数和生成长度成倍增加。
- Appendix G Chainlit：需要构建可选 extras，并单独暴露 Chainlit 服务端口。

这些脚本仍然都在镜像中；先用小参数跑通，再显式提供服务地址、密钥或 GPU。不要把 API key 写进镜像、Dockerfile、notebook 或 Git，使用 `docker compose run -e OPENROUTER_API_KEY=...` 临时注入。

## 6. 清理

只停止容器，保留模型和学习进度：

```bash
docker compose down
```

连同所有 Docker 内部源码、模型、数据和产物一起删除：

```bash
docker compose down -v
```

`down -v` 不会删除宿主机仓库文件。

## 一周实战路线

每天建议投入 3--5 小时：先读对应 notebook，再运行一个最小命令，最后改一个参数或函数并记录结果。章节 2--4 可以 CPU 完成；章节 5--8 最好使用 GPU，完整训练不适合在一周内全部跑完。

### Day 1：环境、张量和生成闭环

- 阅读根 README、`ch01/README.md`、`ch02/01_main-chapter-code/ch02_main.ipynb`。
- 运行 `generate_simple.py` 的 base 模型，理解 tokenizer、embedding、attention、KV cache、sampling。
- 练习：修改 prompt、`max_new_tokens`，比较 `--compile` 与不编译。

```bash
docker compose run --rm reasoning \
  python ch02/05_use_model/generate_simple.py \
  --device cpu --max_new_tokens 64 \
  --prompt "Explain attention in one sentence."
```

### Day 2：推理模型和对话

- 阅读 `ch02` notebook 后半部分，运行 reasoning 模型和多轮对话。
- 重点掌握 `<think>` 格式、chat template、上下文长度和停止 token。

```bash
docker compose run --rm reasoning \
  python ch02/05_use_model/chat_multiturn.py \
  --device cpu --reasoning --max_new_tokens 128
```

### Day 3：评测与 verifier

- 阅读 `ch03/01_main-chapter-code/ch03_main.ipynb`，理解 boxed answer 提取、SymPy 验证和 accuracy。
- 先跑 3--10 题，再决定是否跑完整 MATH-500。

```bash
docker compose run --rm reasoning \
  python ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --device cpu --which_model base --dataset_size 3 --max_new_tokens 128
```

### Day 4：推理时扩展

- 阅读 `ch04` 与 `ch05` notebook，比较 CoT、自洽采样、best-of-N、自我修订。
- 先用 `dataset_size=3`、`num_samples=3`、`iterations=1` 做回归实验。
- 记录准确率、耗时、生成 token 数，建立自己的实验表。

```bash
docker compose run --rm reasoning \
  python ch04/02_math500-inference-scaling-scripts/self_consistency_math500.py \
  --which_model base --dataset_size 3 --num_samples 3 \
  --temperature 0.9 --top_p 0.9 --prompt_suffix "\n\nExplain step by step."
```

```bash
docker compose run --rm reasoning \
  python ch05/02_math500-more-inference-scaling-scripts/self_refinement_math500.py \
  --which_model base --dataset_size 3 --iterations 1 --scoring none
```

### Day 5：RLVR/GRPO 原理

- 阅读 `ch06` notebook 和 `rlvr_grpo_original_no_kl.py`，按顺序追踪 rollout、reward、advantage、policy loss、checkpoint。
- 只做极小 sanity run；完整训练需要 GPU、较大内存和长时间。

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reasoning \
  python ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py \
  --steps 2 --num_rollouts 2 --max_new_tokens 64
```

### Day 6：GRPO 稳定化与蒸馏

- 阅读 `ch07` 中 tracking、clip ratio、KL、format reward 的差异。
- 阅读 `ch08`，用自带 5 条样例跑一次 distillation；理解 teacher data、supervised loss、checkpoint。

```bash
docker compose run --rm reasoning \
  python ch08/04_train_with_distillation/distill.py \
  --data_path ch08/02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 --validation_size 1 --epochs 1 --log_every 1
```

### Day 7：端到端复盘与改造

- 从 prompt 到生成、验证、奖励、更新、评估画出一张数据流图。
- 选一个变量做小型 ablation：temperature、top-p、rollout 数、max token 或 reward 规则。
- 运行 `pytest` 快速套件，整理“实现、指标、瓶颈、下一步”四栏实验报告。

## 完整路线图：7 天入门 + 4 周进阶

### 先建立统一的学习闭环

每个实验都按下面的顺序做，不要只运行 notebook：

1. **读**：先读该章节 README 和 notebook 中的目标、输入、输出。
2. **跑**：先用最小数据量跑通，确认设备、下载、路径和输出格式。
3. **拆**：沿着调用链定位 tokenizer、模型 forward、generation、verifier 或 loss。
4. **改**：只改一个变量，保存命令、耗时、准确率、显存/内存和异常。
5. **讲**：用自己的话解释“数据如何流动、指标为什么变化、瓶颈在哪里”。

建议在 Docker volume 内建立 `notes/week-1.md` 和 `notes/experiments.csv`，这样实验记录也不会写入宿主机。

### 阶段 0：准备与验收（30--60 分钟）

**目标**：确认环境可复现，知道 CPU/GPU 的边界。

- 安装 Docker；GPU 用户安装 NVIDIA Container Toolkit。
- `docker compose config` 和 `docker compose -f docker-compose.yml -f docker-compose.gpu.yml config` 均通过。
- 验证 `torch.__version__`、`torch.cuda.is_available()`。
- 运行 `pytest -q tests` 快速套件。

**完成标准**：容器可启动、Jupyter 可访问、测试能收集并执行，宿主机仓库没有新增模型/日志/checkpoint。

### 阶段 1：生成模型基础（Day 1--2）

**掌握**：tokenizer、embedding、RoPE/attention、KV cache、temperature/top-p、停止条件、chat template。

**代码范围**：`ch02`、`reasoning_from_scratch/qwen3.py`、`reasoning_from_scratch/ch02.py`、Appendix C。

**验收**：

- 能解释 base 与 reasoning tokenizer 的差异。
- 能修改 prompt 和 `max_new_tokens`，并说明输出变化。
- 能定位显存不足、模型下载失败、上下文超限和 `torch.compile` 编译失败。

### 阶段 2：评测与推理时扩展（Day 3--4）

**掌握**：MATH-500 数据格式、答案抽取、SymPy verifier、accuracy、CoT、自洽采样、best-of-N、自我修订。

**代码范围**：`ch03`、`ch04`、`ch05`、Appendix F。

**验收**：

- 能独立写一个 `evaluate_json.py` 风格的评测脚本。
- 能区分“模型变好”和“采样次数变多”带来的收益。
- 能用同一小数据集比较 greedy、CoT、self-consistency，并记录成本。

### 阶段 3：RLVR/GRPO（Day 5 + 第 1--2 周）

**掌握**：rollout、verifiable reward、group advantage、policy-gradient loss、KL、checkpoint、训练稳定性。

**代码范围**：`ch06`、`ch07`、`reasoning_from_scratch/ch06.py`、`ch07`。

**学习顺序**：

1. 先读 `ch06_main.ipynb`，只理解单个 rollout 的数据流。
2. 用 `steps=2`、`num_rollouts=2`、`max_new_tokens=64` 做 sanity run。
3. 再比较 `7_3_plus_tracking.py`、`7_4_plus_clip_ratio.py`、`7_5_plus_kl.py`。
4. 最后再尝试 batch/FSDP；不要一开始就运行长训练。

**完成标准**：能解释 reward 为 0 时为什么可能没有有效梯度，能读懂 metrics CSV，并能从 checkpoint 恢复或评估。

### 阶段 4：蒸馏和 Hugging Face 互操作（第 2--3 周）

**掌握**：teacher trace、监督微调目标、think token、长度过滤、验证集 loss、scratch checkpoint 与 Transformers 的转换。

**代码范围**：`ch08`、`ch08/06_use_via_huggingface`、Appendix D/E。

**实验阶梯**：

1. 自带 5 条样例，跑 1 个 epoch。
2. 逐步增加到 50/500 条，观察 loss 和训练时间。
3. 分别测试 base tokenizer 与 `--use_think_tokens`。
4. 用 chapter 3 evaluator 测 checkpoint，而不是只看训练 loss。
5. 导出 Hugging Face 格式，再用 `model.generate()` 做独立推理。

**完成标准**：能说明蒸馏数据中的 `problem`、`message_thinking`、`message_content` 如何变成监督序列，并能完成一次导出、加载、评估闭环。

## 成为专家：能力阶梯与长期学习路径

先完成上面的阶段 0--5，再沿下面的能力阶梯深入；阶段 5 的工程化能力是进入长期研究和系统优化的基础。

“专家”不是把所有 notebook 跑一遍，而是能在新问题上做出正确取舍：选择数据和目标、设计可验证指标、实现训练/推理、定位失败、控制成本，并把结果复现出来。建议按下面 6 个层级推进。

### L1：能读懂并运行（1 周）

- 目标：完成本文件的 7 天路线。
- 必须会：Python、PyTorch tensor、命令行、Git、Docker、基本概率。
- 项目：改一个 prompt 或采样参数，提交一份带命令和结果的实验记录。
- 晋级标准：从容器启动到模型输出不依赖复制粘贴，遇到错误能给出原因和下一步检查。

### L2：能解释模型（第 2--3 周）

- 目标：从 token id 一路解释到 logits 和生成文本。
- 学习主题：矩阵乘法、梯度、softmax/cross-entropy、attention、mask、位置编码、KV cache、采样。
- 源码顺序：`qwen3.py` → `ch02.py` → Appendix C → `qwen3_batched.py`。
- 项目：实现一个最小 greedy/temperature/top-p 解码器，并与仓库输出对齐。
- 晋级标准：能画出单层 Transformer 的张量形状，解释每个维度和显存增长来源。

### L3：能做可靠评测（第 3--5 周）

- 目标：知道“准确率提升”是否真实，避免只看一条漂亮样例。
- 学习主题：数据切分、泄漏、答案抽取、符号验证、pass@k、方差、置信区间、错误分类。
- 源码顺序：`ch03` → `ch04` → `ch05` → Appendix F。
- 项目：建立一个小型评测 harness，输出 JSONL、总体指标、按题型统计和失败样例。
- 晋级标准：能说明 verifier 错误、解析错误、模型错误和采样噪声各占多少。

### L4：能训练和调试推理模型（第 2--3 个月）

- 目标：理解并能修改 RLVR/GRPO 与蒸馏训练。
- 学习主题：policy gradient、advantage、reward sparsity、KL、clipping、group normalization、梯度累积、checkpoint 恢复。
- 论文阅读顺序：
  1. *DeepSeekMath*：GRPO 和数学 RL 的基本设定。
  2. *DeepSeek-R1*：大规模推理训练的整体流程。
  3. *DAPO*、*Dr. GRPO*：零优势过滤、无 KL、长度偏差等稳定化方法。
  4. 仓库 `ch07` 的 tracking、clip、KL、format reward 和改进脚本。
- 项目：在固定小数据集上比较 vanilla GRPO、no-KL、clip ratio、format reward 四个配置。
- 晋级标准：能从 loss/entropy/reward/advantage 曲线判断训练崩溃、奖励饱和、长度投机和数据问题。

### L5：能做高效系统（第 3--6 个月）

- 目标：把“能跑”变成“跑得快、占用少、可恢复、可复现”。
- 学习主题：batching、padding、KV cache、mixed precision、activation memory、checkpoint sharding、FSDP、数据加载、吞吐与延迟。
- 技术主题：FlashAttention、PagedAttention、speculative decoding、量化、LoRA/QLoRA、张量并行和流水线并行。
- 项目：为一个评测脚本做 sequential/batched 两个版本，报告 tokens/s、峰值显存、准确率和失败率。
- 晋级标准：给定显存和时延预算，能选择 batch size、max sequence length、精度和并行策略，并解释取舍。

### L6：能独立研究和交付（持续）

- 目标：提出可信假设，设计最小实验，复现基线，解释负结果，并交付可维护代码。
- 每个研究项目必须包含：可证伪假设；固定数据版本、代码 commit、随机种子和环境；baseline、ablation、误差分析和成本记录；可重复的启动命令和 checkpoint；对负结果和限制条件的说明。
- 项目方向：verifier-aware reasoning、reward hacking 与格式奖励、长上下文推理、蒸馏数据质量、推理成本/准确率 Pareto 前沿、小模型 reasoning 的迁移和泛化。
- 晋级标准：别人只拿到仓库、配置和命令，就能复现实验并得到相近结论。


## 面向岗位的四项专项能力路线

下面四项能力不建议同时铺开。先用本仓库建立“模型能跑、指标可信、性能可测”的基础，再按“架构 → 算子 → 推理系统 → AI 辅助工程”的顺序推进。每个阶段都应留下代码、基准数据和一页说明，而不是只完成阅读。

### 1. 模型架构：从 Qwen3 迁移到 Llama、Qwen 和 DeepSeek

**本仓库的入口**

- `reasoning_from_scratch/qwen3.py`：完整的 Transformer、RoPE、RMSNorm、SwiGLU、GQA 和 KV cache。
- `reasoning_from_scratch/qwen3_optimized.py`：用 SDPA 和预分配 cache 做推理优化。
- `reasoning_from_scratch/qwen3_batched.py`、`chE/`：批量生成、padding、位置 id 和吞吐测试。
- `chC/01_main-chapter-code/chC_main.ipynb`：按模块拆解 Qwen3 源码，适合逐段调试。

**学习顺序**

1. 给 `Qwen3Model.forward`、`GroupedQueryAttention.forward` 和 `FeedForward.forward` 标出每个 tensor 的形状、dtype、显存占用和数据流。
2. 只改 `n_kv_groups`，分别跑 MHA、MQA、GQA，记录 KV cache 大小、prefill/decode 速度和输出一致性。
3. 对照 Llama、Qwen、DeepSeek 的模型配置，建立一张“组件差异表”：归一化、位置编码、Q/K/V 投影、FFN、MoE 路由、上下文扩展和权重命名。
4. 在独立实验文件中实现一个最小 MoE：`router -> top-k experts -> weighted sum`，先用 2 个 expert 和很小的 hidden size 验证前向与反向。
5. 再实现一个只用于教学的 MLA 张量流，重点确认 latent KV 压缩、解压和 RoPE 的形状；不要一开始尝试复刻完整生产实现。

**阶段产出**

- 一张模型结构和张量形状图。
- 一份 MHA/MQA/GQA/MoE/MLA 对比表。
- 一个能加载同一份权重、只切换 attention 配置的最小实验。
- 一篇错误分析：哪些差异来自架构，哪些差异只是权重、tokenizer 或采样配置。

**验收标准**：给出任意一个新模型配置时，能从配置文件推断参数量、KV cache 大小、主要瓶颈，并指出它与仓库 Qwen3 实现需要修改的模块。

### 2. 算子编写：从 PyTorch 参考实现到 CUDA/Triton kernel

**学习顺序**

1. 先使用 `torch.profiler`、CUDA events 和 `torch.cuda.max_memory_allocated()` 建立 PyTorch 参考实现；没有 GPU 时先完成 CPU 正确性测试。
2. 按难度实现 `vector add → RMSNorm → row-wise softmax → fused SwiGLU → causal attention` 五个小算子。每个算子都要有随机输入、边界 shape、dtype 和误差阈值测试。
3. 用 Triton 重写前两个算子，再用 CUDA C++ 重写至少一个算子，比较 kernel launch 数、内存读写、延迟和吞吐。
4. 学习 FlashAttention 的核心思想：分块加载 Q/K/V、在线 softmax、避免显式保存完整的 attention score 矩阵；先实现教学版，再和 `scaled_dot_product_attention` 对比。
5. 最后做融合和调参：block size、warp 数、shared memory/register 压力、occupancy、不同 sequence length 下的拐点。

**必须保留的实验记录**

- PyTorch、Triton、CUDA 三个版本的前向误差和反向梯度误差。
- `seq_len`、batch、head_dim、dtype 变化下的 p50/p95 延迟和峰值显存。
- kernel 在短序列、长序列、非整除 block size 下的结果。
- 失败版本及原因，例如越界、数值溢出、非 contiguous tensor 或同步错误。

**验收标准**：能解释一个 kernel 的访存模式和瓶颈；在不改变结果的前提下，证明自己的实现在哪些 shape 上更快、哪些 shape 上更慢。没有 GPU 时不要虚构 benchmark，可先交付 reference、测试和待测脚本。

### 3. vLLM 机制：把本仓库的 KV cache 扩展成服务系统

本仓库没有 vLLM 实现，因此学习重点是“建立机制映射”，而不是把 vLLM API 当黑盒使用。

**先在本仓库完成**

1. 用 `qwen3.py` 理解单请求 prefill/decode 和连续追加 KV 的过程。
2. 用 `qwen3_batched.py` 理解 padding、position id、batch 内不同长度请求和停止条件。
3. 在一个小 notebook 中把连续 KV tensor 改成固定大小 block pool，维护 `request -> block table -> token range` 的映射。
4. 加入两个请求交错到达的模拟器，比较静态 batching 与 continuous batching 的 TTFT、TPOT、总吞吐和尾延迟。
5. 写出 block 回收、请求结束、超长上下文和共享 prefix 的不变量测试。

**再阅读 vLLM 源码时的顺序**

- scheduler：请求何时进入、暂停、恢复和完成。
- block manager / KV cache manager：物理 block、逻辑 token、block table 的映射。
- paged attention kernel：如何按 block 读取 K/V，避免连续大块内存要求。
- worker/model runner：prefill 与 decode 的执行边界、采样和并行。

**验收标准**：能画出一次请求从入队到完成的时序图，并解释 PagedAttention 解决的是内存碎片和动态请求管理问题，Continuous Batching 解决的是请求到达不齐导致的 GPU 空转问题。报告必须同时给出吞吐和延迟，不能只报 tokens/s。

### 4. AI 辅助开发：把 Claude Code 等工具纳入工程闭环

AI 工具的学习目标不是“让模型替你写更多代码”，而是让分析、实现、测试和复盘形成可审计流程。

**推荐工作流**

1. **Explore**：先让 AI 只读代码，输出调用链、关键张量、风险点和待验证假设；禁止直接改文件。
2. **Plan**：把任务拆成源码分析、最小实现、测试、benchmark、review 五个独立交付物。
3. **Implement**：一次只改一个行为边界，要求列出修改文件、兼容性影响和未覆盖情况。
4. **Verify**：自动运行格式检查、单元测试、数值对齐测试和性能脚本；失败时让 AI 解释日志，不接受“看起来应该可以”。
5. **Review**：让另一个上下文或 subagent 专门检查回归、边界条件、资源泄漏和 benchmark 是否公平。
6. **Memory**：只记录稳定事实，例如目录职责、启动命令、已知限制、权衡结论和常见故障；不要把未经验证的猜测写入长期记忆。

**Subagent、Hook、Memory 的练习方式**

- **Subagent**：为“架构阅读、测试设计、性能分析、代码审查”分别定义输入、输出和禁止事项；主线程只合并可验证结论。
- **Hook**：先接入 `ruff`/格式检查、快速单测和 diff 检查，再逐步加入 GPU benchmark；hook 失败时必须阻止提交或明确标记。
- **Memory**：维护一份短的项目事实清单，包含命令、环境、数据版本、模型 checkpoint 和已知失败案例，定期删除过时内容。
- **安全边界**：不向第三方模型发送密钥、私有数据和未公开权重；AI 生成的 CUDA、训练和部署代码必须人工审查并通过测试。

**验收标准**：从一个 issue 到可合并变更，能够保留提示词/任务拆分、代码 diff、测试日志、benchmark 和 review 结论；换一个人也能复现你的判断过程。

### 建议的 12 周安排

| 周次 | 主线 | 最小交付 |
| --- | --- | --- |
| 1--2 | Qwen3 源码、张量形状、KV cache | 结构图、MHA/MQA/GQA 对比 |
| 3--4 | Llama/Qwen/DeepSeek 架构差异、MoE/MLA 原理 | 配置差异表、MoE/MLA 教学实现 |
| 5--7 | PyTorch profiler、Triton、CUDA、FlashAttention | 3 个算子、正确性与 benchmark |
| 8--9 | PagedAttention、block pool、continuous batching | toy scheduler 和不变量测试 |
| 10 | vLLM 源码映射与服务指标 | TTFT/TPOT/吞吐报告 |
| 11--12 | AI 辅助开发闭环、capstone 整合 | 可复现 PR、测试、性能报告 |

每周只设置一个主问题，先用小模型、小 batch 和短序列跑通，再扩大规模。最终 capstone 可以选择“带 paged KV cache 的 Qwen3 batch 推理器”，同时提交架构说明、一个 Triton 算子、调度模拟器、测试和性能报告。

### 专家级每周节奏

- 1 天：读一篇论文，写半页“问题、方法、假设、局限”。
- 2 天：复现一个最小基线，不追求大规模。
- 1 天：做单变量 ablation 和错误分析。
- 1 天：优化代码、测试和日志，让实验可恢复。
- 1 天：写实验报告或技术分享，把结果讲给不熟悉代码的人。

### 专家级最终作品

建议完成一个端到端 capstone：

1. 选择一个公开、可验证的数学或代码数据集。
2. 用 base Qwen3 建立 greedy/CoT/self-consistency 基线。
3. 实现 verifier 和错误分类。
4. 训练一个小规模 GRPO 或蒸馏模型。
5. 做至少 3 个 ablation，并报告准确率、长度、吞吐、显存和成本。
6. 提供 Docker 一键复现、测试、配置文件、checkpoint 下载说明和限制条件。

完成这个项目后，才算具备“从论文到可复现实验，再到可交付系统”的专家能力。

### 阶段 5：工程化与性能（第 3--4 周）

**掌握**：batching、吞吐/延迟权衡、混合精度、显存分析、日志、可恢复训练、实验可复现和结果审计。

**代码范围**：Appendix D/E/F/G、`tests/`、所有 bonus scripts。

**验收项目**：

- 为一个评测脚本增加 JSONL 输出和断点续跑。
- 比较 sequential 与 batched 生成的吞吐、峰值显存和结果一致性。
- 为一个 GRPO 变量做最小 ablation，至少 3 个随机种子或 3 个配置。
- 写一页实验报告：问题、假设、配置、结果、失败原因、下一步。


## 第一步：第 1 周启动清单

第一周不要急着学习 MoE、CUDA 或 vLLM。目标只有一个：**能够解释 Qwen3 从输入 token 到输出 logits 的完整路径，并独立跑通一次 KV Cache 推理**。

### Day 1：确认环境并跑通最小生成

在仓库根目录执行：

```bash
docker compose build
docker compose run --rm reasoning python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果暂时不使用 Docker，则先确认 Python、PyTorch 和 tokenizer 可以导入，再运行仓库已有的最小生成脚本。第一次只使用很短的输入和少量新 token，不要直接运行完整评测。

记录以下信息：

- Git commit：`git rev-parse HEAD`
- Python、PyTorch 版本和设备类型
- 模型 checkpoint 来源
- 输入文本、`max_new_tokens`、dtype 和耗时
- 是否成功使用 KV Cache

### Day 2：阅读模型入口

按下面顺序阅读，不要从头到尾通读所有代码：

1. `reasoning_from_scratch/qwen3.py` 的 `QWEN_CONFIG_06_B`
2. `Qwen3Model.forward`
3. `TransformerBlock.forward`
4. `GroupedQueryAttention.forward`
5. `FeedForward.forward`
6. `ch02/05_use_model/generate_simple.py`

为每个模块填写一行：输入 shape、输出 shape、dtype、主要显存对象和作用。

### Day 3：手算并验证 tensor shape

以 batch size `B=1`、输入长度 `T=8` 为例，填写以下形状：

```text
input_ids       [B, T]
tok_embeds      [B, T, emb_dim]
queries        [B, n_heads, T, head_dim]
keys/values    [B, n_kv_groups, T, head_dim]
attention      [B, n_heads, T, T]
logits         [B, T, vocab_size]
```

然后在 forward 的关键位置加入临时打印或断点，确认实际 shape 与推导一致。完成后删除临时调试输出，保留一份记录即可。

### Day 4：理解 KV Cache

完成一次对比实验：

- 不使用 cache 生成固定数量 token。
- 使用 cache 生成相同数量 token。
- 比较首 token 延迟、平均 decode 延迟和峰值显存。

重点回答三个问题：

1. prefill 阶段计算了什么？
2. decode 阶段为什么只需要输入最新 token？
3. cache 中保存的 key/value 张量形状是什么？

### Day 5：做一次最小改动

只选择一个变量进行实验，例如：

- 修改 `n_kv_groups`，比较 MHA、MQA 和 GQA；或
- 修改 `max_new_tokens`，观察 decode 延迟和显存；或
- 对比 `qwen3.py` 与 `qwen3_optimized.py` 的 SDPA 实现。

不要同时修改多个变量。输出一张表：配置、生成结果是否一致、tokens/s、峰值显存和观察结论。

### 第一周记录模板

```text
日期：
Git commit：
设备：
Python/PyTorch：
实验问题：
修改变量：
命令：
结果：
遇到的问题：
原因判断：
下一步：
```

### 第一周完成标准

满足以下条件才进入第 2 周：

- 能说明 token、embedding、attention、logits 的数据流。
- 能写出 Q、K、V 的实际 tensor shape。
- 能解释 prefill、decode 和 KV Cache 的区别。
- 能独立运行一次生成实验并记录环境和结果。
- 能完成一个单变量实验，并说明结果是否可信。

## 资源与时间边界

| 内容 | CPU | 单 GPU | 备注 |
| --- | --- | --- | --- |
| ch02 生成、聊天 | 可以 | 更快 | 首次下载模型 |
| ch03 小规模评测 | 可以 | 推荐 | 先跑 3--10 题 |
| ch04/ch05 小规模实验 | 可以但慢 | 推荐 | 自洽采样成本按样本数增长 |
| ch06/ch07 训练 | 仅 sanity run | 推荐 | 长训练需要较大内存 |
| ch08 小样本蒸馏 | 可以但慢 | 推荐 | 先用自带 sample |
| 完整 MATH-500、长训练 | 不建议 | 视显存而定 | 先估时再启动 |

完整 MATH-500 和长时间训练不要放在第一天验证；先用小规模实验确认代码和参数，再扩大规模。

## 现实边界

“一周精通”不现实；一周可以达到：能读懂主线、跑通核心例子、定位下载/设备/显存问题、修改一个实验并解释指标。完整掌握 GRPO、FSDP、蒸馏和大模型工程，还需要继续做实验和阅读论文。
