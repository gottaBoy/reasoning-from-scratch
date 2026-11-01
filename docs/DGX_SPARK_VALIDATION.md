# DGX Spark 验证记录

验证日期：2026-10-30（UTC）。
代码基线：`2c6e047`，加本次 Spark 教程、环境检查和 smoke 改动。

## 结论

本机已跑通 CUDA 环境、Triton Vector Add、Qwen3 小规模推理/评测、
蒸馏、GRPO 和普通测试。**没有跑完所有 notebook、所有训练变体或正式长训练。**

这是一份运行链路验证报告，不是模型质量、收敛性或生产吞吐报告。
后续 Day 6–30 教学代码与 Notebook 的新增验证见
[课程验证记录](COURSE_VALIDATION.md)，本页早期 177 项测试计数保留为历史基线。

## 环境

| 项目 | 实际值 |
| --- | --- |
| 主机架构 | ARM64 / aarch64 |
| GPU | NVIDIA GB10，compute capability 12.1 |
| 驱动 | 580.173.02 |
| Python | 3.12.14 |
| PyTorch | 2.10.0+cu130 |
| PyTorch CUDA | 13.0 |
| Triton | 3.6.0 |
| GPU 注入 | Docker CDI |
| 镜像 | `reasoning-from-scratch:spark` |
| 最终实测镜像 ID 前缀 | `781d6a68f63e` |
| 数据 volume | `reasoning_spark_workspace` |

CPU 对照使用此前已构建的 `reasoning-from-scratch:local`，
PyTorch 为 `2.10.0+cpu`。

## 已执行项目

| 项目 | 规模 | 结果 |
| --- | --- | --- |
| 容器 GPU 可见性 | `nvidia-smi` | 通过 |
| CUDA 运算检查 | FP32 MatMul、BF16 causal SDPA 前后向 | 通过 |
| Triton JIT | Vector Add，1003 / 65537 元素，含 masked tail | 通过 |
| 第 2 章生成 | base 模型，最多 16 个新 token，KV Cache | 通过 |
| 第 3 章 MATH-500 | 1 题，最多 16 个新 token | 通过 |
| 第 4 章 CoT | 1 题 | 通过 |
| 第 4 章 Self-consistency | 1 题，2 次采样 | 通过 |
| 第 5 章 Self-refinement | 1 题，1 轮，critique 最多 16 token | 通过 |
| 第 5 章 Best-of-N | 1 题，2 次采样，heuristic scorer | 通过 |
| 第 6 章 GRPO | 1 step，2 rollouts | 通过并保存 checkpoint |
| 第 7 章 Format reward | 1 step，2 rollouts，reasoning 模型 | 通过并保存 checkpoint |
| 第 8 章蒸馏 | 5 条样例，4 train / 1 val，1 epoch | 通过，4 步更新并保存 checkpoint |
| 普通测试，Spark 镜像 | `SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0` | 177 passed，27 skipped |
| 普通测试，旧 CPU 镜像 | 同上，不含本次新增 4 项工具测试 | 173 passed，27 skipped |
| 新增工具测试 | CPU 自检、CUDA 失败不回退、smoke 日志/退出码 | 4 passed |
| 新增 Python 文件 lint | Ruff | 通过 |
| JupyterLab | 仅本机 8888，带 token 的 `/api/status` | HTTP 200 |
| Day 4 KV Cache | Tiny FP32、Qwen3 BF16/FP32，7-token prefill + 1-token decode | 通过；见 `docs/DAY04_KV_CACHE.md` |

GPU 可见的测试运行仍可能有大量 CPU 单元测试，不能把 177 项全部表述为
177 个 GPU Kernel 测试。跳过的 27 项不算通过，正式下载和高成本测试未全量执行。

两次短 GPU 生成脚本分别报告约 24 / 25 tokens/s，耗时 0.65 / 0.63 秒；蒸馏报告峰值 CUDA
allocated memory 为 13.51 GiB。这些是单次短任务观察值，不能与不同 prompt、
不同环境的 CPU 结果直接计算加速比。

短 GRPO 验证中的奖励和 advantage 为 0，只证明代码路径可执行、checkpoint
可保存，不代表模型学到了有效能力。1 题 MATH-500 的准确率也没有统计代表性。

## 重现命令与日志

完整 smoke 包含 11 个阶段，失败即停止：

```bash
docker compose -f docker-compose.spark.yml run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e NO_PROXY \
  -e MAX_NEW_TOKENS=16 -e DATASET_SIZE=1 \
  -e RUN_TRAINING=1 -e RUN_ADVANCED_TRAINING=1 \
  reasoning sh docker/smoke_examples.sh
```

首次成功运行的日志保存在 volume 内：

```text
/workspace/logs/spark-validation-20261030/summary.tsv
/workspace/logs/spark-validation-20261030/environment.json
/workspace/logs/spark-validation-20261030/tests.log
```

补上 C 编译工具后的最终镜像又完整重跑一次，11 个阶段退出码均为 0，
普通测试为 177 passed / 27 skipped（23.90 秒）：

```text
/workspace/logs/spark-validation-final-20261030/summary.tsv
/workspace/logs/spark-validation-final-20261030/environment.json
/workspace/logs/spark-validation-final-20261030/tests.log
```

Triton 在补上编译工具后单独运行通过，未包含在这 11 个 smoke 阶段内。

三个训练输出：

```text
ch06/02_rlvr_grpo_scripts_intro/checkpoints/rlvr_grpo_original_no_kl/qwen3-0.6B-rlvr-grpo.pth
ch07/03_rlvr_grpo_scripts_advanced/checkpoints/7_6_plus_format_reward/qwen3-0.6B-rlvr-grpo.pth
ch08/04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```

日志和模型没有加入 Git。部分评测脚本会重复使用相同的根目录 JSONL 文件名，
应以每个阶段独立的 `.log` 和本次 `summary.tsv` 确认运行结果，不要误把最后
一个 JSONL 当作全部评测结果。

## 已发现并处理的问题

1. pip 与 uv 的源/代理是两条链路，构建时需要分别考虑。
2. ARM64 CPU wheel 不会因为注入 GPU 而变成 CUDA wheel，Spark 单独使用
   CUDA 镜像标签。
3. 本机 Docker bridge 地址池耗尽，Spark 使用 Host 网络，不清理其他项目。
4. 本机 CDI 可用，使用 `nvidia.com/gpu=all` 注入。
5. Python slim 缺少 C 编译器，导致 Triton 首次 JIT 失败；
   Spark 镜像补上 `build-essential` 后 Vector Add 通过。
6. 初次 CPU 普通测试有 3 项 GitHub 数据下载超时；向容器传递已配置的代理后，
   同一套旧 CPU 测试为 173 passed / 27 skipped，无失败。

保留的警告：

- PyTorch wheel 列出 `sm_120` / `compute_120`，GB10 为 `sm_121`。
  已验证的运算可以运行，但第三方扩展仍需逐项实测。
- `generate_simple.py` 的统计函数接收 CPU token 列表时可能提示
  `CUDA is available but tensors are on cpu`。本次生成模型实际在 CUDA 上，
  不能把这个统计警告解释成全部推理回退到 CPU。

## 尚未验证

- 所有 notebook 从头执行、练习解答逐格执行。
- 完整 MATH-500、多随机种子的质量对比。
- 长时间 GRPO/蒸馏收敛、断点恢复、所有第 7 章变体。
- FSDP/多 GPU/多节点训练。
- 所有模型尺寸和 checkpoint，INT8/INT4 量化路径。
- 全量真实网络下载测试和高成本测试。
- vLLM、PagedAttention、Continuous Batching 的生产级实现。
- 手写 FlashAttention 或其他学习计划中尚未实现的算子。
- 外部 OpenRouter 等付费 API、Ollama 服务。
- Appendix F/G 的可选依赖和交互式 Chainlit 工作流。

这些项目应继续按 [30 天学习计划](../LLM_30_DAY_STUDY_PLAN.md) 分步推进，
不能由本次 smoke 结果推断已完成。
