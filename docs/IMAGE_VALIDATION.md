# 课程镜像与新工作区验收

验证日期：2026-11-01（America/Los_Angeles）；UTC 为 2026-11-02。
构建源码基线：`87428f3`。本报告为构建后补充文档，不改变已验收的课程实现。

## 结论

课程已实际打入 Spark 镜像，不再依赖给旧学习卷手工同步源码。
从空 volume 初始化后，完整 smoke、普通测试及课程 Notebook 均已重新验证。
Day 22 仍为 `external_service_not_configured`，不计通过。

| 项目 | 实测值 |
| --- | --- |
| 当前镜像 | `reasoning-from-scratch:spark` |
| 本次版本标签 | `reasoning-from-scratch:spark-course-87428f3` |
| 镜像 ID | `sha256:4c3249a6ab5d437206d0753c0272ccf7e22d5e04414ebacebe6afd52100503b7` |
| 旧镜像回退标签 | `reasoning-from-scratch:spark-pre-course-781d6a68f63e` |
| 独立验证卷 | `reasoning_spark_audit_20261102T030329Z` |
| Python / PyTorch / Triton | `3.12.14` / `2.10.0+cu130` / `3.6.0` |
| GPU | NVIDIA GB10，CUDA 13.0 |

版本标签是本机留存标签，不是已发布的远端镜像；需要严格固定版本时使用镜像 ID。

## 验证方法与结果

1. 使用仓库 `Dockerfile` 和 `docker-compose.spark.yml` 重新构建，
   保留 `uv sync --frozen`，未借用其他项目的运行环境。
   构建复用本机已有包缓存及已配置的代理/包源，不是首次无缓存下载验收。
2. 用独立 Compose 项目创建空 named volume，无宿主机源码 bind mount。
   Entrypoint 自行从 `/opt/repo` 初始化 `/workspace`。
3. 在复制模型缓存前，对 150 个课程、文档、测试和配置文件核对 SHA-256：
   镜像 `/opt/repo`、新卷 `/workspace` 均与构建基线一致，零缺失、零差异。
4. 仅从旧学习卷只读复制两份 Qwen3 0.6B 权重、两份 tokenizer、
   `math500_test.json` 和 `math_train.json`。没有复制旧源码、日志或 checkpoint。
5. 在新容器运行以下检查：

| 检查 | 结果 |
| --- | --- |
| CUDA MatMul、BF16 Attention 前后向 | 通过 |
| Triton JIT Vector Add、masked tail | 通过 |
| CUDA C++ Vector Add / Shared Memory Reduction | 6 组输入通过；宿主机 nvcc 重新编译 |
| 新工作区 Jupyter | 临时本机 8889 端口，`/api/status` HTTP 200 |
| 完整 smoke | 11 阶段退出码均为 0，含 ch06 / ch07 短训练 |
| smoke 内普通测试 | **282 passed / 27 skipped / 1 warning**，36.60 秒 |
| Day 6–30 共 25 个 Notebook | 均执行结束；24 passed，Day 22 外部服务未配置 |
| 配置/内存、MLA、Online Softmax、Triton IR 专项 Notebook | 4 份全部执行通过 |
| 课程 Python/Ruff、Shell/Compose 检查 | 通过；Notebook 路径初始化的 E402 单独排除 |

蒸馏使用 5 条样例、4 train / 1 val、1 epoch，完成 4 次更新并保存 checkpoint，
峰值 CUDA allocated memory 为 13.51 GiB。
ch06 / ch07 各执行 1 step、2 rollouts、最多 16 个新 token 并保存 checkpoint。
这两次短训练 reward/advantage 为零，不代表学到了有效能力。

## 保留的失败与限制

首次 smoke 在加载 Qwen3 权重时发生 CUDA OOM，当时 CUDA 报告可用内存约 7 GiB。
没有停止用户的 Notebook 内核，也没有执行全机 `drop_caches`。
仅对本次复制涉及的模型文件请求 `POSIX_FADV_DONTNEED`，释放可回收的文件页缓存，
随后以同样参数完整重试成功。首次失败日志与重试日志分别保留。
这说明共享内存压力会影响运行，不能承诺任意并行负载下都能成功。

PyTorch 的 GB10 capability 提示仍存在。已测 GPU 路径通过不等于所有第三方扩展兼容。
Notebook 内核还报告本机 TCP 传输未加密的提示；本次未将 Jupyter 暴露到公网。

未新增真实 vLLM 服务、完整 MATH-500、长训练收敛、多卡/多节点或生产吞吐验收。
本次没有验证所有原书 Notebook；课程参考实现通过不改变个人 Day 5 的学习进度。

## 日志与复现

原始证据保留在独立验证卷内：

```text
/workspace/logs/image-validation/source-manifest.json
/workspace/logs/image-validation/environment.json
/workspace/logs/image-validation/cache-seed.json
/workspace/logs/image-validation/smoke/summary.tsv
/workspace/logs/image-validation/smoke-retry1/summary.tsv
/workspace/logs/image-validation/smoke-retry1/tests.log
/workspace/logs/image-validation/notebooks/summary.json
/workspace/logs/image-validation/supplemental-notebooks/summary.json
/workspace/logs/image-validation/cuda-vector-reduce
```

新卷的创建与基础检查命令见 [Spark 教程](../DGX_SPARK.md)。
重新完整验证时使用新日志目录，不覆盖以上证据：

```bash
SPARK_WORKSPACE_VOLUME=reasoning_spark_audit_20261102T030329Z \
  docker compose -p reasoning-course-check -f docker-compose.spark.yml \
  run --rm --no-deps \
  -e MAX_NEW_TOKENS=16 -e DATASET_SIZE=1 \
  -e RUN_TRAINING=1 -e RUN_ADVANCED_TRAINING=1 \
  reasoning sh docker/smoke_examples.sh
```

旧 `reasoning_spark_workspace` 和原本机 8888 Jupyter 保持不变。
验证用的临时服务与容器在完成后移除，独立验证卷、版本镜像和日志留存。
重建镜像不会自动刷新已有学习卷；不要用 `down -v` 更新课程。
