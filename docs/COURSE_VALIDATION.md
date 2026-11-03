# Day 6–30 材料与参考实现验证

验证日期：2026-10-31（America/Los_Angeles）；日志 UTC 日期为 2026-11-01。
代码基线 `2c6e047` 加尚未提交的课程修改。**个人进度仍为 Day 5。**

## 结果

| 范围 | 结果 | 说明 |
| --- | --- | --- |
| Day 6–30 共 25 个新 Notebook | 全部执行到结束 | 不等于所有外部依赖项通过 |
| 其中参考实验 | 24 个 `passed` | 包含概念模拟、模型数值对齐、真实 GPU 算子与引擎 |
| Day 22 | `external_service_not_configured` | 没有部署/调用真实 vLLM 服务，不计通过 |
| 首轮课程全仓库普通测试 | **261 passed / 27 skipped / 1 warning** | 非 expensive/真实全量下载套件 |
| 补充配置/内存实验后的普通测试 | **269 passed / 27 skipped / 1 warning** | 新增 8 项配置算术/参数核对测试 |
| 补充Triton IR后的普通测试 | **274 passed / 27 skipped / 1 warning** | 新增 5 项导出/CPU拒绝/真实GPU编译产物测试 |
| 补充MLA代数实验后的普通测试 | **282 passed / 27 skipped / 1 warning** | 新增 8 项展开/吸收对齐、状态计数与非法维度测试 |
| CUDA C++ 独立程序 | Vector Add / Shared Memory Reduction 通过 | 6 组输入大小，宿主机 nvcc 编译 |
| Python/Ruff | 通过 | Notebook 路径初始化的 E402 单独排除，其他检查保留 |
| Notebook schema/代码 | 生成器和执行检查通过 | 25 份新文件，不含未执行的个人练习答案 |

环境为现有 DGX Spark / GB10，PyTorch `2.10.0+cu130`，Triton `3.6.0`，
CUDA Toolkit 13.0。保留 PyTorch 对 `sm_121` capability 的提示；
这里的实际 kernel 都已运行，但不推断所有第三方扩展兼容。

后续已将提交 `87428f3` 的课程构建进独立镜像，并在全新 volume 中重跑完整 smoke、
普通测试和课程 Notebook；见 [课程镜像验收记录](IMAGE_VALIDATION.md)。
以下保留原学习卷上的历史验证，不混用两次运行的日志。

## 实测项目

- Online Softmax 的大幅值数值稳定和分块 Attention 与 SDPA 对齐。
- 独立 RMSNorm/RoPE 与书籍实现对齐。
- MHA/MQA/GQA、因果前缀不变、完整/缓存多步对齐。
- Block backward、所有参数有限梯度、超长输入不破坏旧 Cache。
- Triton Add/RMSNorm/Softmax/LayerNorm/SwiGLU/MatMul/Online Attention。
- FP32/FP16/BF16 kernel 多种大小与尾部 Mask，对照独立参考。
- 分页池非连续映射、OOM 原子性、回收、旧 handle 失效。
- 真实小模型的分桶 batch、greedy 对齐、异常清理。
- 精确前缀 snapshot 命中、clone 隔离与 continuation 对齐。
- 模拟 scheduler 的到达、预算、顺序和完成轮次。
- vLLM 探针的 HTTP payload 只用 mock 测试，不冒充服务端实测。

MatMul 全仓库回归曾发现旧测试启用 TF32 的全局设置影响 reference：
现改为独立 CPU FP64 oracle 后按 FP32 误差阈值检查，未用放宽阈值掩盖差异。

## Day 5–6 的补充验证

以下两份 Notebook 已另行执行全部代码单元格，不改变原 25 份课程的计数：

- `learning/day05_config_memory.ipynb`：投影参数形状、Hkv/F 单变量实验、
  T 翻倍时权重/Cache/score 的 1/2/4 倍关系。
- `learning/day06_online_softmax.ipynb`：错误的块内归一化平均、正确 m/l/o 合并、
  块宽变化、大幅值数值稳定，以及 PyTorch 分块 Attention 对齐 SDPA。

固定 Qwen3 配置且 B=1、T=4096、每元素2字节时，计算出的全部层 KV 为448 MiB，
单层朴素 score 为512 MiB；只是公式估算，不是完整进程实测峰值。
手算示例的正确输出约34.9265，错误的块内输出平均约27.3106。
配置工具不加载权重、不分配长序列张量，并以真实小模块核对 projection 参数。

## Triton IR补充验证

2026-11-01 UTC，在现有Spark Jupyter容器的Python GPU内核完整执行
`learning/day17_triton_ir.ipynb`，没有修改运行依赖：

- Triton3.6.0；数值对齐后导出真实TTIR/TTGIR/LLVM IR/PTX/cubin。
- TTGIR目标`cuda:121`，本次PTX为`.version 8.8`、`.target sm_121a`。
- Vector Add的N=1003/BLOCK=256，num_warps=4；报告16寄存器/线程、0 shared bytes。
- 5项专项测试通过，覆盖1/1003/1025元素、不同BLOCK、尾部Mask、拒绝覆盖旧目录。
- 这些产物不等于性能加速证据；本实验未给出加速比。

已执行Notebook：

```text
/workspace/logs/day17-triton-ir-executed-20261101T094814000699Z.ipynb
```

实际编译产物和报告：

```text
/workspace/logs/triton-ir-20261101T094815307170Z/
```

新增 [五方向实践路线](INFERENCE_ENGINEERING_PATH.md)，标明模型优化、算子、系统、
部署与工程验证分别的入口与限制。真实vLLM服务仍未配置，不因IR实验完成而改变其状态。

## Day 5 MLA补充验证

2026-11-01，在现有Spark Jupyter容器完整执行
`learning/day05_mla_intuition.ipynb`，不下载模型，不修改运行依赖：

- 同一组随机低秩参数，两条路径分别展开历史K/V、直接在latent上计算。
- 含独立RoPE分支；比较最终位置的分数、softmax权重与Attention输出。
- 默认CPU FP32配置：分数最大误差约`2.61e-8`，输出最大误差约`8.73e-9`。
- 7个历史位置：物化完整各头K/V为1008字节，latent加位置Key为140字节。
  前者重复拼接各头位置Key；仅为教学数组计数，不是实际DeepSeek压缩率。
- 8项专项测试通过；最终全仓库普通回归为282 passed / 27 skipped / 1 warning。
  保留GB10 capability提示；Ruff、Notebook schema和配套本地链接检查通过。

已执行Notebook：

```text
/workspace/logs/day05-mla-executed-20261101T104038254204Z.ipynb
```

本实验没有实现完整DeepSeek模型、训练质量评估、生产KV更新或性能加速对比。
“等价”仅指同一套低秩参数的两种计算顺序，不保证任意普通MHA无损改为MLA。
教学数据流与参考答案见 [MLA直观讲义](MLA_INTUITION.md)；个人进度仍为Day 5。

## 原始记录

已执行 Notebook 和分日实验报告保存在现有 volume 中，不加入 Git：

```text
/workspace/logs/course-notebooks-20261101/summary.json
/workspace/logs/course-notebooks-20261101/day06_course.ipynb
...
/workspace/logs/course-notebooks-20261101/day30_course.ipynb
/workspace/logs/day06-course.json
...
/workspace/logs/day30-course.json
/workspace/logs/cuda-vector-reduce
```

这些 JSON 包含 `status`，不能仅因 nbclient 没抛异常就把 Day 22 写成通过。
同步前的旧文件保存在 `/workspace/logs/course-sync-backup/<UTC时间>/`，
不会自动覆盖浏览器中尚未保存的编辑。

复现：

```bash
# 宿主机：先编译，再复制；本机架构为 sm_121
nvcc -arch=sm_121 -O2 learning/kernels/vector_reduce.cu -o /tmp/reasoning-vector-reduce
docker compose -f docker-compose.spark.yml cp \
  /tmp/reasoning-vector-reduce reasoning:/workspace/logs/cuda-vector-reduce

# 每次使用新的报告目录，不覆盖上次验证证据
docker compose -f docker-compose.spark.yml exec -T \
  -e CUDA_LESSON_BINARY=/workspace/logs/cuda-vector-reduce reasoning \
  /opt/venv/bin/python -m learning.verify_course --output-dir logs/course-notebooks-rerun
```

全套普通测试在新课程代码可见的环境中运行：

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 python -m pytest -q tests
```

此套件仍有少量下载 fixture，网络受限时按 Spark 教程传入可用代理。
27 个 skipped 没有被改为 passed，也没有执行全部重训练。

## 不作出的结论

- 没有称用户已完成 Day 6–30 或已“精通”。
- 没有称部署了真实 vLLM 或验证其真实调度/分页 kernel。
- 没有称教学 Triton kernel 优于生产 FlashAttention/cuBLAS。
- 没有把模拟 round 写成 GPU tokens/s。
- 没有把小随机 Transformer 当成加载了 Qwen3 权重的语言模型。
- 没有执行本月全部长训练、量化、多卡、多节点或正式大规模服务压测。

材料已具备可复现的学习与面试训练入口；当前继续 Day 5 MLA 自测，
完成个人验收后再进入 Day 6 的推导与独立实现。
