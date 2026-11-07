# Day 6–30 可执行课程

当前学习位置统一见 [进度记录](LEARNING_PROGRESS.md)，不在各课程副本中固定学习日。
当前行动入口见 [Day 8学习记录](DAY08_STUDY_LOG.md)；Day 7记录见
[Day 7学习记录](DAY07_STUDY_LOG.md)。
本页记录后续材料与参考实验已经准备，不能代替个人理解或面试验收。
所有天的原理、公式、误区、面试问答见 [每日关键知识](KNOWLEDGE_INDEX.md)。
五方向的实践顺序与未完成的部署边界见 [推理工程路线](INFERENCE_ENGINEERING_PATH.md)。
Day 17 可补做 [Triton IR实验](TRITON_IR.md)，不替代原来的数值/性能测试。
Day 9–30现在各有一份 [详细教程](COURSE_DAY09_30_DETAILS.md)，
原Notebook增补图解和自测参考答案，不重建原代码/输出/个人记录。

## 每日入口

在已经启动的 Jupyter 文件浏览器打开 `learning/dayNN_course.ipynb`，
例如从 `learning/day06_course.ipynb` 开始，不需要再启动容器。
每份包含解释、参考源码、实验、自测追问和待填写验收单。

| Day | 知识主线 | 参考实现 | 证据/限制 |
| --- | --- | --- | --- |
| 6 | Online Softmax/Attention | `online_attention.py` | PyTorch 分块数值参考 |
| 7 | 调度机制地图 | `mini_vllm/scheduler.py` | 轮次模拟，不是 GPU 时间 |
| 8 | RMSNorm/RoPE | `mini_transformer/model.py` | 与仓库算子对齐 |
| 9 | 单头 causal Attention | 同上 | causal prefix 与 SDPA |
| 10 | MHA 拆头与合头 | 同上 | 明确 E/H/D |
| 11 | MQA/GQA/MHA | 同上 | 每个结构自身 Cache 对齐 |
| 12 | 请求级 Cache | 同上 | prefill/decode、多步增长 |
| 13 | Block/SwiGLU/backward | 同上 | 有限梯度 |
| 14 | 正确性与 benchmark | `bench.py` | warm-up、同步 wall-clock |
| 15 | CUDA 索引与 Mask | `kernels/vector_reduce.cu` | nvcc 另在宿主机编译 |
| 16 | Shared memory reduction | 同上 | block sums 的最后合并在 CPU |
| 17 | Triton Add/RMSNorm | `kernels/triton_ops.py` | 真实 GPU forward |
| 18 | Softmax/LayerNorm | 同上 | 稳定 reduction、非 affine LN |
| 19 | Tiled MatMul | 同上 | tile 16/32/64、边界 Mask |
| 20 | Attention 参考与 SDPA | `online_attention.py` | 数值先对齐 |
| 21 | Fused causal Attention/SwiGLU | `kernels/triton_ops.py` | 等长 prefill、forward only |
| 22 | 外部真实 vLLM | `vllm_probe.py` | 未配置服务不算通过 |
| 23 | 逻辑/物理分页映射 | `mini_vllm/paged_kv_cache.py` | 非连续块测试 |
| 24 | 连续调度状态 | `mini_vllm/scheduler.py` | budget、arrival、释放 |
| 25 | 分配/回收/OOM | `mini_vllm/paged_kv_cache.py` | 原子分配、旧 handle 失效 |
| 26 | 真实模型分桶合批 | `mini_vllm/engine.py` | 同位置/长度 stack，真实 batch trace |
| 27 | 模拟/真实性能对照 | 同上、`bench.py` | 同 prompt/budget，不能保证更快 |
| 28 | Prefix Cache | `mini_vllm/engine.py` | exact snapshot、clone，不是 COW |
| 29 | 可复现实验报告 | `mini_vllm/README.md` | 模板和版本/环境记录 |
| 30 | 综合验收 | `curriculum.py` | 输出、回收、数值对齐、限制 |

源文件路径均相对 `learning/`。这是可供学习者独立重写的参考实现，
不是提交空 TODO 的目录，也不是把全部生产能力实现完毕。

## 统一执行

在 Jupyter Terminal（已经在容器内）：

```bash
cd /workspace
python -m learning.curriculum --day 6 --device cpu --output logs/day06.json
python -m learning.curriculum --day 21 --device cuda --output logs/day21.json
python -m learning.curriculum --all --device cuda --output logs/course-all.json
```

运行全课程前，按 [CUDA 算子说明](../learning/kernels/README.md) 编译 C++ 示例并设置
`CUDA_LESSON_BINARY`；否则 Day 15/16 会报告依赖未满足。
普通 CPU 模式允许概念/模型/调度实验，但 GPU 算子会明确报告 `requires_cuda`。
不得把所有天都运行过，误写成所有天都通过。
Notebook 从 Jupyter 服务启动的内核中运行，不会自动继承另一个 Terminal
临时 export 的变量。Day 15/16 可在运行单元格前显式设置：

```python
import os
os.environ["CUDA_LESSON_BINARY"] = "/workspace/logs/cuda-vector-reduce"
```

外部服务同理，可在 Day 22 单元格运行前设置 `os.environ["VLLM_BASE_URL"]`，
前提是确有已部署且授权使用的服务。

在宿主机终端操作已运行容器：

```bash
docker compose -f docker-compose.spark.yml exec -T reasoning \
  /opt/venv/bin/python -m learning.curriculum --all --device cuda \
  --output logs/course-all.json
```

## 测试

```bash
python -m pytest -q tests/test_curriculum.py tests/test_learning_kernels.py
```

另有 `tests/test_learning_day03.py`、`test_learning_day04.py`、`test_learning_day05.py`
验证前面的教学工具。GPU 不可用时 kernel 测试会跳过，须如实报告。

Notebook 是用 `python -m learning.build_course_notebooks` 生成的新课程文件。
生成器遇到现有同名文件直接拒绝，避免覆盖学习者笔记；通常无需重新生成。

若要同步宿主机源码至已启动的 Jupyter 工作区，在宿主机执行：

```bash
python3 -m learning.sync_workspace
python3 -m learning.sync_workspace --apply
```

首条只列文件；第二条同步前将不同的旧文件备份到容器
`logs/course-sync-backup/<UTC时间>/`，不删除目标文件。
已打开的 Notebook 不强制重载；先保存自己的浏览器编辑，再选择保留自己的版本
或加载磁盘新版本。同步与 Git 提交是两件事，工具不自动提交。

## 外部 vLLM 不是教学模拟器

Day 22 准备了明确的接口实验，而非把模拟返回值当真实服务：

1. 在独立容器/虚拟环境按官方文档部署可在 ARM64/GB10 上实际运行的 vLLM。
2. 保留原 `reasoning-from-scratch:spark` 及其 volume，不在其中 pip upgrade torch。
3. serving 模型使用目标框架支持的 HF 配置/tokenizer/权重目录，不直接传仓库自定义 `.pth`。
4. 明确获得服务使用许可后，设置 `VLLM_BASE_URL` 指向包含 `/v1` 的地址。
5. 运行探针并保存环境/服务版本。当前自动准备不进行重量级外部安装。

```bash
VLLM_BASE_URL=http://127.0.0.1:8000/v1 \
  python -m learning.vllm_probe --output logs/vllm-api.json
```

该探针访问 models 与 non-streaming completions，不测 TTFT，不证明 paged kernel 性能。
如果服务需鉴权，应先实现/配置相应凭证传递；当前探针没有自动读取 API key。

## 验证与学习状态

自动执行结果见 [课程验证记录](COURSE_VALIDATION.md)，而个人进度仍由
[统一进度表](LEARNING_PROGRESS.md) 维护。
建议按“先口述 → 写预测 → 独立尝试 → 参考对齐 → 改错 → 回答面试追问”学习，
不要一次 Run All 后就把 30 天标为精通。

## 第一手资料

- [PyTorch SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [vLLM 安装](https://docs.vllm.ai/en/stable/getting_started/installation/)
- [vLLM Prefix Cache](https://docs.vllm.ai/en/stable/design/automatic_prefix_caching/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
