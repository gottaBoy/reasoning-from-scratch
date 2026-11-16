# 学习产出目录

这里保存 30 天学习过程中的可复现实验和小型实现。

**当前阶段：P2-01个人kernel已通过，进入P2-02共享内存归约。**
先阅读 [Day 16教程](../docs/DAY16_SHARED_REDUCTION.md) 并手算8项折半相加；
当前使用 `kernels/vector_reduce.cu` 的 `block_sum` 作为参考，
P2-02个人candidate尚未接入验收。
P2-01个人文件 `learning/my_vector_add.cu` 已通过6/6及memcheck，
[线程索引教程](../docs/PRACTICE_P2_01_VECTOR_ADD.md)与
`python3 -m learning.practice_vector_add` 保留作复查；
`my_ops.py`、`my_cache.py`、`my_block.py`、`my_model.py` 保留，不覆盖。

以下P1统一模板与命令保留作复习：
查看 [统一注释模板](my_ops_template.py)，只在 `learning/my_ops.py` 中实现：
保留RMSNorm/RoPE，新增causal_attention；模板本身保持未完成，便于复习。
需要更直观的说明时，打开 [中文图解模板](my_ops_template_zh.py)：
RMSNorm用调音台，RoPE用二维指针，Attention用问题查资料来解释，
每个TODO都有中文、shape和手算；两版接口相同，任选一版读，不是两份作业。
完整的文件职责、公式、边界和验收口径统一见
[P1-02实战教程](../docs/PRACTICE_P1_02_ATTENTION.md)。
Notebook只作逐格观察补充，不替代个人实现或验收记录。
用 `python -m learning.local_practice --task attention --candidate learning/my_ops.py` 验收。
见 [进度](../docs/LEARNING_PROGRESS.md) 和 [实战日志](../docs/LOCAL_PRACTICE_LOG.md)。
先看 [每日关键知识与面试索引](../docs/KNOWLEDGE_INDEX.md) 和
[Day 6–30 可执行课程](../docs/COURSE_DAY06_30.md)。
Day 9–30的详细教程、实验入口和个人记录模板见
[剩余22课详细总表](../docs/COURSE_DAY09_30_DETAILS.md)；
现有Notebook已补充图解与参考答案，不替换原有代码或个人验收记录。
遇到 `B/H/T/D`、Q/K/V或不同源码的同名字母时，统一查
[符号与张量形状规范](../docs/NOTATION.md)。

## 当前入口

| 阶段 | 入口 |
| --- | --- |
| 本地实战P1-01 | `practice01_start.ipynb`、`local_practice.py`；CPU先对齐，之后明确请求CUDA |
| 本地实战P1-02 | `my_ops_template.py`的A1–A5、自己的`my_ops.py`、`practice02_attention.ipynb`、`practice_attention.py` |
| 本地实战P1-03 | `my_ops_template.py`的S1–S2/M1–M2、自己的`my_ops.py`、`practice03_layout.ipynb`、`practice_layout.py` |
| 本地实战P1-04 | `my_ops_template.py`的G1–G2、自己的`my_ops.py`、`practice04_gqa.ipynb`、`practice_gqa.py` |
| 本地实战P1-05 | `PRACTICE_P1_05_CACHE.md`、自己的Cache模块、`practice_cache.py`、`--task cache` |
| 本地实战P1-06 | `PRACTICE_P1_06_BLOCK.md`、自己的Block模块、`practice_block.py`、`--task block` |
| 本地实战P1-07 | `PRACTICE_P1_07_BASELINE.md`、自己的`my_model.py`、`practice_model.py`、`--task baseline` |
| 本地实战P2-01 | `PRACTICE_P2_01_VECTOR_ADD.md`、自己的`my_vector_add.cu`、`practice_vector_add.py` |
| 本地实战P2-02 | `DAY16_SHARED_REDUCTION.md`、`kernels/vector_reduce.cu`；个人candidate待接入 |
| Day 3 | `day03_debug.ipynb`、`day03_shapes.py` |
| Day 3 RoPE/SwiGLU | `day03_rope_swiglu.ipynb` |
| Day 4 | `day04_kv_cache.ipynb`、`day04_cache.py` |
| Day 5 | `day05_architecture_map.ipynb`、`day05_moe.py` |
| Day 5 配置与内存 | `day05_config_memory.ipynb`、`config_math.py` |
| Day 5 Attention手算 | [讲义](../docs/ATTENTION_HAND_CALC.md)、`day05_attention_handcalc.ipynb` |
| Day 5 MLA 直觉与对齐 | [讲义](../docs/MLA_INTUITION.md)、`day05_mla_intuition.ipynb`、`mla_algebra.py` |
| Day 6 手算与验证 | `day06_online_softmax.ipynb`、`day06_course.ipynb` |
| Day 7 调度与分页 | [讲义](../docs/DAY07_SCHEDULER_MAP.md)、`day07_vllm_map.ipynb`、`day07_course.ipynb` |
| Day 7 存储与生命周期 | [抽屉图解](../docs/DAY07_KV_LIFECYCLE.md)、`day07_kv_lifecycle.ipynb`、`day07_memory.py` |
| Day 8 RMSNorm/RoPE | [讲义](../docs/DAY08_RMSNORM_ROPE.md)、`day08_rmsnorm_rope.ipynb`、`day08_course.ipynb` |
| Day 17 编译IR | `day17_triton_ir.ipynb`、`triton_ir.py` |

## 已准备的后续目录

```text
mini_transformer/  # Day 8–14
kernels/           # Day 15–21
mini_vllm/         # Day 22–30
```

Day 6–30 每天的 Notebook 为 `day06_course.ipynb` 至 `day30_course.ipynb`。
参考实现可以运行，学习时建议自己重写并与之对齐。CUDA、外部服务未满足时
报告相应状态，不把 skipped 写成通过。

每个实验应该包含：

```text
问题 → 预测 → 实现 → 测试 → Benchmark → 结论
```

不要把“代码运行成功”直接写成“性能更好”或“机制已掌握”。
