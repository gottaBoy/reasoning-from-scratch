# 学习产出目录

这里保存 30 天学习过程中的可复现实验和小型实现。

**个人学习进度：Day 5。** 材料准备完成不代表个人验收通过。
先看 [每日关键知识与面试索引](../docs/KNOWLEDGE_INDEX.md) 和
[Day 6–30 可执行课程](../docs/COURSE_DAY06_30.md)。

## 当前入口

| 阶段 | 入口 |
| --- | --- |
| Day 3 | `day03_debug.ipynb`、`day03_shapes.py` |
| Day 3 RoPE/SwiGLU | `day03_rope_swiglu.ipynb` |
| Day 4 | `day04_kv_cache.ipynb`、`day04_cache.py` |
| Day 5 | `day05_architecture_map.ipynb`、`day05_moe.py` |
| Day 5 配置与内存 | `day05_config_memory.ipynb`、`config_math.py` |
| Day 5 MLA 直觉与对齐 | [讲义](../docs/MLA_INTUITION.md)、`day05_mla_intuition.ipynb`、`mla_algebra.py` |
| Day 6 手算与验证 | `day06_online_softmax.ipynb`、`day06_course.ipynb` |
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
