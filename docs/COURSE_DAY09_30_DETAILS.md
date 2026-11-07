# Day 9–30逐日详细教程总入口

当前个人学习位置仍为 [Day 8](LEARNING_PROGRESS.md)。
本包是按用户要求提前准备材料，不表示Day 9–30已经完成或个人已通过验收。
每日都包含：图解与数字、概念/计算契约、Jupyter与源码路线、调试优化、自测参考答案、资料。

## 每天怎么开始

```text
读直观图 → 自己算一个小例子 → 找源码对应行 → 在Notebook写预测
      → 执行与参考对齐 → 造一个失败 → 改一个变量 → 写个人结论
```

Notebook保留原代码、单元格ID、输出与个人笔记，只增加有明确标识的图解和答案Markdown单元格。
不需要删除旧Notebook或重新安装torch；GPU/编译器/外部服务依赖仍按实际状态报告。
Day 22默认没有真实服务，因此不是passed；CUDA缺失的课不能用CPU概念阅读冒充GPU验证。

## 第二周：独立模型

| Day | 详细教程 | Notebook | 本日要交付的证据 |
| --- | --- | --- | --- |
| 9 | [单头因果Attention](DAY09_SINGLE_ATTENTION.md) | [运行](../learning/day09_course.ipynb) | 逐行匹配、Mask与SDPA、未来扰动负例 |
| 10 | [多头与布局](DAY10_MULTIHEAD.md) | [运行](../learning/day10_course.ipynb) | 拆合头shape/stride与标签负例 |
| 11 | [GQA/MQA](DAY11_GQA_MQA.md) | [运行](../learning/day11_course.ipynb) | 共享映射、紧凑Cache与各结构自身对齐 |
| 12 | [请求级Cache](DAY12_REQUEST_CACHE.md) | [运行](../learning/day12_course.ipynb) | 多split、位置、矩形Mask与拒绝边界 |
| 13 | [Block与backward](DAY13_BLOCK_BACKWARD.md) | [运行](../learning/day13_course.ipynb) | 两次残差、SwiGLU与有限梯度 |
| 14 | [模型验收/基线](DAY14_MODEL_BASELINE.md) | [运行](../learning/day14_course.ipynb) | 正确性与可复现计时分开 |

## 第三周：GPU算子

| Day | 详细教程 | Notebook | 设备/范围 |
| --- | --- | --- | --- |
| 15 | [CUDA索引](DAY15_CUDA_INDEXING.md) | [运行](../learning/day15_course.ipynb) | nvcc与可运行CUDA binary，六组边界 |
| 16 | [Shared Reduction](DAY16_SHARED_REDUCTION.md) | [运行](../learning/day16_course.ipynb) | 同一binary，明确CPU最终合并 |
| 17 | [Triton行算子](DAY17_TRITON_ROWS.md) | [运行](../learning/day17_course.ipynb) | CUDA、forward-only；另有IR补充 |
| 18 | [稳定Softmax/LayerNorm](DAY18_STABLE_ROW_OPS.md) | [运行](../learning/day18_course.ipynb) | padding、精度、非affine LN |
| 19 | [Tiled MatMul](DAY19_TILED_MATMUL.md) | [运行](../learning/day19_course.ipynb) | M/N/K尾部、tile和独立oracle |
| 20 | [Attention基准](DAY20_ATTENTION_ORACLE.md) | [运行](../learning/day20_course.ipynb) | CPU数学；CUDA才额外跑Triton |
| 21 | [融合算子](DAY21_FUSED_KERNELS.md) | [运行](../learning/day21_course.ipynb) | 等长causal prefill，不含backward/GQA |

## 第四周：推理系统

| Day | 详细教程 | Notebook | 本日要分清 |
| --- | --- | --- | --- |
| 22 | [真实服务边界](DAY22_SERVING_BOUNDARY.md) | [运行](../learning/day22_course.ipynb) | 未配置、mock、授权真实服务是三种状态 |
| 23 | [分页地址](DAY23_PAGED_ADDRESSING.md) | [运行](../learning/day23_course.ipynb) | 逻辑/物理顺序、目录与数据 |
| 24 | [调度与公平](DAY24_SCHEDULER_FAIRNESS.md) | [运行](../learning/day24_course.ipynb) | slots、budget、arrival、轮次与服务时间 |
| 25 | [所有权/OOM](DAY25_OWNERSHIP_OOM.md) | [运行](../learning/day25_course.ipynb) | 原子拒绝、旧句柄、请求与pool生命周期 |
| 26 | [真实分桶合批](DAY26_BATCHED_ENGINE.md) | [运行](../learning/day26_course.ipynb) | B轴隔离、真实forward与循环单请求 |
| 27 | [指标/对照](DAY27_METRICS_BENCHMARK.md) | [运行](../learning/day27_course.ipynb) | TTFT/ITL、wall-clock与模拟、尾延迟 |
| 28 | [Prefix Cache](DAY28_PREFIX_CACHE.md) | [运行](../learning/day28_course.ipynb) | 精确clone快照，不是自动COW共享 |
| 29 | [可复现报告](DAY29_REPRODUCIBLE_REPORT.md) | [运行](../learning/day29_course.ipynb) | 基线、失败、版本、统计和个人贡献 |
| 30 | [综合答辩](DAY30_CAPSTONE.md) | [运行](../learning/day30_course.ipynb) | 端到端契约、实测边界与下一阶段 |

## 每日个人记录

参考答案已经在各讲义第5节和Notebook中，不需要独自猜答案。
但先写自己的推导，再展开参考；每个学习日另存一份个人记录：

```text
Day / 日期：
我先预测：
我独立实现或修改：
输入shape / dtype / 设备：
正确性oracle / 容差 / 最大误差：
一个失败例与原因：
测的是模拟 / CPU / GPU / 外部服务：
性能口径与原始日志：
我能口述：
尚未满足的依赖：
是否进入下一天：
```

验收勾选由学习者完成，助手执行过的参考实验另见 [验证记录](COURSE_VALIDATION.md)。
用户有新的进度决定时只更新 [统一进度](LEARNING_PROGRESS.md)，不把每个文件写死某一天。

## Notebook补充与重跑

`learning.course_guides`按本包固定六节结构读取讲义，仅管理两个新增Markdown单元格。
默认只预览，--apply前先备份差异Notebook，不修改原代码、输出或其他cell ID：

```bash
python -m learning.course_guides
python -m learning.course_guides --apply
```

后续只重跑指定范围并使用新日志目录；详情见 [课程执行入口](COURSE_DAY06_30.md)。
例如在当前Spark服务内，先配置已编译的 `CUDA_LESSON_BINARY`，
再运行指定范围（不重建原Notebook）：

```bash
python -m learning.verify_course --days 9 10 11 12 13 14 \
  --output-dir logs/day09-14-verification-new
```

验证工具把旧的顶层分日报告复制进本次目录的prior-reports，
再保留本次分日JSON、执行Notebook和summary；已存在summary时拒绝覆盖。
原书全量训练、真实vLLM部署、量化、多卡和生产压测并非本次提前准备的已完成实验。
