# Day 14：模型验收与基线，先证明同一件事，再比较快慢

前置：[Day 9–13](KNOWLEDGE_DAY08_14.md)。参考模型是小随机Transformer，不是预训练LM。

## 1. 图解与数字例子

把Benchmark理解成同一条跑道上的计时，不是一个人跑100米、另一个跑10米：

```text
同权重 + 同输入 + 同输出预算 + 同dtype/设备
     ↓
full/cached与因果性先对齐
     ↓
预热 → 同步 → 重复采样 → 中位数/范围
```

prompt长8、输出4时，当前generate先prefill 8位置，
再把前三个输出逐个送回，最后一个输出不再forward。
因此缓存最多11位置，不是12；计时覆盖的是这整次生成，而非单token耗时。

## 2. 概念与计算契约

本仓库 `measure` 返回median_ms、min_ms、repeats、峰值allocated等。
CUDA前后同步，测的是端到端同步wall-clock，不是kernel-only CUDA event时间。
prefill与generation两种计时的工作内容不同，不应直接相除宣布Cache加速比。
CPU报告peak_allocated为None，不是“内存占用0”。

绝对误差、相对误差、top1一致是不同证据；正确性阈值应结合dtype与参考精度。
固定种子不能消除所有设备/算法浮点差异，也不证明语言质量。

## 3. Jupyter与源码路线

打开 [Day 14 Notebook](../learning/day14_course.ipynb)，先读Cache对齐结果，再看两个延迟字段。
定位 `learning/bench.py` 与Mini `generate`，数清到底执行多少次forward。
新建自己的小表：输入长、输出预算、dtype、设备、预热数、重复数、误差、时长口径。
用新文件名保存报告，不覆盖第一次失败或较慢结果。

## 4. 调试与优化顺序

1. 先跑CPU正确性，增加T=1、非整齐T、多个Batch与多split。
2. 检查计时函数调用前是否重置请求级Cache，别复用上次历史。
3. 首次JIT/分配在预热范围；测启动延迟时另立指标。
4. 分析CPU调度、模型forward和内存分配，再选择一个瓶颈。
5. 优化后复跑同一oracle与同一工作负载，报告更慢的情况。

`max_memory_allocated`是当前设备allocator统计，不等于系统总内存；
峰值包含测试当时仍存活的张量，不能不减基线就称为某个操作的增量内存。

## 5. 自测与参考答案

**问：一次生成4个token的ms可以直接当TPOT吗？**
答：不能，含prefill、首token及多个decode，应按明确定义拆阶段或记录token时间戳。

**问：不同prompt长度比较谁更快合理吗？**
答：可作为不同场景报告，不能作为同工作负载的优化对照。

**问：为何不用一次最短耗时作为结论？**
答：会受缓存、调度、噪声影响，应预热、多次测量并报告统计口径。

**问：本周通过意味着什么？**
答：能独立说明并验证模型核心、Cache和梯度，不代表随机模型有语言能力或生产性能。

## 6. 验收与资料

- [ ] 交付一份能复现的输入/权重身份、误差与计时表。
- [ ] 区分prefill、generation、kernel-only及端到端时间。
- [ ] 第二周个人验收未完成的保留待补，不替学习者打勾。

深入：[PyTorch CUDA语义](https://docs.pytorch.org/docs/2.10/notes/cuda.html)；
[Benchmark参考](../learning/bench.py)。
下一课：[Day 15](DAY15_CUDA_INDEXING.md)。
