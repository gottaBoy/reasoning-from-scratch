# Day 30：综合项目与答辩，把一次请求从头讲到回收

前置：[Day 9–29详细教程](COURSE_DAY09_30_DETAILS.md)。本页不是“30天自动精通证书”。

## 1. 图解与数字例子

回到同一条请求，把所有技术放对位置：

```text
文本 → tokenizer → Request → 准入/调度 → Prefill
                 ↓                      ↓
              停止条件          权重计算、RoPE/GQA、KV写入
                 ↑                      ↓
                 └─ 采样 ← logits ← 多轮Decode
                                          ↓
                              完成/异常 → 请求状态及页回收
```

优化的对象不同：GQA/MLA影响KV表示，FlashAttention影响Attention中间IO，
Paged KV影响分配映射，continuous batching影响运行集合，prefix cache影响重复前缀计算。
它们可以组合，但名字同时出现不代表所有实现和生产能力都已经具备。

## 2. 概念与计算契约

本项目是随机小Transformer、显式gather分页、同长度/位置分桶、独立greedy对齐，
加上精确clone前缀快照与离散调度模拟。
没有真实生产paged kernel、ragged batch、抢占/offload、外部取消、多卡、
量化、完整协议/鉴权/限流/持续压测。

Day 30统一参考会执行Engine、Cache对齐与同步计时；
它没有自动调用所有课程测试、部署Day 22或验证全部prefix机制。
不要因status=passed声称整个生产平台通过。

## 3. Jupyter与源码路线

打开 [Day 30 Notebook](../learning/day30_course.ipynb)，读scope、Engine输出与页回收结果。
结合 [Day 28](DAY28_PREFIX_CACHE.md) 的独立快照测试、
[Day 25](DAY25_OWNERSHIP_OOM.md) 的错误路径，以及全仓库普通回归。
输出一份Day 29模板的最终报告，附代码版本与日志目录。

检查每项证据属于哪类：CPU数学、GPU kernel、模拟、真实小模型、HTTP mock、
授权外部服务。Day 22没有服务时应继续保留未配置，不为“全绿”删掉限制。

## 4. 调试与优化顺序

1. 固定一次请求的token IDs、各层shape、位置和所有权，先定位错误层。
2. full/cached、分桶/独立、prefix续写分别按同权重对齐。
3. 压力与异常另测，回收不变量与输出正确都要有证据。
4. profiling后选单一瓶颈，重做同负载对照与尾延迟。
5. 把未覆盖能力列成下一阶段小项目，不一次承诺全部生产能力。

可选后续课题：真实服务TTFT/ITL、量化误差、CUDA Graph地址/shape契约、
speculative decoding接受验证、多卡通信。每项先独立研究与安装隔离，不自动认为本月已实现。

## 5. 自测与参考答案

**问：当前项目与生产vLLM差在哪里？**
答：权重加载/模型规模、真实paged kernel、异长合批、抢占/取消、
分布式/量化、协议、安全、监控和服务压测等；要按实测范围逐项回答。

**问：怎样证明“从Cache读到了正确请求”？**
答：请求身份、逻辑/物理映射、有效长度和实际数值对照，不能只看shape。

**问：最佳优化方向是什么？**
答：由当前性能/质量/成本目标和profiling证据决定，不按热门技术列表决定。

**问：材料齐备和个人完成如何区分？**
答：课程提供参考；个人要独立口述、改写、构造失败、复现并在新输入上解释权衡。

## 6. 验收与资料

- [ ] 一次请求端到端口述、源码追踪、数值/错误/回收证据。
- [ ] 一份可复现报告，含失败和不支持表；个人验收由本人填写。
- [ ] 下一步只选一个有基线的实际问题，不自动升级掌握程度。

深入：[面试训练](INTERVIEW_PRACTICE.md)；
[五方向工程路线](INFERENCE_ENGINEERING_PATH.md)；
[vLLM官方架构](https://docs.vllm.ai/en/latest/design/arch_overview/)。
