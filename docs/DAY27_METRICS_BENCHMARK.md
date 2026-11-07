# Day 27：端到端Benchmark，吞吐与尾延迟要一起看

前置：[Day 14计时](DAY14_MODEL_BASELINE.md)、[Day 26真实合批](DAY26_BATCHED_ENGINE.md)。

## 1. 图解与数字例子

把一个请求的时间拆开，别把整次耗时当首token：

```text
发送 → 排队 → prefill → 首token → 后续token... → 完成
|<---------- TTFT ---------->|
|<---------------- 端到端耗时 ---------------->|
```

10秒内完成100个输出token，按该口径吞吐为10 output tokens/s；
但一个请求等9秒才见到首token，体验仍可能很差。
高吞吐不自动满足P95/P99延迟目标。
non-streaming只有整次返回时间，无法还原每个token到达时刻。

## 2. 概念与计算契约

TTFT=首token时间减明确定义的请求起点；ITL=相邻输出token时间间隔。
TPOT常按首token后至末token耗时/(输出数-1)估算，但不同工具口径需核对，
输出不足2个时不要机械相除。
请求/s与输出token/s不同；输入长度、输出预算、到达率与并发都要固定或给分布。

本课同步wall-clock比较独立greedy与小模型engine；
没有网络/streaming/EOS、真实SLO或长时间容量压测。
每次生成新Request对象，不能重复用含旧output的请求。

## 3. Jupyter与源码路线

打开 [Day 27 Notebook](../learning/day27_course.ipynb)，查看timing_scope，
分开读engine_latency、sequential_latency与模拟轮次。
源码在 `curriculum._engine_lab`、`bench.measure`、`Engine.run`。
先对齐相同权重/输入/预算与页回收，再读耗时。

写自己的基线表，至少两种输入/输出长度组合与并发；
本课只有3条请求，不能从一张短表推出生产最大并发。

## 4. 调试与优化顺序

1. 固定模型版本/设备/dtype/输入与输出统计，保留raw samples。
2. 预热、CUDA同步、多次重复，确认不是计到异步提交或首次JIT。
3. 拆模型、gather/copy、调度和协议时间，定位而不是猜。
4. 每次只改slots、预算、后端等一个变量，重测正确性与尾延迟。
5. 服务压测另列错误/OOM/取消/超时、稳定运行区间，别只报告成功请求中的最好值。

CPU轮次模拟不能转换成GPU毫秒；速度变慢也应保留，说明瓶颈/规模而不是隐藏结果。

## 5. 自测与参考答案

**问：为何总吞吐升、用户却觉得更慢？**
答：批量等待或长任务竞争可能增加TTFT/尾延迟，应一起看分布与SLO。

**问：median和P99能互相替代吗？**
答：不能，前者描述中心，后者看尾部，且小样本P99本身不稳健。

**问：本Engine benchmark是HTTP服务测试吗？**
答：不是，是本进程小模型同步wall-clock，没有协议/网络/流式指标。

**问：time.perf_counter包CUDA调用就够吗？**
答：若不同步会只测提交；应明确同步或用适当CUDA event并说明口径。

## 6. 验收与资料

- [ ] 交付同负载对照与指标定义，保留比基线更慢的场景。
- [ ] 区分逻辑轮次、模型同步耗时与完整服务指标。

深入：[PyTorch Profiler](https://docs.pytorch.org/docs/2.10/profiler.html)；
[vLLM Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)。
下一课：[Day 28](DAY28_PREFIX_CACHE.md)。
