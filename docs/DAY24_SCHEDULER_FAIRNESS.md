# Day 24：调度与公平，不只是把Batch填满

前置：[Day 7调度](DAY07_SCHEDULER_MAP.md)、[Day 23分页](DAY23_PAGED_ADDRESSING.md)。

## 1. 图解与数字例子

运行名额像两张工作台，预算像这一轮允许做的工序数量：

```text
slots=2：最多两条请求留在running
budget=1：本轮只让其中一条执行一次
arrival：什么时候开始有资格排队
```

A=3、B=1在round=0到达，C=2在round=1到达；slots=2，共6个decode工作项。
budget=2时本例三种策略为6/5/3轮；
budget=1时本例三者都为6轮。连续补位不是任意条件下总轮数更少的定律。
晚到请求何时开始、长请求是否饿死，与总轮数同样重要。

## 2. 概念与计算契约

`simulate`按arrival稳定排序，waiting/running为队列，
每轮最多对running中的每条请求执行一次，未完成者轮转回队尾。
static只有running为空才装入新批次；sequential容量固定1。
trace round从0编号，first_round/finish_round记录执行结束计数，即round+1。

本模拟没有真实prefill、页分配、服务网络、GPU时间或取消API。
first_round是首次执行，不是独立admission事件；预算不足可先准入后执行。
与Engine的FIFO页准入/分桶实现是两条不同教学路径，别把模拟配置当Engine参数。

## 3. Jupyter与源码路线

打开 [Day 24 Notebook](../learning/day24_course.ipynb)，默认A/B/C/D工作负载；
复习 [Day 7逐轮Notebook](../learning/day07_vllm_map.ipynb) 的另一组A/B/C预算对照。
两组名字相同但tokens/arrival不同，不能拿两份表直接逐行比数值。
定位while循环中的到达、装入、执行、完成和轮转五步。

自己固定一个到达序列，只改slots或budget，检查总工作守恒、到达前不执行、
每轮无重复。给极晚到请求构造空闲轮，别把空闲轮错误计作已执行token。

## 4. 调试与优化顺序

1. 先定义公平目标：FIFO、最大等待或优先级，而不是口头说“最优”。
2. 检查arrival、first/finish与每轮executed，排除编号差1。
3. 同工作负载比较等待与完成分布，不只总轮数。
4. 概念上研究chunked prefill/aging/抢占，但未实现的逐项标注。
5. 进入真实服务后测P95/P99、错误率和吞吐，不能把round换成毫秒。

较大token预算也不是无限并发：slot、KV容量、形状兼容与硬件都会限制实际执行。

## 5. 自测与参考答案

**问：slots和budget相同吗？**
答：前者限制运行集合容量，后者限制本轮实际工作量。

**问：first_round=2而trace round=1矛盾吗？**
答：不矛盾，前者是完成首次执行后的计数，后者从0编号。

**问：吞吐最高一定最公平吗？**
答：不一定，大请求饥饿或短请求尾延迟可能很差，要先定义服务目标。

**问：这里能证明prefill优先或decode优先哪种更好吗？**
答：不能，本模拟没有prefill成本，只能验证所实现的decode逻辑。

## 6. 验收与资料

- [ ] 交付一组可手算轨迹、预算负例和公平目标说明。
- [ ] 把模拟指标与服务指标分开，不预设策略优胜。

深入：[vLLM架构](https://docs.vllm.ai/en/latest/design/arch_overview/)；
[调优权衡](https://docs.vllm.ai/en/latest/configuration/optimization/)。
下一课：[Day 25](DAY25_OWNERSHIP_OOM.md)。
