# Day 26：真正的合批，不是for循环调用两次单请求

前置：[Day 12 Cache](DAY12_REQUEST_CACHE.md)、[Day 25所有权](DAY25_OWNERSHIP_OOM.md)。

## 1. 图解与数字例子

同批加工可以共用模型调用，但不能把不同人的历史串成一句话：

```text
A历史 [1,Hkv,3,D] ─┐
B历史 [1,Hkv,3,D] ─┴→ Batch [2,Hkv,3,D]
                          ↓ 一次model forward
                    按Batch行拆回A/B
```

A/B的prompt内容不同但同长，可一起prefill。
若A已缓存3位置、B刚到需要prefill 2位置，
本实现按 `(cache_position,input_length)` 分桶，同一tick调用两次不同桶，
不是强行stack形状不兼容的请求。

## 2. 概念与计算契约

Request保存name/prompt/max_tokens/arrival/output。
Engine校验输入，按FIFO申请最大容量页，按位置/输入长分桶，调用model(ids,BatchCache)。
BatchCache.get把每请求的gather结果沿batch拼接；
update按batch切片写回各自页。生成用最后位置argmax，无EOS提前停止。

容量是prompt长P+输出上限G-1，最后输出未再送回模型。
Engine复用一份权重，但不同请求KV有独立所有权；PrefixStore并未自动接入。
没有ragged batching、真实paged kernel、多租户取消或生产协议。

## 3. Jupyter与源码路线

打开 [Day 26 Notebook](../learning/day26_course.ipynb)，读outputs和batches。
参考实验逐请求独立greedy作oracle，再与真实小模型Engine输出一致性比较，
最后检查全部物理块归还。
定位Engine.run的buckets、BatchCache.get/update、容量计算及finally。

先自己画两条同长prompt的batch，再改变其中一条长度，预测batch trace如何变化。
细化跟踪可运行 [Day 7真实小模型实验](../learning/day07_kv_lifecycle.ipynb)。
随机模型ID输出只证明路径，不评价文字质量。

## 4. 调试与优化顺序

1. 比较每请求完整输出ID，不只比较总token数。
2. trace中有真实batch_size>1，才能证明不是循环单请求假合批。
3. 检查Cache batch切片和request ID，防止跨请求串历史。
4. 逐层对齐后再profile gather、torch.cat、Python桶与model成本。
5. 再研究更多形状的packed/ragged路径，需新Mask与位置契约，不直接删除分桶检查。

## 5. 自测与参考答案

**问：两条用户历史能沿T拼接吗？**
答：不能这样冒充batch，它会改变上下文语义；应沿B并保证请求隔离。

**问：一个tick有多个model调用正常吗？**
答：在此分桶实现正常，一个桶一次调用，不同桶分开。

**问：P=3、G=3为什么预留5位置？**
答：prefill给首输出，后续只反馈前G-1个输出，最终token不再次forward。

**问：合批一定更快吗？**
答：不一定，小模型与gather/调度成本可能抵消收益，需要Day 27测量。

## 6. 验收与资料

- [ ] 输出独立greedy对齐及batch trace，核对身份与回收。
- [ ] 解释此引擎分桶限制与生产系统差距。

深入：[本仓库Engine](../learning/mini_vllm/engine.py)；
[vLLM架构](https://docs.vllm.ai/en/latest/design/arch_overview/)。
下一课：[Day 27](DAY27_METRICS_BENCHMARK.md)。
