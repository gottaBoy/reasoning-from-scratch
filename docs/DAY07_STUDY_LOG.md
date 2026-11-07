# Day 7学习记录：请求生命周期与调度地图

开始日期：2026-11-07（America/Los_Angeles）。
状态：**学习中**。用户明确要求在Day 6教程完善后继续下一天。
当前学习位置见 [进度记录](LEARNING_PROGRESS.md)。

## 今天的核心问题

> 多个请求同时到来时，谁现在运行、谁等待、谁释放资源？

入口：

- [Day 7调度讲义](DAY07_SCHEDULER_MAP.md)
- [Day 7可执行Notebook](../learning/day07_vllm_map.ipynb)
- [容易理解的四角色与抽屉图解](DAY07_KV_LIFECYCLE.md)
- [真实KV存取、复用及小模型调用Notebook](../learning/day07_kv_lifecycle.ipynb)
- [离散调度参考实现](../learning/mini_vllm/scheduler.py)
- [Paged KV教学池](../learning/mini_vllm/paged_kv_cache.py)

## 学习顺序

| 步骤 | 行动 | 个人完成状态 |
| --- | --- | --- |
| 1 | 画 `waiting → prefill → running decode → finished/released` | 待记录 |
| 2 | 用 A/B/C 三个请求跑 sequential/static/continuous | 待记录 |
| 3 | 解释 C 的 arrival、B 的完成和空闲槽位 | 待记录 |
| 4 | 区分 scheduler、KV manager、Attention kernel | 待记录 |
| 5 | 手算 `position // block_size` 与 `% block_size` | 待记录 |
| 6 | 写出 round 不是 ms、不能直接推出 tokens/s | 待记录 |
| 7 | 看A/B借出、写入、B归还、C复用抽屉；区分容量、有效长度与pool内存 | 待记录 |
| 8 | 对照Engine调用轨迹，检查异常后页回收，按图解步骤定位再优化 | 待记录 |

固定实验：

```text
A：3个decode token，arrival=0
B：1个decode token，arrival=0
C：2个decode token，arrival=1
slots=2，token_budget=2
```

这里的token是教学模拟的decode工作项，不包含真实prefill或首token采样。
先画每轮执行集合，再看Notebook输出；round从0编号，
`first_round/finish_round`为对应执行结束后的轮次计数，详见讲义第3节。

第一题：B在第0轮结束后完成，C在第1轮到达，它应该一直等A完成吗？
参考答案：本例静态批次要等；连续批处理可在第1轮补入C。
前提是有请求槽位、工作预算；真实服务还要满足KV容量等条件。

## 自测

- [ ] 能画出一个请求从提交到释放的状态转移。
- [ ] 能解释 static batch 为什么要等整批，continuous batching 为什么可在轮次边界补入。
- [ ] 能说明 token budget、slots、arrival 各自影响什么。
- [ ] 能区分 Paged KV 的物理页映射和 scheduler 的请求准入。
- [ ] 能说明模拟轮次不是 GPU 性能。
- [ ] 能用一条请求解释Request、Scheduler、KV Manager、Attention Kernel各自的职责。
- [ ] 能解释页归还后storage仍保留，以及allocated/reserved与free页数的不同。
- [ ] 能说明FIFO、全预算预留、按位置/长度分桶是本教学引擎的选择，不是通用定律。

```text
我预测三种策略的轨迹：
我实际运行的轨迹：
C 何时进入running：
我发现的一个边界：
仍不理解：
是否进入Day 8：
```

个人运行与口述结果：**尚未登记**。
