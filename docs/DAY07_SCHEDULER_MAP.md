# Day 7：请求生命周期、Continuous Batching 与 Paged KV

前置：[Day 6 Online Softmax](DAY06_ONLINE_SOFTMAX.md)。
当前学习位置与个人验收见 [进度记录](LEARNING_PROGRESS.md)。
配套实验：[Day 7 Notebook](../learning/day07_vllm_map.ipynb)。
如果术语不直观，先看 [一条请求与四个角色：工作台、抽屉和目录](DAY07_KV_LIFECYCLE.md)
第0–4节，再运行 [真实CPU页池与小模型实验](../learning/day07_kv_lifecycle.ipynb)。
后者补充实际存取、回收与异常清理；本页原来的scheduler仍只是离散模拟。

## 1. 今天先分清四个对象

```text
请求 Request
  ├─ prompt token
  ├─ decode 预算
  ├─ 状态：waiting / running / finished / cancelled
  └─ 自己的 KV Cache 所有权

Scheduler
  ├─ 决定哪些请求进入本轮
  ├─ 分配 token budget
  └─ 在迭代边界接纳、完成、释放请求

KV Cache Manager
  ├─ 给请求分配物理页/block
  ├─ 记录逻辑位置到物理位置的映射
  └─ 完成/取消/异常时回收

Attention Kernel
  ├─ 读取当前 Query
  ├─ 读取请求可见的历史 K/V
  └─ 计算输出；不负责决定请求是否准入
```

**调度器不是 Attention，分页也不是调度。**
调度器安排“谁现在计算”；Paged KV 管理“历史 K/V 放在哪里”。

常用术语先备注清楚：

| 名称 | 中文 | 本页含义 |
| --- | --- | --- |
| Scheduler | 调度器 | 决定执行次序和预算，不计算Attention分数 |
| Waiting / Running | 等待 / 运行集合 | 请求状态，不是CPU线程数量 |
| Slot | 请求槽位 | 教学模拟中running集合最多容纳多少条请求 |
| Token Budget | 每轮token工作预算 | 本模拟每执行一条请求一次，消耗1单位 |
| Arrival | 到达轮次 | 请求从哪轮起可以被排队/接纳 |
| Block / Page | KV块 / 页 | 每块可存固定数量位置，不是CUDA thread block |
| SLO | Service-Level Objective，服务等级目标 | 如首token或后续token延迟目标 |
| TTFT / TPOT | Time to First Token / Time per Output Token | 首token耗时 / 输出token间隔；本模拟均未测量 |

## 2. 一个请求到底经历什么

把一个请求记成 `prompt + max_new_tokens`：

```text
提交
  ↓
waiting：排队，可能还没有运行
  ↓ 准入并分配Cache
prefill：一次处理 prompt 的多个 token，建立历史 K/V
  ↓ 最后一个prompt位置的logits可采样出第1个输出token
  ├─ 已满足停止条件 → finished，释放资源
  ↓
running decode：每轮通常输入1个新 token，追加K/V，产生下一个token
  ├─ 未完成 → 下一轮
  ├─ 达到停止词/长度 → finished
  └─ 用户取消/不可恢复异常 → cancelled/failed
  ↓
释放 KV 页、输出缓冲和调度状态
```

教学 `scheduler.py` 只模拟 decode 轮次，不执行真实 prefill、模型 forward、
GPU kernel 或物理 KV 写入。因此它的 `round` 不是毫秒，也不是 tokens/s。
真实请求还可能分块prefill、抢占、重新排队或重算；上图是教学主线，不是所有版本的
生产状态机。KV容量不足可以等待、拒绝或触发实现支持的恢复策略，不一定直接失败。
已采样出的最终token也未必需要再次送入forward。

## 3. 具体例子：短、长、晚到请求

固定三个请求：

```text
A：剩余3个decode工作项，在第0轮到达
B：剩余1个decode工作项，在第0轮到达
C：剩余2个decode工作项，在第1轮到达

slots=2，token_budget=2
```

模拟假设请求进入运行集合时已能执行decode，不计prefill、首token采样或KV容量。
每条请求每轮最多执行一个工作项；`tokens`只是剩余工作计数，不是prompt长度。
下面的轮次从0编号，`arrival=1`表示在第1轮开始前到达。

### Sequential

同一时刻只服务一个请求：

```text
round 0：A
round 1：A
round 2：A 完成
round 3：B 完成
round 4：C
round 5：C 完成
```

### Static Batch

初始批次装入 A/B；C 要等当前批次中所有请求完成：

```text
round 0：A、B（B完成）
round 1：A
round 2：A完成
round 3：C
round 4：C完成
```

### Continuous Batching

每轮边界都可把完成请求的空位给等待请求：

```text
round 0：A、B（B完成）
round 1：A、C
round 2：A、C 完成
```

这说明 C 的**接纳时机**不同，不说明 GPU 一定更快。
真实系统还要计 prompt 长度、prefill/decode 竞争、kernel 时间、通信、
KV 页容量、队列公平和尾延迟。

### 对照代码输出，避免差一轮

现有模拟器保留两种编号：

```text
trace中的round       = 从0开始的执行轮编号
first_round[name]    = 首次执行结束时的计数 = 对应round+1
finish_round[name]   = 最后一次执行结束时的计数 = 对应round+1
```

因此C在continuous的 `trace round=1` 首次执行，对应 `first_round['C']=2`；
在static的 `trace round=3` 首次执行，对应 `first_round['C']=4`。
不是C又晚到了一轮，也不是首token延迟2秒/4秒。
**first_round记录首次执行，不是一般意义的准入时间**；预算不足时，
请求已进入running，也可能在下一轮才真正获得执行机会。
本例预算足够，C准入后当轮即可执行，才可从首次执行推断其准入轮次。

## 4. token budget 是什么

`token_budget` 是调度轮允许执行的工作额度，不等于请求最大长度：

```text
每轮预算=2
running=[A,B,C]
本轮最多选择2个 decode 工作项
```

预算太小：调度轮更多、Python/调度开销和等待可能增加。
预算太大：长请求可能占据关键路径，短请求和新请求等待更久。
生产系统通常还会分别考虑 prefill budget、decode budget、优先级、SLO、
最大 batch token 数和 KV 可用容量。
这些预算的具体单位和规则应按目标框架版本查证，不将本页的decode工作项
直接等同于真实框架的一切token计数。

本例把 `token_budget` 从2改1，保持slots=2：
最多两条请求留在running，但每轮只执行一条。三个请求总计6个工作项，
三种策略都会用6轮；continuous不保证任何预算下总轮数都少。
Notebook逐轮检查预算、重复执行和到达时间；不据逻辑轮数推导GPU加速比。

## 5. Continuous Batching 与 Static Batch 的区别

| 维度 | Static Batch | Continuous Batching |
| --- | --- | --- |
| 加入请求 | 常在批次开始前 | 迭代边界可加入 |
| 补充新请求 | 本例等整批结束，即使内部已有空位 | 本例在下一轮边界补入 |
| 利益 | 实现简单、形状规整 | 减少短请求空等，提高资源利用机会 |
| 成本 | 长短请求互相拖慢 | 状态、预算、公平、取消和回收更复杂 |
| 证明方式 | 看批次结果 | 看状态轨迹、延迟和吞吐实测 |

“能把 C 提前放入运行集合”是调度行为证据，不是性能证据。
Static并不意味着已经完成的请求必须继续产生token，也不意味着必须一直保留其KV；
本例已经把B从running中移除，只是不提前填补空位。

## 6. Paged KV 在这里负责什么

连续 KV Cache 可以想成：

```text
请求A：一段连续的大数组
请求B：另一段连续的大数组
```

请求长度动态增长时，连续数组容易产生扩容、搬迁和碎片。
Paged KV 把历史位置切成固定大小的物理 block：

```text
逻辑 token position
    ↓ position // block_size
logical block id
    ↓ block_table[request]
physical block id
    ↓
物理 KV page
```

例如 `block_size=4`，请求的逻辑位置6：

```text
logical_block = 6 // 4 = 1
offset        = 6 % 4 = 2
physical_block = block_table[request][1]
```

请求的逻辑顺序保持不变，物理 block 可以不连续。
位置和block编号都从0开始：逻辑位置6是第7个位置，
logical block 1是第2个逻辑块；offset 2是块内第3个槽位。

分页也不是零浪费。例如已保留7个位置、block_size=4，按需分配需要2块，
总容量8个位置，最后一块有1个空槽；页尾内部碎片仍需计入。
本仓库教学池按请求容量预留页，不是完整生产框架的逐步按需分配方案。
page table只保存地址映射，不把真实KV数值压缩成几个整数。

Paged KV 主要解决分配/映射问题；它不自动实现：

- prefix sharing；
- copy-on-write；
- 抢占和换入换出；
- 生产级 paged GPU Attention kernel；
- 调度公平或请求取消。

本仓库 `learning/mini_vllm/paged_kv_cache.py` 是教学池，
`scheduler.py` 是离散状态模拟；不要把它们称为完整 vLLM。

## 7. 一轮调度应记录什么

至少记录：

```text
round：
running_before：
admitted：
executed：
completed：
cancelled/failed：
waiting_after：
token_budget：
KV pages allocated/freed：
```

这是生产观测/后续实现的建议字段，不是当前 `simulate` 已经提供的所有字段。
当前返回 `trace` 只含round、executed、waiting，以及整体first/finish计数；
没有物理页分配、取消、失败或准入事件明细。Notebook不会假造这些事件。

只打印最终输出，不足以解释调度。要能回答：

1. C 什么时候到达？
2. C 什么时候进入 running？
3. B 完成后哪个资源被释放？
4. 下一轮为什么能接纳 C？
5. 如果没有空闲 KV 页，是等待、拒绝还是 OOM？

## 8. Day 7 自测与参考答案

**问：Continuous Batching 的核心是什么？**

答：在 decode 迭代边界动态更新运行集合，完成请求释放位置，新请求可补入；
不是把所有请求一次性拼成静态大 batch。

**问：为什么短请求完成后能提高利用机会？**

答：它释放计算槽位和 KV 资源，等待请求可以在下一轮进入；但实际吞吐还取决于
模型、batch shape、kernel、内存和调度开销。

**问：Paged KV 和 Continuous Batching 是一回事吗？**

答：不是。前者管理 KV 的物理页和映射，后者管理请求何时进入/退出计算集合；
可以组合，也可以分别存在。

**问：模拟 round 能否直接换算 GPU tokens/s？**

答：不能。round 是离散逻辑事件，没有 wall-clock、GPU 同步、kernel、通信和
真实 prompt/decode 工作量。

**问：prefill 为什么不能完全按 decode 的一 token 逻辑处理？**

答：prefill 通常一次处理多个 prompt token，建立整段历史状态，计算形状和预算
不同；decode 通常每请求每轮追加一个 token，但仍读取全部可见历史 K/V。

**问：B的KV页在本模拟中是否真的被释放？**

答：没有实际KV分配。代码只从running中移除完成的Job，从逻辑上腾出请求槽位。
真实服务要由KV manager配合回收；不能把列表变化写成已测显存回收。

**问：Paged KV能节省每个有效token的KV向量宽度吗？**

答：单纯分页主要改善分配与映射，并不改变每个有效位置K/V的数学维度；
GQA、MLA或KV量化是另外的机制。页尾和元数据仍有成本。

**问：第一周结束后怎样进入Day 8？**

答：先画出模型数据流、Cache与请求生命周期，口述本页自测并填写学习记录；
下一阶段独立实现RMSNorm/RoPE。没有独立验证的项目保留待补，不自动勾选通过。

## 9. 本日范围

今天只完成请求生命周期、调度轨迹和 Paged KV 的职责边界。
没有部署真实 vLLM，没有测 TTFT/TPOT/吞吐，没有实现生产抢占、取消、
SSE、鉴权、限流或 paged GPU kernel。
深入的存储结构、内存生命周期、源码调用链、debug与单变量优化步骤见
[抽屉图解](DAY07_KV_LIFECYCLE.md)，官方阅读链接也按学习顺序集中在该页末尾。

## 一手参考

- [vLLM官方仓库：Continuous Batching与PagedAttention的职责入口](https://github.com/vllm-project/vllm)
- [PagedAttention原论文：Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- [vLLM官方Paged Attention设计说明](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [本课模拟器](../learning/mini_vllm/scheduler.py)、[教学分页池](../learning/mini_vllm/paged_kv_cache.py)

外部框架会演进；本页轨迹与数字按本仓库模拟器生成，不声称逐行复现当前vLLM调度器。
