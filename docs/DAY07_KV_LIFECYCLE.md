# Day 7图解：用“工作台、抽屉、目录”看懂推理系统

今天仍是Day 7，不提前标记Day 8。
先看本页，再运行 [抽屉实验Notebook](../learning/day07_kv_lifecycle.ipynb)；
原 [A/B/C调度图](DAY07_SCHEDULER_MAP.md) 保留作为上一层入口。

你要抓住的主线只有一句：

> 一份模型负责计算，多条请求各有自己的历史记录；
> 调度器安排谁来算，缓存管理器安排记录放哪，算完后归还空间。

## 0. 先只看一条请求：四个角色分别说什么

假设用户问“帮我解释KV Cache”，我们不假定这句话实际会切成几个token；
为方便手算，另设一个已经编码好的教学输入：

```text
R1：prompt IDs=[1,2,3]，最多输出2个token
一抽屉放2个历史位置
```

把这次生成看成“带任务单去共享工作室加工”：

| 角色 | 形象理解 | 这次具体做什么 | 不负责什么 |
| --- | --- | --- | --- |
| Request | 任务单 | 记住R1、输入ID、输出上限和已输出内容 | 不是一个独立模型或GPU线程 |
| Scheduler | 排班员 | 看槽位、预算与可用KV空间，决定这轮让R1运行 | 不计算Q/K/V匹配分数 |
| KV Cache Manager | 抽屉管理员 | 分配位置、保存目录、管理使用与归还 | 不决定“回答应该是什么” |
| Attention Kernel | 实际加工计算的一道工序 | 将本轮Q与可见K匹配，按权重汇集V | 不负责排队、释放请求或把输出解码为文字 |

完整的一次往返如下：

```text
1. Request：
   “我是R1，这是输入[1,2,3]，最多给我2个输出token。”

2. Scheduler：
   “现在有运行名额，也能安排KV空间，这轮处理R1。”

3. KV Cache Manager：
   “为R1安排抽屉，目录记为[物理块4,物理块1]。”
   目录指路，真正的K/V向量由模型计算并写入；管理员不从文字猜出KV。

4. 模型执行：
   先算各层Q/K/V，再由Attention Kernel读取可见历史做加权汇集。
   还要经过输出投影、FFN等层，最后LM Head给出logits，采样器/argmax选出y0。

5. 下一轮：
   任务单已有输出[y0]。
   Scheduler再次安排R1；模型输入y0，追加它的KV，查询历史，选出y1。

6. Request输出达到2个：
   Scheduler/执行引擎确认完成，KV Manager归还R1的独占抽屉。
   任务单输出[y0,y1]，不再把最后的y1送回模型。
```

**Attention Kernel只是一道工序，不是整台语言模型。**
它给出向量，不直接给出文字；logits、采样、解码属于后续环节。
上面物理块号是示意，不声称与后面A/B/C实验相同。

本教学引擎预留最大容量；此处P=3、G=2，最多缓存 `P+G-1=4` 个位置，
2个抽屉即可。其他框架可能按需扩页，不都一次预留最大长度。

为什么需要排班员？如果R2也到来：
R1/R2可以共用一份模型规则、在同一批次计算，但各自只读写自己的KV目录；
完成的请求把位置让给新请求，而不是每个用户都启动一份模型。

## 1. 先看一间共享工作室

```text
模型权重 = 一份共用的加工规则，通常不为每个请求复制一套

GPU计算 = 共享工作台
    ↑
Scheduler = 排班员：本轮让A、B上台，C先等待

KV pool = 一座固定大小的存储柜
    ├─ 抽屉0：可存2个位置的KV记录
    ├─ 抽屉1：可存2个位置的KV记录
    ├─ ...
    └─ 抽屉5：可存2个位置的KV记录

每条请求 = 自己的一本目录
    A目录：[抽屉0，抽屉1]
    B目录：[抽屉2，抽屉3]
```

工作台、抽屉是理解角色的比喻，不是独立硬件设备：
权重和KV都可能位于同一GPU可访问内存中，只是用途、所有权和存活时间不同。
KV里不是对话文字，也不是“token编号的备份”，而是每层已算出的浮点K/V向量。
目录只记录位置，不能代替抽屉里的向量数据。

## 2. 抽屉里面究竟存什么

对一个普通GQA层、某个历史位置：

```text
隐藏向量h_t
    ├─ K投影、必要的Norm/RoPE → K_t
    └─ V投影                 → V_t

K_t：下次Query来时，用来计算匹配程度
V_t：按匹配权重取走的内容
```

本仓库小Transformer缓存旋转后的K和投影后的V；
Q只用于当前查询，不存进历史KV。MLA等其他架构的存储表示另查实现。

一个“位置槽位”不止一个数：

```text
位置t：
  第0层 → K的Hkv*D个数 + V的Hkv*D个数
  第1层 → 同样一组
  ...
```

为了让数字小到能亲眼看到，实验选：

```text
6个抽屉，每抽屉2个位置
2层，1个KV头，每头2个数，FP32每数4 bytes

每抽屉 = 2位置 * 2层 * 2份K/V * 1头 * 2维 * 4 bytes
       = 64 bytes
总柜子 = 6 * 64 = 384 bytes
```

这是教学数组大小，不是完整模型或真实进程内存。
真实模型的每抽屉可能更大；MLA、KV量化、不同层布局不能盲套这组配置。

## 3. 把比喻翻译成真实数据结构

现有 `PagedPool` 就有三样核心成员：

| 比喻 | 代码成员 | 实际内容 |
| --- | --- | --- |
| 一整座存储柜 | `pool.storage` | 一块预先分配的PyTorch张量，真的存KV数值 |
| 还没借出去的抽屉号 | `pool.free` | `deque`空闲块队列，如 `[0,1,2,3,4,5]` |
| 谁借了哪些抽屉 | `pool.requests` | 请求ID到`PagedRequest`的字典 |
| A的目录 | `state.table` | 请求自己的逻辑块到物理块映射，如 `[4,5,2]` |
| 能放多少位置 | `state.capacity` | 预留容量，可能大于已写入长度 |
| 真正写到哪里 | `state.lengths[layer]` | 每层的有效KV位置数，不能越界读未写区域 |
| 借阅证还有效吗 | `state.active` | 释放后为False，旧handle访问会被拒绝 |

本仓库物理张量的轴顺序是：

```text
storage[
    physical_block, layer, K_or_V, kv_head, offset_in_block, feature
]

实验shape = [6,2,2,1,2,2]
```

**为什么没有B轴？** 这是所有请求共享的物理块池；
请求身份通过各自block table映射，不是所有请求共用一份历史。
读取某请求的一层后，教学代码gather成 `[1,Hkv,T,D]`，1才是该请求的Batch大小。

这是本仓库布局，不是vLLM所有版本或所有kernel的统一排列。
有的系统按层建池、分开存K/V或采用特殊打包布局，先看实际shape和stride。

## 4. 跟着A/B/C走完一遍内存生命周期

**这里A/B/C是存储实验的新请求，不沿用上一页的decode轮次数。**
抽屉有2个位置；`A0`指A的第0个逻辑块，`空`指不属于活跃请求。

### 第一步：柜子先建好，不等请求来才建

```text
物理抽屉号：  0    1    2    3    4    5
归属：      [空] [空] [空] [空] [空] [空]

free = [0,1,2,3,4,5]
storage = 已存在的384-byte张量
```

启动时创建pool；请求结束时通常归还块给pool，不拆掉整座柜子。

### 第二步：A/B到来，先借抽屉

两条请求各预留4个位置，因此各借2个抽屉：

```text
物理抽屉号：  0    1    2    3    4    5
归属：      [A0] [A1] [B0] [B1] [空] [空]

A.table = [0,1]，A.capacity=4
B.table = [2,3]，B.capacity=4
free = [4,5]
```

这一步只分配所有权，KV有效长度仍是0。**预留4个，不等于已经写入4个。**
本教学池按最大容量预留，便于先验证所有权；不是生产按需扩页。

### 第三步：Prefill写入，Decode接着写

```text
A先写3个位置：抽屉0放A位置0/1，抽屉1放A位置2，另一个槽位待写
B先写2个位置：抽屉2放B位置0/1，抽屉3仍待写

A再增加1个位置：写到抽屉1的第1个槽位
```

由于A已预留4个位置，3变4时不需再借抽屉。预留到6、9的策略另当别论。
每个位置要在每一层分别更新；实验先把全部层写完，再观察请求整体长度。

### 第四步：B结束，归还抽屉，不搬动A

```text
物理抽屉号：  0    1    2    3    4    5
归属：      [A0] [A1] [空] [空] [空] [空]

free = [4,5,2,3]     # 本实现把归还块追加到队尾
B.active = False
```

旧B目录句柄不能再读；A仍然用自己的抽屉0/1，没有搬家。
**归还不是擦除**：物理数组中的旧数值可能还在，但不再是有效B状态。
新请求有效长度从0开始，不能把旧内容作为自己的历史读出。
这不是安全擦除或多租户隔离保证；拿到原始storage仍可绕过教学handle。

### 第五步：C到来，借3个不必连续的抽屉

C预留6个位置，拿到队首三个空块：

```text
物理抽屉号：  0    1    2    3    4    5
归属：      [A0] [A1] [C2] [空] [C0] [C1]

C.table = [4,5,2]，free=[3]
```

虽然C的抽屉散在三个地方，C的逻辑历史仍按0、1、2、3、4、5排序。
目录负责翻译，无需先腾出一整段连续空间。

现在想找C的逻辑位置4：

```text
每抽屉2位置 → logical_block=4//2=2，offset=4%2=0
C.table[2]=2 → 找物理抽屉2的第0个槽位
```

注意两处2只是本例碰巧相同：前一个是逻辑块编号，后一个是物理块编号。
实验会写不同浮点标记，验证读出的是C的数据，不是曾放在这里的B数据。

### 第六步：空间不足，先拒绝，不能借一半

现在只剩1个空抽屉，D却要2个：

```text
先检查：need=2，free_count=1
    ↓
MemoryError；D不进入requests，空闲队列与A/C数据都不改变
```

这个异常表示教学页池容量不足，不等于torch分配CUDA张量失败。
生产准入策略可选择等待/拒绝/抢占等；本课不替外部服务指定统一行为。

### 第七步：A/C都结束，柜子留下备用

```text
所有抽屉归还 → requests为空、6块可重新分配
但storage还是原来那块384-byte张量，地址不变
```

这是本实验最重要的观察：
**请求生命周期结束了，页池生命周期没有结束。**

## 5. 调用策略：不是每个请求都各跑一份完整模型

先看本仓库真实的小引擎调用链：

```text
Engine.run(requests)
    ↓ 校验prompt、预算、上下文、每请求是否可能装入池
pending → 到达轮次 → waiting
    ↓ 有running槽位 + 有足够页
pool.reserve(request_id, capacity)
    ↓
按(cache_position, 本次输入长度)分桶
    ↓ 相同桶的请求堆成Batch
model(ids, BatchCache(states))
    ↓ 每层
get历史 → gather页 → Q/K/V计算 → append → update写回新位置
    ↓
取最后位置logits → greedy argmax → output追加token
    ↓
完成的请求release；未完成的请求进入下一轮
```

把同一桶的请求看成“工作台同批加工”，不是让A读B的抽屉：
`BatchCache.get`将各请求KV临时拼Batch，
`BatchCache.update`再按Batch行分别写回各自的页。
这个实现会复制/gather数据，不能当成生产PagedAttention的原地间接读取kernel。

**为什么分桶？** 本教学模型要求同一Batch的Cache位置、本次输入长度一致。
同一个tick可能因此执行多次model调用；不是running有两条就一定一次合批。
生产框架可用packed/ragged表示处理更多形状，这不是所有系统都必须分桶的定律。

### 首token与容量：不要多算一个位置

设prompt长P，最多输出G个token，本引擎不做EOS停止：

```text
prefill输入P个prompt位置 → 得到第1个输出token
把第1个输出送回模型 → 得到第2个输出
...
最后一个输出不再送回模型，已经达到G

最多缓存的位置 = P + G - 1
```

例如P=3、G=3，缓存最多5个位置，而输出3个token。
不是5个输出，也不是必须存6个位置。
这只是当前自回归循环的规则，不将其盲套到speculative decoding等路径。

### 当前策略与可选优化，别混写

| 策略 | 直观理解 | 代价与边界 |
| --- | --- | --- |
| 当前FIFO准入 | 先排队的先申请抽屉 | 队首大请求暂时装不下时，本引擎break，后面小请求也会等 |
| 可选fit-first | 先让装得下的小请求进入 | 可能饿住大请求，需aging/公平限制；当前未实现 |
| 当前全预算预留 | 一次借够最大需求 | OOM简单、无需运行中扩页，但可能过度预留 |
| 可选按需扩页 | 写到边界再借下一抽屉 | 空间利用更高，但增长时不足怎么办必须设计 |
| Chunked prefill | 长prompt分段，别长期霸占工作台 | 预算/调度更复杂，吞吐与延迟需一起测；当前未实现 |
| Prefix复用 | 同前缀少算一段历史 | 必须核对模型、前缀身份、引用计数/写时复制和驱逐 |

`scheduler.py`的轮转decode模拟和`engine.py`的真实小模型分桶执行是**两个教学工具**：
Engine并未调用simulate，也没有暴露相同的token_budget策略。
`PrefixStore`只是另一个独立的clone快照工具，不是自动接在Engine里的节省页实现。

## 6. 内存生命周期：看四只不同的表

```text
服务启动：
  模型权重 ─────────────────────────── 服务关闭
  KV物理页池 ───────────────────────── 服务关闭/显式重建

请求A准入：
  A的页所有权 ───── prefill ─ decode ─ 释放归还池

某一轮计算：
  Q/临时score/算子buffer ─ 本轮消费完 ─ 不再被活对象引用后可回收/复用
```

| 你查看的指标 | 回答的问题 | 请求结束后一定下降吗 |
| --- | --- | --- |
| free页数 / live请求表 | 柜子还有多少抽屉可借 | 正常独占页回收后应恢复；共享prefix另看引用 |
| 页池storage字节数 | 整座柜子多大 | 不一定，预分配池可一直存在 |
| `torch.cuda.memory_allocated()` | 当前活跃CUDA张量占多少字节 | 池仍被引用就不会因页逻辑回收而减少 |
| `torch.cuda.memory_reserved()` | PyTorch allocator保留的总缓存空间 | 可大于allocated，释放张量后也可能保留 |

`nvidia-smi`又是另一层观察，不等于上述任意一个数字。
`empty_cache()`不能替你把仍被引用的KV张量变成空闲请求页；
逐token调用也不应被当成例行优化方案。
先看所有权和引用，再看allocator，再看系统进程统计。
Spark为共享内存架构，系统内存压力也要观察，不能把所有现象按独立显存卡解释。

### 释放、复用和GPU异步要分清

CPU实验每步同步执行，所以能马上观察目录变化。
真实CUDA多stream路径还要保证：

```text
没有正在读取旧页的kernel
    + 没有有效请求仍持有该页
    + 必要的event/stream依赖完成
    ↓
该页才可交给新请求写入
```

共享前缀仍有引用时，某个请求结束不等于那页可立刻覆盖。
本教学池只有单线程独占所有权，没有refcount/COW、异步租约或用户取消API。
Engine的finally会回收仍登记的请求页，但不是跨stream、跨进程、任意故障恢复保证。

## 7. 怎么调试：先看抽屉账本，再看耗时

第一遍只用CPU小输入，不加载大checkpoint：

1. 在Notebook运行storage实验，看每个event的owners、free、table、lengths。
2. 在 `PagedPool.reserve` 观察 `needed`、`free`，确认不足时未改变旧状态。
3. 在 `PagedRequest.update` 观察 `position → physical_block,offset`。
4. 在 `PagedRequest.get` 核对按逻辑顺序拼回的数据，与同一组连续参考K/V对齐。
5. 在 `PagedPool.release` 确认请求消失、旧handle失效、页回到free；storage大小不变。
6. 在 `Engine.run` 的buckets循环观察每次调用的position、input length、batch size。
7. 注入一次模型异常，检查finally回收；不要把“没报错”当成“没有泄漏”。

建议在Jupyter变量面板盯住这组字段：

```text
pool.free
pool.requests
state.table
state.capacity
state.lengths
pool.storage.shape
```

先学这6项，再学完整框架的指标命名。具体断点操作可复用
[Day 3 Jupyter调试](DAY03_SHAPES_DEBUGGING.md)，只更换函数位置，不必重新创建容器。

## 8. 怎么优化：每轮只改一个原因

顺序是：**先不读错、不错释放 → 再定位瓶颈 → 再改一个参数 → 再回归验证**。

| 症状 | 先检查什么 | 一次尝试什么 | 必须防止的副作用 |
| --- | --- | --- | --- |
| 请求结束后不能再接单 | free页数、live所有权、异常路径 | 修复漏release、重复引用 | 不能提前复用还在读的页 |
| free很多但新请求仍等 | 队首能否装下、slots、分桶、预算 | 准入或预算单变量实验 | 公平、饥饿、拒绝率 |
| 刚启动就OOM | 权重、池预算、临时buffer、上下文 | 减并发或降低已确认的容量上限 | 不把CPU fallback当GPU修复 |
| KV占主导 | 每token字节数、长度分布、页尾 | block_size、容量预留策略、已支持KV精度 | 不是分页就能压缩向量；低精度需质量对齐 |
| 首token慢 | 排队、tokenization、prefill分别计时 | prefill分块/预算；有重复时再看prefix | 不能牺牲decode尾延迟而不报告 |
| 后续token慢 | 活跃上下文、KV读带宽、batch形状 | batch上限、预算和attention后端 | 吞吐升但P99延迟变差 |
| kernel快、端到端却慢 | CPU调度、gather/copy、同步、网络 | 优先处理profiling证实的主项 | 不拿模拟round算速度 |

本仓库可先做两个**不需要GPU计时**的单变量实验：

```text
长度为[3,5,7]，比较block_size为2与4时的页尾空槽；
比较slots和token_budget，分别观察准入与首次执行，而非只看总轮数。
```

一个例子：长度[3,5,7]，block_size=2需要2/3/4块，18个槽，3个空槽；
block_size=4需要1/2/2块，20个槽，5个空槽。
小块降低这组工作负载的页尾浪费，但表项更多，实际kernel与元数据成本也会变化。
不能据此宣布block_size=2普遍最优。

以后真实服务的性能报告至少写：

```text
固定：模型/权重、dtype、prompt与输出长度分布、到达率、硬件/框架版本
观测：排队时间、TTFT、token间隔、端到端耗时、P50/P95/P99、吞吐、错误率
内存：池总块数/free/live/前缀缓存页、峰值allocated/reserved、系统内存
方法：预热、并发、请求数、同步口径、实际attention后端
变更：只改一个变量；与同一基线比较；保留失败日志
```

本课不进行真实服务压测；TTFT与后续token间隔的统计起止点必须在报告里定义。

## 9. 易混问题，直接回答

**抽屉里保存文字吗？** 不，是各层数值K/V；文字token另有输入/输出列表。

**一个请求结束，模型也卸载吗？** 一般不需要；模型权重服务期间共用，回收的是请求状态。

**释放抽屉后显存没降，是泄漏吗？** 不一定。先看free页是否回来了，pool张量可能一直保留。

**C借到B曾用的抽屉，会看见B的历史吗？** 正确接口只读有效长度、写后再读；
本实验也验证新请求为空。原始物理数据不自动清零，不能当成安全擦除。

**目录能代替KV数据吗？** 不能，目录是地址表，不是压缩表示。

**分配失败有没有破坏已运行的A/C？** 实验断言free、requests与实际storage都保持原样。

**为何scheduler模拟不够？** 它没调用模型、没写KV、没计GPU时间；
新Notebook另用真实随机小模型验证一次分桶调用与回收，但仍不是生产服务。

## 一手参考与边界

建议按以下顺序深入，每篇先带着一个问题读，不先钻完整kernel：

| 顺序 | 一手资料 | 这次只回答什么 |
| --- | --- | --- |
| 1 | [Hugging Face：Caching图解，固定v4.57.0](https://huggingface.co/docs/transformers/v4.57.0/en/cache_explanation) | 为什么历史K/V能复用，为什么每层各有Cache？ |
| 2 | [vLLM：Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/) | Scheduler、Engine Core、Worker、Model Runner各在哪？ |
| 3 | [vLLM：Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/) | 逻辑块如何定位物理KV，kernel读的究竟是什么？ |
| 4 | [Triton：Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | Q/K/V块怎样计算并维护m/l/o？ |
| 5 | [vLLM：Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) | 怎么用真实指标定位容量、预算与抢占问题？ |

先做本课CPU实验，再看第3–5项的代码和性能内容。
vLLM的`latest`资料会演进；其V1可用统一token预算循环处理不同阶段，
不要把本课便于理解的prefill/decode分段图误读成所有版本的实际类结构。

- [本仓库物理页池](../learning/mini_vllm/paged_kv_cache.py)
- [本仓库真实小模型Engine与BatchCache](../learning/mini_vllm/engine.py)
- [PyTorch 2.10 CUDA内存管理、allocated/reserved与stream语义](https://docs.pytorch.org/docs/2.10/notes/cuda.html)
- [vLLM Paged Attention设计](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [vLLM调优说明：准入、抢占与分块prefill的权衡](https://docs.vllm.ai/en/latest/configuration/optimization/)

本页数值按当前仓库教学实现生成。外部框架的具体布局、参数和策略会演进；
未部署真实vLLM、未验证生产异步生命周期、未做质量或GPU加速比承诺。
