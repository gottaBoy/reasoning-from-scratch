# 第四周关键知识：Day 22–30

返回 [索引](KNOWLEDGE_INDEX.md)。代码在
[mini_vllm](../learning/mini_vllm/README.md)，不是外部 vLLM 项目本体。

## Day 22：真实框架集成与服务边界

**核心概念**：模型文件格式、服务协议、依赖隔离、GPU 架构兼容、请求参数。
本仓库的 `.pth` 自定义 checkpoint 不可直接当作 HF 模型目录传给任何 serving API。

**直觉**：API 服务是外壳，模型加载/执行/调度是内层；能 HTTP 200 不代表优化正确。
Spark 是 ARM64/GB10，wheel、扩展、驱动/CUDA 必须适配；不能覆盖已有教学镜像来试运气。

**实验**：独立启动授权服务后设置 `VLLM_BASE_URL=http://127.0.0.1:8000/v1`；
执行 `python -m learning.vllm_probe`。默认没有服务时显式报告未配置，不假报通过。

**面试主问**：部署失败按什么顺序定位？
**答案要点**：先版本/硬件和容器 GPU，自检 kernel，再权重/格式/tokenizer，
随后最小单请求，最后并发与压测。保存日志、版本、最小输入。
**追问**：为什么要隔离？答：serving 框架可能锁定另一套 torch/CUDA/triton，
避免破坏已验证环境并方便回滚。

**误区/验收**：探针测 non-streaming 总耗时，不是 TTFT；真实服务验证仍是外部依赖项。

## Day 23：Paged KV 的地址翻译

**核心概念**：逻辑位置、block table、物理页、page offset、free list、碎片。
给定 page size P，`page=position//P`，`offset=position%P`，
再用 `table[page]` 找物理 block，不能把逻辑页号直接作物理地址。

**例子**：P=4，table=[7,2,9]，逻辑位置 6 对应物理 block 2、offset 2。
不是按物理编号排序读；读取顺序必须遵守 block table。

**实验**：构造释放后重用的非连续表；验证逻辑顺序仍正确。
入口：`day23_course.ipynb`。

**面试主问**：分页为什么能降低外部碎片影响？
**答案要点**：请求不再需要一整块连续大内存，独立块可复用；仍有最后一页的
内部浪费和元数据/访问成本，不是完全消除浪费。
**追问**：page size 如何选？答：小页减少内部浪费却增元数据/访问复杂度；
大页更规整却浪费更多，需要 workload/硬件测试。

**误区/验收**：block manager 不等于 paged GPU Attention kernel，二者要协作。

## Day 24：调度、公平与 token budget

**核心概念**：arrival、waiting、running、finish、准入、迭代边界、预算、公平性。
token budget 控制每轮总工作，而不是简单等于“最多几个用户”。

**直觉**：短请求释放位后可加入新请求，但大 prefill 可能挤占 decode；
chunked prefill、优先级等策略需单独设计与验证。

**实验**：同一到达/长度序列下比较三种策略，记录首执行轮、完成轮、每轮预算。
入口：`day24_course.ipynb`。模拟只含 decode 轮，无真实 prefill 成本。

**面试主问**：怎样防止短请求长期被长请求阻塞？
**答案要点**：先定义 SLO、公平准则、准入和预算，再考虑分块/优先级；
吞吐和延迟存在权衡，不提供无条件最佳算法。
**追问**：最高吞吐是否最优？答：未必，p95/p99 TTFT/TPOT、取消率和公平都重要。

**误区/验收**：round 不是 ms；不能在一个轮次里反复对同一请求 decode 却当作公平组批。

## Day 25：所有权、回收、容量与错误处理

**核心概念**：atomic allocation、OOM、释放后失效、引用、容量上界。
当前示例按请求最大已知预算提前预留页，简化 admission；不支持抢占/offload。

**直觉**：一张旧房卡失效后，不应访问已分给其他人的房间。
否则是错误结果乃至跨请求泄漏，不只是性能下降。

**实验**：OOM 前后 free list 不变，释放后旧 handle 报错；分配失败不留半个请求。
入口：`day25_course.ipynb`。

**面试主问**：取消请求后必须释放哪些状态？
**答案要点**：KV pages、scheduler 状态、streaming/输出缓冲、引用计数和 pending 工作；
GPU 异步执行中的资源生命周期也必须考虑。
**追问**：教学引擎支持取消吗？答：当前不支持外部取消 API，只有完成与异常清理，
不能把概念说明冒充已实现功能。

**误区/验收**：多进程/多线程安全、生产引用计数不由 Python dict 自动提供。

## Day 26：真实模型合批与 Cache 拆分

**核心概念**：分桶、batch 轴、不同 Cache position、padding、request identity。
当前只把位置相同且本轮输入长度相同的请求 stack 成一次模型 forward。

**公式/直觉**：两条 `[1,KV,T,D]` Cache 沿 batch 合为 `[2,KV,T,D]`，
每层返回后按 batch 拆回请求。不是把两条用户历史沿序列拼成一条。

**实验**：两条同长度不同 prompt 合批，输出逐条与独立 greedy 比较。
入口：`day26_course.ipynb`。

**面试主问**：异长请求为什么不能直接 stack？
**答案要点**：张量 shape、位置和 Mask 不同；需要 padding/ragged/paged kernel
或分桶。填充也要屏蔽并维护每请求真实长度。
**追问**：为什么分页仍做 gather？答：这是可读参考实现，
真实 paged kernel 可直接按表读，不保证参考路径有生产性能。

**误区/验收**：真正 batch_size=2 的 forward 与“循环调用两次单请求”不同，
报告要给出实际 batch trace。

## Day 27：端到端指标与对照实验

**核心概念**：TTFT、TPOT/ITL、request/s、output tokens/s、p50/p95/p99、并发与输入分布。
TTFT 通常含排队、prefill、首输出等；non-streaming 总时长不能直接视作 TTFT。

**实验**：同权重、同 prompt 列表、相同输出预算，用同步 wall-clock
对比 sequential 与真实 engine。另保存模拟轮次，两者不混算。
入口：`day27_course.ipynb`。

**面试主问**：怎样设计 Continuous batching 的 benchmark？
**答案要点**：固定输入/输出长度与到达分布，warm-up，定义延迟/吞吐口径；
测试不同并发，记录错误、OOM、取消和尾延迟，分清模型时间与 HTTP 时间。
**追问**：为什么 toy engine 可能更慢？答：小模型 launch-bound、Python 开销、
gather/复制、批次不够大，性能优势不是由功能名自动带来。

**误区/验收**：不从一个 3 请求小样本推出正式容量或生产加速比。

## Day 28：Prefix Cache 与共享语义

**核心概念**：精确前缀匹配、内容键、模型身份、缓存命中、失效、clone/COW。
同一模型/配置/位置前提下，同 token 前缀可复用 K/V；
仅“文字看起来类似”或 tokenizer 不同不能直接复用。

**直觉**：公共前缀像保存过的草稿，复用后每个分支继续写自己的部分。
本课程为 correctness 克隆 snapshot，不做 COW 或共享物理页的内存节省承诺。

**实验**：首次 miss、再次 hit，追加 suffix 后与 full forward 对齐；
另取 snapshot 的原前缀长度不能被污染。权重变更后 clear。
入口：`day28_course.ipynb`。

**面试主问**：前缀缓存最容易遗漏的 invalidation 条件？
**答案要点**：权重/adapter、tokenizer、位置/rope 配置、精度/实现身份、
多租户隔离和可变状态；哈希碰撞/引用计数也要考虑。
**追问**：为什么只存 logits 不够？答：能选首 token，但继续 Attention 仍需历史 KV。

**误区/验收**：命中率高不必然总体更快，要扣掉 hash、查表、复制和容量管理成本。

## Day 29：技术报告、可复现与归因

**核心概念**：基线、假设、控制变量、版本、原始结果、失败记录、限制。
分清“数据正确”“有性能提升”“模型质量提高”是三类证据。

**实验**：按 Mini 项目 README 模板填写，标明 seed、版本/commit、device、
dtype、shape、测试阈值、warm-up、重复次数及每个跳过项。
入口：`day29_course.ipynb`。

**面试主问**：讲一个真实优化项目，怎么避免空话？
**答案要点**：业务/性能目标 → 基线 → profiling 证据 → 单变量修改 →
正确性 → 对照数据 → 回归/限制。讲失败实验及其原因比只报百分比更可信。
**追问**：不允许提交权重时如何复现？答：给模型来源/版本/hash、下载/转换步骤、
配置和小输入，输出原始报告；不要泄漏访问令牌。

**误区/验收**：不能用“AI 生成了代码”作为算法正确或优化有效的证明。

## Day 30：综合答辩与下一阶段

**核心概念**：系统边界、质量/时延/吞吐/成本的权衡、资源生命周期。
一次请求依次经过编码/准入/prefill/KV 分配/多轮 decode/输出/释放；
相邻模块的契约比单个术语更重要。

**实验**：综合运行模型与 paged engine；对照单请求 greedy，验证释放、
前缀隔离和输出一致性。外部 vLLM、量化和生产 kernel 仍单独列状态。
入口：`day30_course.ipynb`。

**面试主问**：现在的项目与生产 vLLM 差在哪里？
**答案要点**：模型规模/加载、真实分页 kernel、异长组批、抢占/取消、
分布式与量化、协议、监控、安全和压测覆盖；明确只完成教学的哪些部分。
**追问**：下一步最值得做什么？答：根据已测瓶颈选择一个方向，用可复现数据推进，
不是一次性再加入所有热门技术。

**误区/验收**：材料全备齐不等于个人精通。用户仍需逐日口述、改错、独立实现，
并在新输入/新硬件下验证原有结论。

## 扩展方向（非本月全部实现）

| 方向 | 至少需要理解 | 面试时必须说明的边界 |
| --- | --- | --- |
| 量化 | weight-only/activation、scale、校准、误差、支持 kernel | 权重小不必然快，反量化/带宽/质量需测试 |
| Speculative decoding | draft、target 验证、接受率、分布保持条件 | 不等同于贪心同时多猜几个词 |
| CUDA Graph | capture/replay、固定地址、shape bucket、warm-up | 动态分配/shape 不满足条件时不能直接 capture |
| Tensor Parallel | 线性层切分、collective、通信 | 本机单 GPU 未验证多卡吞吐 |
| Pipeline Parallel | 分层、microbatch、bubble | 不只是将 layers 分到不同 GPU |
| 多租户服务 | 配额、公平、取消、隔离、鉴权、审计 | 开发 token 和 localhost 配置不是公网安全方案 |
