# 第一周关键知识：Day 1–7

回到 [每日知识索引](KNOWLEDGE_INDEX.md)。图示、公式中的具体数字均标明配置前提。

## Day 1：环境与一次生成

**核心概念**：checkpoint 是参数，模型代码是计算过程，tokenizer 负责文本/ID 转换；
inference 包括编码、forward、选择 token、解码，不只是 `model(...)`。

**直觉**：模型交评分表，生成策略选编号，tokenizer 将编号还原为文本片段。
`eval()` 切换模块行为，不等于关闭梯度；推理还用 `no_grad` / `inference_mode`。

**实验**：检查 `torch.__version__`、`torch.version.cuda`、设备，运行真实矩阵乘法；
随后固定 prompt 和输出预算，确认模型与输入设备一致。入口：[Spark 教程](../DGX_SPARK.md)。

**面试主问**：Docker 能看见 GPU，但 torch 不能用，怎样查？
**答案要点**：先分层：GPU 注入/CDI、CPU wheel、CUDA wheel/驱动、计算架构，
用张量分配和实际 kernel 检查。驱动的 CUDA Version 不是安装的 Toolkit 或 wheel 版本。
**追问**：为什么不直接升级所有依赖？答：会破坏已验证组合，先保留版本、日志和最小复现。

**误区/验收**：base 模型网页续写不等于部署失败；能解释一次生成的每个阶段，
但不能把 16 token 的结果称为正式质量评测。

## Day 2：Transformer 的主路径与模块职责

**核心概念**：Embedding、RMSNorm、Attention、SwiGLU、Residual、LM Head。
主路径 `[B,T,E]` 经过所有 Block 时 T 不变，最终投影为 `[B,T,Vocab]`。

**公式/直觉**：
`x1=x+Attn(norm1(x))`；`x2=x1+FFN(norm2(x1))`。
保留当前状态，查询上下文，合并更新，再逐位置加工特征。28 层不是生成 28 token。
`FFN=Wd[SiLU(Wg x)⊙Wu x]`，两条分支并行；SiLU 不是 `[0,1]` 的概率。

**实验**：hooks 捕获两次相加前后，与手动相加对齐。
入口：[模块图解](DAY01_DAY02_NOTES.md)、`learning/day03_shapes.py`。

**面试主问**：Attention 与 FFN 区别？
**答案要点**：Attention 跨允许的位置混合信息；FFN 独立处理各位置的特征，
但输入已含上下文，不代表其输出没有上下文。
**追问**：down projection 为什么回到 E？答：匹配残差接口；不是直接裁掉多余通道。

**误区/验收**：不能把每一维命名为固定“语法/颜色”。能逐模块画图并解释 logits
与选择 token 的区别，详见 [SwiGLU](ROPE_SWIGLU_INTUITION.md)。

## Day 3：MHA、MQA、GQA、RoPE 与因果性

**核心概念**：`E` 残差宽度、`Hq/Hkv` 头数、`D` head dimension；
GQA 要求 Query heads 可按实现分组，教学实现要求 `Hq % Hkv=0`。

**公式/形状**：Q `[B,Hq,T,D]`，紧凑 K/V `[B,Hkv,T,D]`，
score `[B,Hq,T,T]`。Qwen3 0.6B 显式 `D=128`，因此 Q 投影宽 `16×128=2048`，
不是强行将 E=1024 平分为 16×64。
RoPE 每对特征做 `R(position×frequency)` 旋转，dot 带上相对位置差；
多频率减少短范围混淆，但不保证无限唯一或远距分数单调递减。

**实验**：预写每步 shape，用断点检查 view/transpose 与线性投影的区别；
修改后半段输入，前缀 logits 应不变。入口：[调试教程](DAY03_SHAPES_DEBUGGING.md)。

**面试主问**：GQA 相比 MHA 省什么？
**答案要点**：减少 KV 投影与 Cache heads，Q heads 不变；仍需 Query 对上下文的计算。
**追问**：`repeat_interleave` 真省显存吗？答：缓存保持紧凑，但教学计算临时扩展；
生产 kernel 可直接读共享 KV，不必物化副本。

**误区/验收**：RoPE 不负责遮挡未来。Mask bool 语义因 API 而异，需核对；
不能说同样距离一定同样分数，因为 q/k 内容也在变化。

## Day 4：Prefill、Decode、KV Cache 与误差

**核心概念**：Prefill 处理 prompt，常见 decode 每请求每轮 1 token；
缓存属于请求和层。因果性使历史表示不依赖未来，历史 K/V 可复用，Q 通常不缓存。

**公式/内存**：
GQA KV 字节数约 `2×L×B×T×Hkv×D×bytes`，不含 allocator/页碎片。
输入 7-token prefill 时 Mask 为 `[1,1,7,7]`；再输入 1 token 时为
`[1,1,1,8]`，Cache 从 T=7 增至 8。最后位置 logits 预测的是下一个 token。

**实验**：同权重、固定输入、FP32 full/cached 对齐，再观察 BF16。
真实 Qwen3 本机 BF16 最大误差 0.25，top-1 一致；FP32 约 `1.35e-5`。
入口：[Day 4](DAY04_KV_CACHE.md)。数据依赖本次配置，不能直接推广。

**面试主问**：Cache 是否把 decode Attention 变成 O(1)？
**答案要点**：不是。避免重投影历史 K/V，但当前 Q 仍读全部可见 K/V；
上下文越长读带宽和 attention 工作越多。
**追问**：清 Cache 只清张量可以吗？答：还要检查 RoPE 位置状态和请求身份。

**误区/验收**：逐位相等、容差通过、top-1 相同是三种不同结论。
先证明 FP32 正确，再检查低精度误差，不随意放宽阈值“造通过”。

## Day 5：版本化模型比较、MoE、MLA

**核心概念**：比较具体模型版本；Dense FFN 逐 token 使用同一 FFN，
MoE 的 router 为 token 选择部分 routed experts，可能另有 shared experts。

**公式/直觉**：
`y=Σi∈TopK wi Expert_i(x)`；总参数大不代表每个 token 用完全部参数。
FFN 的连续 gate 是通道调节，MoE router 是专家选择，不能混称。
MLA 改变 KV 表示；教学理解为 latent KV 与位置分支，不是简单减少 KV heads。
实际缓存与投影吸收必须查看目标实现，不能把普通 GQA 公式直接套用。

**实验**：`32 tokens × top_k=2 = 64` 次分配，不是 64 个新 token。
改变 router 偏置观察 experts 负载；随机 router 不代表真实语义专家。
入口：[Day 5](DAY05_ARCHITECTURES.md)、`learning/day05_moe.py`。
配置容易混淆时，先做 [投影与内存实验](../learning/day05_config_memory.ipynb)，
验证 T 翻倍后权重不变、KV两倍、单层prefill score四倍。
MLA 不直观时，先看 [两数生成多头 K/V 的解释](MLA_INTUITION.md)，
再运行 [展开与 latent 两条路径的对齐实验](../learning/day05_mla_intuition.ipynb)。
每层每个 token 的 latent 分别保存，不是把整段历史合成一个摘要。

**面试主问**：MoE 能否同时省参数内存和计算？
**答案要点**：主要减少每 token 激活计算，相对同总参数 dense；权重仍需存储或分布，
增加路由/通信、负载均衡成本，不能笼统说省全部显存。
**追问**：为何热点 expert 影响吞吐？答：并行系统可能等待最重 shard，通信和容量也限制效率。

**误区/验收**：不要说 MoE 与 MLA 二选一；它们分别改变 FFN 与 Attention/KV 路径，
可以组合。本日目标是画图和推导，不是已部署 DeepSeek。

## Day 6：Online Softmax 与 FlashAttention

**核心概念**：IO-aware tiling、稳定 softmax、每行 running maximum/denominator/numerator。
FP16、B=1、H=16、T=4096 时，单张 score 矩阵约 `16×4096²×2=512 MiB`。

**公式**：新 score 块 s，`m'=max(m,max(s))`，`α=exp(m-m')`，
`l'=αl+Σexp(s-m')`，`o'=αo+exp(s-m')V`，最终 `o/l`。
旧量要换到新最大值的基准，不能直接相加不同归一化尺度下的 softmax。

**实验**：改变块宽 1/3/8，与普通 softmax/SDPA 对齐；大幅值输入必须仍有限。
入口：`learning/day06_course.ipynb`、`learning/online_attention.py`。
第一次学习时先看 [两块小数字的手算解释](DAY06_ONLINE_SOFTMAX.md) 和
[逐步实验](../learning/day06_online_softmax.ipynb)，再读压缩版递推公式。

**面试主问**：FlashAttention 是否把 dense prefill 复杂度从 T² 降为 T？
**答案要点**：不改变 exact dense attention 的主要算术复杂度，主要优化 IO/中间存储，
利用片上存储分块融合。某些 sparse attention 是另一个问题。
**追问**：为什么用最大值？答：指数数值稳定，分块更新时需重缩放旧累积量。

**误区/验收**：PyTorch 分块参考可验证数学，不自动获得融合 GPU 性能。
能推导两块 score 合并、说明稳定性和临时内存边界。

## Day 7：推理系统机制地图

**核心概念**：请求生命周期、准入、调度轮、token budget、等待队列、释放与取消。
Continuous batching 在迭代边界调整请求集合；Paged KV 管理物理块，负责不同层面。

**直觉**：不同任务不是必须一起开始一起结束；空位可接新任务，
但是否提速由输入长度、硬件、调度开销和 SLO 决定。

**实验**：用短/长/晚到请求画 sequential/static/continuous 的轮次。
入口：`learning/day07_course.ipynb`、`learning/mini_vllm/scheduler.py`。

**面试主问**：Continuous batching 的代价是什么？
**答案要点**：请求/Cache 生命周期更复杂，prefill/decode 争资源，存在公平、饥饿、
内存超额和长 prompt 阻塞风险；吞吐最大不必然满足延迟目标。
**追问**：PagedAttention 是否一定启用 prefix sharing？答：分页与共享分开，
共享还需内容键、引用计数/COW、释放语义。

**误区/验收**：模拟 round 不可直接转换为 ms/token 或 GPU tokens/s。
能把 GQA、RoPE、KV Cache、FlashAttention、PagedAttention、调度放在不同框内。
