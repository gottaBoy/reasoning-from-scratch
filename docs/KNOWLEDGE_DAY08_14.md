# 第二周关键知识：Day 8–14

返回 [索引](KNOWLEDGE_INDEX.md)。参考实现：[Mini Transformer](../learning/mini_transformer/model.py)。
模型为独立的 FP32 优先教学实现，不加载 Qwen3 预训练权重，不声称文本质量。

## Day 8：RMSNorm 和 RoPE 的实现契约

**核心概念**：最后一维归一化、FP32 reduction、epsilon、缩放参数、split-halves、offset。
`RMSNorm(x)=γ⊙x/sqrt(mean(x²)+eps)`；LayerNorm 另需减均值，二者不等价。

**直觉/形状**：缩放一行信号强度，不读其他 token；RoPE 在每个 head 的维度对里转角，
并不更改元素个数。不同代码的旋转配对不同，权重/布局必须配套。

**实验**：固定随机输入，对齐仓库 RMSNorm/RoPE；offset=0/5、
非整齐 T、偶数 D，并检查不同 token 各自的位置。
入口：`day08_course.ipynb`。

**面试主问**：FP16 做方差/均方值有什么风险？
**答案要点**：动态范围与累积精度受限，平方/求和可能溢出或误差放大，
常用 FP32 中间累积再转回原 dtype，epsilon 也影响小数输入。
**追问**：RoPE 为什么只改 Q/K？答：本实现用它调节相似度；V 保存待汇集内容，
具体模型也可能有不同位置机制，不能无限泛化。

**误区/验收**：不能只测随机正态分布；加全零、常数、幅值较大/小的输入，
并给出 shape、dtype、误差阈值。

## Day 9：单头 Attention、Mask 与缩放

**核心概念**：`softmax(QK^T/√D + mask)V`，Q/K/V 的角色与学习投影。
Q/K 决定匹配分数，V 是被汇集的表示，不是原始文字值。

**公式**：`Q[B,T,D] @ K^T[B,D,S] → scores[B,T,S]`，
softmax 沿 S 归一化。按独立同分布直觉，点积方差随 D 增大，√D 缩放控制量级。

**实验**：修改未来 token、比较前缀；对照 SDPA，检查对角线是否可见。
入口：`day09_course.ipynb`。

**面试主问**：Mask 为何常用 -inf 而不是零？
**答案要点**：softmax 中 `exp(-inf)=0`；零分数仍有非零权重。
**追问**：一整行都 Mask 会怎样？答：普通 softmax 出 NaN，需要明确有效 query、
padding 和 kernel 的契约，不得默认所有 API 自动处理。

**误区/验收**：不能比较 argmax 来证明 logits 正确。对比完整数值、边界 T=1、
mask 方向及不允许访问的权重为零。

## Day 10：多头 Attention 与布局

**核心概念**：投影宽 H×D、拆头、transpose、stride、contiguous、输出投影。
对每个 head 单独匹配，合并后的宽度 H×D 再映射回 E。

**直觉**：多张不同投影空间中的位置关系表，不是在序列轴上拼成一张大表。
改变 stride 不等于复制；reshape 需要时可能分配新内存。

**实验**：故意用 E≠H×D，打印每步 shape/stride；对照 SDPA。
入口：`day10_course.ipynb`。

**面试主问**：transpose 后为什么 `.view()` 可能报错？
**答案要点**：新 stride 不满足目标 view 的连续性要求；用适当 reshape 或
contiguous 后 view，明确潜在复制成本。
**追问**：head 越多一定越强吗？答：还取决于 head_dim、总宽度、训练和成本，
不能脱离配置只比较 head 数。

**误区/验收**：Qwen3 0.6B 不符合“E 永远等于 H×D”的简化假设，
必须让 projection 和 residual 接口一致。

## Day 11：参数化 GQA/MQA

**核心概念**：`Hkv=1` 为 MQA，`1<Hkv<Hq` 为 GQA，`Hkv=Hq` 为 MHA；
按本实现分组，Query head `h` 读 KV head `h // (Hq/Hkv)`。

**公式**：K/V projection 参数各约 `E×Hkv×D`，Query 参数约 `E×Hq×D`；
KV Cache bytes 同 Hkv 成正比。公式忽略 bias，模型其他参数不因此减少。

**实验**：KV=1/2/4 的同配置路径分别做完整/Cache 自身对齐，比较字节数，
不比较不同 KV 结构的随机模型输出。
入口：`day11_course.ipynb`。

**面试主问**：GQA 能直接把 MHA 权重改个配置就加载吗？
**答案要点**：K/V 权重 shape 已改变，需要有定义的转换/训练策略；
不能保证任意切 head 数不改变语义或质量。
**追问**：为什么 decode 可能受益更多？答：历史 KV 读取与带宽常重要，
较少 KV heads 减少数据量，实际收益要看 kernel 和工作负载。

**误区/验收**：减的是紧凑状态，不代表临时 repeat 后的张量也小；给出保存位置。

## Day 12：请求级 Cache 与边界处理

**核心概念**：每请求独立状态、层一致长度、位置偏移、批维、context limit。
本实现 position 从 Cache 推导，避免模型上单一可变位置影响并发请求。

**直觉**：不同请求各有进度条，模型权重共用但笔记不能串用。
合法请求内的每层 K/V 长度应在本轮结束后相同。

**实验**：prefill 4 后连续 decode 5 步；重复独立请求；超长拒绝后原 Cache 仍可用。
入口：`day12_course.ipynb`。

**面试主问**：Cache reset 不全会出现什么？
**答案要点**：位置错位、历史串请求、数值错误/隐私泄漏；需要清状态与严格所有权。
**追问**：多 token decode chunk 能直接用方形 Mask 吗？答：query 长 q，key 长 p+q，
必须按绝对位置构造矩形 causal Mask。

**误区/验收**：只测 7+1 不足以覆盖多步增长。增加多 batch、不同 split、重复重置、
位置错误负例和容量边界。
70B 估算练习见 [KV Cache 70B 公式](KV_CACHE_70B.md)；不要把参数量直接代入 Cache 公式。

## Day 13：SwiGLU、残差与训练梯度

**核心概念**：门控、非线性、残差、参数共享范围、梯度追踪。
`SiLU(z)=z/(1+exp(-z))`，可为负和大于 1；不是 MoE 专家概率。

**公式**：`FFN(x)=Wd(SiLU(Wg x)⊙Wu x)`；参数约 `3E×F`（无 bias）。
Block 的两次相加分别保留 x 和更新后的 x1，不是都加初始 x。

**实验**：小模型 cross-entropy backward，检查全部参数 grad 存在、有限。
入口：`day13_course.ipynb`。

**面试主问**：`eval()`、`no_grad()`、`inference_mode()` 有什么区别？
**答案要点**：eval 控制模块行为；另外两者影响 autograd，inference_mode 更严格，
其中创建的 tensor 不应直接用于需要保存中间值的反向。
**追问**：没有 dropout 时 eval 还有必要吗？答：保持推理契约，但不能代替关闭梯度。

**误区/验收**：参考实现 Cache 用于推理，不设计为长序列训练的梯度状态。

## Day 14：正确性与性能基线

**核心概念**：参考模型、固定输入、容差、warm-up、同步、重复采样、峰值内存。
数学等价不等于浮点逐位相同，top-1 一致也不等于 logits 对齐。

**计时**：先 warm-up，再多轮测量；CUDA 记录前后同步，区分 wall-clock、
CUDA events 和内核 profiling。`memory_allocated` 与 reserved/系统内存不同。

**实验**：给出 prefill 和 cached generation 独立基线，记录 median/min、
输入 shape、随机 seed、dtype、设备。入口：`day14_course.ipynb`、`learning/bench.py`。

**面试主问**：为什么优化后“快 10 倍”的结论可能不可信？
**答案要点**：未同步、含编译差异、工作量不同、单次测量、CPU/GPU 混比、
上下文/Batch/输出长度不一致，或只测空操作。
**追问**：怎样证明瓶颈？答：用 profiler/带宽和算力估计定位，再单变量实验验证，
不能仅凭代码感觉。

**误区/验收**：第二周完成意味着自己可解释实现和测试，不是获得预训练语言能力；
Mini 模型随机 token 输出不用于评价问答质量。
