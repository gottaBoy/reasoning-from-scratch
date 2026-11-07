# Day 9：单头因果Attention，从匹配表到内容汇集

前置：[Day 8](DAY08_RMSNORM_ROPE.md)。符号统一见 [NOTATION](NOTATION.md)。
本日从一张匹配表开始，不急着拆成多头；当前个人进度不因准备本页而改变。

## 1. 图解与数字例子

把每个Query当成“查资料的问题”，每个Key当成“资料的索引”，Value是资料内容。

```text
位置0：只能看资料0
位置1：可以看资料0、1
位置2：可以看资料0、1、2

Query → 与Key点积 → 除sqrt(D_k) → 屏蔽未来 → softmax → 汇集Value
```

假设某一行缩放后的分数为 `[0, ln(3)]`，对应Value为 `[10,20]`：
权重是 `[1/4,3/4]`，输出为17.5。
如果第二个位置属于未来，应把该分数设为负无穷，权重变成 `[1,0]`，输出10。
**把未来分数设成0并不等于屏蔽**：exp(0)=1，仍有权重。

## 2. 概念与计算契约

```text
Q [B,1,T_q,D_k]，K [B,1,T_kv,D_k]，V [B,1,T_kv,D_v]
S = Q @ K.transpose(-2,-1) / sqrt(D_k) + M
P = softmax(S,dim=-1)
O = P @ V → [B,1,T_q,D_v]
```

缩放按每头D_k，不是E、T或词表大小。Mask在softmax前生效。
本仓库Mini内部bool mask的True表示禁止；PyTorch SDPA的bool mask中True表示允许。
比较时必须显式翻译，不是把同一张bool表不加判断地传过去。
至少一个有效Key的行才有本例的正常概率含义；全Mask行另行定义策略。

## 3. Jupyter与源码路线

打开 [Day 9 Notebook](../learning/day09_course.ipynb)，先读新增图解，再运行setup与参考实验。
查看 `learning/mini_transformer/model.py` 中 `Attention.forward`：
投影、RoPE、key/query positions、Mask、softmax、输出投影。
`run_day(9)`固定单头小模型，修改后半段输入，断言前半段logits不变；
另外调用PyTorch分块Attention对照SDPA。它没有加载真实语言模型权重。

先自己重算一行17.5，再遮住参考实现写单头forward。加入T=1、非方阵及未来token扰动。
注意Cached Decode的Query可能有绝对位置偏移，不应直接套左上角三角Mask。

## 4. 调试与优化顺序

1. 打断点查看Q/K/V的shape，分清位置轴与特征轴。
2. 打印最后一行权重，检查Key轴行和及禁止位置为0。
3. 改未来token，确认前缀不变；去Mask应能构造失败例。
4. 与独立SDPA输出按容差对齐，不只比较argmax。
5. 然后才看score物化、dtype与后端，不能先用更快的错误Mask。

当前实验只证明支持输入上的数值和因果性，不证明训练收敛或GPU性能。

## 5. 自测与参考答案

**问：为什么softmax沿最后的Key轴？**
答：一个Query在它可访问的资料间分配权重，不是在不同Query或不同用户之间归一化。

**问：为什么未来分数不能简单填0？**
答：exp(0)不为0；加法Mask常用负无穷来使概率贡献为0。

**问：分数表为什么是T_q×T_kv，而不是D×D？**
答：每个位置对的特征做点积求和，留下Query位置和Key位置两条轴。

**问：只比较生成文字是否相同够吗？**
答：不够，采样和argmax可掩盖logits差异；应比较全部支持输出和数值误差。

## 6. 验收与资料

- [ ] 独立画出一张三角可见性表，并手算17.5与屏蔽后的10。
- [ ] 解释Mask的API语义及offset，保留一个错误Mask负例。
- [ ] 独立实现与参考对齐，记录shape/dtype/容差，不预填通过。

深入：[PyTorch 2.10 SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)；
[Day 5逐步手算](ATTENTION_HAND_CALC.md)。下一课：[Day 10](DAY10_MULTIHEAD.md)。
