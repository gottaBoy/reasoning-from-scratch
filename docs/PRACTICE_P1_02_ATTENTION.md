# P1-02：把匹配、Mask、softmax、汇集写成你自己的Attention

上一关：[RMSNorm/RoPE](MY_OPS_IMPLEMENTATION.md)。这里进入对应原Day 9的本地实战。
本关只写计算核心，**Q/K/V已经给定**；不创建Linear投影、RoPE或持久Cache。
不要一次加入五个模块，让第一次报错不知道来自哪一步。
**主要练习入口是 [单文件注释模板](../learning/my_attention_template.py)**：
变量、手算、位置规则、A1–A5和验收命令已集中在模板中。
可直接填写模板，不必为了学习再另建文件；本页仅作可选深入阅读。

## 1. 先手算一个输出17.5

只看最后一个Query，它能读取两个位置：

```text
Q = [0, sqrt(2)*ln(3)]       D_k=2
K位置0=[1,0]，K位置1=[0,1]
V位置0=[10]，V位置1=[20]     D_v=1

Q @ K转置 / sqrt(2) = [0,ln(3)]
softmax = [1/4,3/4]
加权V = 1/4*10 + 3/4*20 = 17.5
```

这里D_k与D_v可以不同；头输出最后宽度来自V，而不是Key位置数。
这不是文字语义测试，只是数值算例。

## 2. 五步写完核心

在 `learning/my_attention_template.py` 内完成以下接口：

```python
def causal_attention(q, k, v, start=0):
    ...
```

模板已提供参数校验；保留A1–A5注释作引导，完成后将占位异常换成return。
如果你更愿意保留一份空白模板，也可另建my_attention.py并将candidate换为那个路径，
但这不是必要步骤。
保留 `my_ops.py`，不要把上一关的函数删掉。

| 步骤 | 你实现什么 | shape |
| --- | --- | --- |
| A1 | Q/K矩阵乘，除以sqrt(D_k) | `[B,1,T_q,T_kv]` |
| A2 | 构造Query绝对位置与Key位置 | `[T_q]`、`[T_kv]` |
| A3 | key位置大于query位置为禁止，填负无穷 | 广播到分数 |
| A4 | 沿Key轴softmax | `[B,1,T_q,T_kv]` |
| A5 | 权重矩阵乘V | `[B,1,T_q,D_v]` |

本关FP32、同device、单头，输入非空。
负start、Query范围超出Key长度、Q/K维度不匹配、K/V位置数不同都应抛ValueError。
保留autograd，不能在函数内部detach或进入no_grad。

## 3. start为什么不能忽略

完整3位置Prefill：

```text
start=0，T_q=T_kv=3
允许：[1,0,0]
      [1,1,0]
      [1,1,1]
```

历史已有4个位置，本次给3个新Query时：

```text
start=4，T_q=3，T_kv=7
Query绝对位置=[4,5,6]
Key位置=[0,1,2,3,4,5,6]
每行分别允许前5/6/7个Key
```

这里只模拟“已经给定历史K/V”的矩形匹配，不实现Cache保存/追加。
未来分数不能填0，因为exp(0)=1；应该在softmax前让禁止位置不贡献权重。
本模板bool True表示禁止，与SDPA的bool True表示允许的语义不要混用。

## 4. 本地验收命令

现有Jupyter Terminal中，从自己的文件运行：

```bash
cd /workspace
python -m learning.local_practice --task attention --device cpu --candidate learning/my_attention_template.py
python -m learning.local_practice --task attention --device cuda --candidate learning/my_attention_template.py
```

未传candidate时，仅运行 `reference_baseline`；用户candidate未完成时不能记成个人通过。
每轮使用新报告名，默认自动生成到 `logs/practice/`，旧结果不覆盖。
从宿主机运行要先把新文件有选择地同步进 `/workspace/learning/`；
Jupyter卷不等于宿主机checkout，上一关首次报“candidate不存在”就是两处文件未同步。

14项检查：

```text
17.5手算、完整Prefill、T=1、单Query历史匹配、多Query历史匹配
未来K/V扰动不影响前缀、零Q在有效Key上均匀汇集、输入不原地修改
Q/K/V反向梯度有限并对齐SDPA
负start、Query越界、K/V长度不符、Q/K维度不符、空Key拒绝
```

独立数值oracle用CPU FP64逐Query截取合法Key，不复用你的Mask代码。
这些是本关支持范围的验收，不是全部Attention变体、低精度、训练质量或性能证明。

## 5. 反复练习与故障注入

1. 先手写17.5与三角可见表，再填A1–A5。
2. 先让CPU通过，失败从第一个case定位，不先放宽容差。
3. 故意去掉sqrt(D_k)，检查手算/随机数值测试能否发现。
4. 故意softmax沿Query轴、未来填0或忽略start，观察哪些项失败，再修回。
5. 新增自己的B/T_q/T_kv/D_k/D_v组合，再明确请求CUDA。
6. 口述“哪一轴在点积后消失、哪一轴在汇集后消失”，填写实战日志。

上一关文件中的已完成TODO和return后的占位异常可在你确认后清理，
但不需要为了进入本关重写已经通过的计算。练习与参考实现分开，不自动提交你的填写结果。

## 6. 文件职责：为什么集中在一个模板

本关不再要求另建 `my_attention.py`。这样安排是为了让“解释、待实现步骤和运行命令”
在同一个可执行入口中，减少在多个文件之间来回猜测。

| 文件 | 职责 | 是否是你的实现 |
| --- | --- | --- |
| `learning/my_attention_template.py` | 主要练习入口；A1–A5、变量含义、手算例和命令都在这里 | 是，填写后作为candidate验收 |
| `docs/PRACTICE_P1_02_ATTENTION.md` | 解释公式、shape、Mask、`start`、失败模式和验收边界 | 否，学习与复盘文档 |
| `learning/practice02_attention.ipynb` | 把手算、Mask可视化和验收命令放进Jupyter，便于逐格观察 | 否，可选观察工具 |
| `learning/practice_attention.py` | 独立oracle、负例和梯度验收；不应被改成“放宽测试”的版本 | 否，验收器/参考基线 |
| `learning/local_practice.py` | 根据 `--task attention` 加载candidate并生成报告 | 否，统一入口 |
| `learning/my_ops.py` | 上一关已经完成的RMSNorm/RoPE实现 | 是，保留，不在本关覆盖 |

因此，本关的唯一实现动作是：**填写
`learning/my_attention_template.py` 的A1–A5，先CPU后CUDA**。
参考baseline通过只说明验收器和环境可用；candidate通过才记录为你这一关的实测结果。
模板中的 `NotImplementedError` 在填写前是故意保留的，不能通过删除测试或直接调用参考实现来绕过。

## 7. 阅读与边界

- [单头Attention详细课](DAY09_SINGLE_ATTENTION.md)
- [Day 5汇集到输出投影的手算](ATTENTION_HAND_CALC.md)
- [PyTorch 2.10 SDPA：参考公式、shape与Mask语义](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [本关验收器与参考基线](../learning/practice_attention.py)

P1-02完成后再进P1-03多头拆合，不提前加入GQA、FlashAttention性能或服务调度。
