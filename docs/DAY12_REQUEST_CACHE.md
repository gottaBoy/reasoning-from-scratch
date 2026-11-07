# Day 12：自己实现Cache，别把不同请求的记忆混在一起

前置：[Day 11](DAY11_GQA_MQA.md)、[Day 7抽屉生命周期](DAY07_KV_LIFECYCLE.md)。

## 1. 图解与数字例子

同一个模型像共用的加工规则，Cache像每个请求自己的笔记：

```text
模型权重：A/B共用
A笔记：已存位置0..3 → 新token必须使用位置4
B笔记：已存位置0..1 → 新token必须使用位置2
```

固定8个输入位置，分成3+2+3计算：
每次T_q分别为3、2、3，T_kv分别变成3、5、8。
对相同权重，拼接全部位置的logits应与一次full forward在容差内一致。
最后一个位置的logits预测下一个token，不是“还原当前token”。

## 2. 概念与计算契约

```text
Cache.entries[layer] = (K,V)
K/V shape = [B,Hkv,T_kv,D]
position = 当前Cache长度
new_position = position + 本次输入长度
```

Mini把位置从Cache推导，不在模型上维护单个共享可变游标。
一个forward结束后，各层有效长度应一致；中途层更新顺序不等于已经完成整个请求。
超长输入必须在改变旧Cache之前拒绝。教学Cache用于推理，不提供长序列训练图契约。

## 3. Jupyter与源码路线

打开 [Day 12 Notebook](../learning/day12_course.ipynb)，参考实验默认5+1+1+1。
定位 `Model.forward` 的context检查与 `Attention.forward` 的历史拼接。
再自己尝试3+2+3，记录每段offset与矩形Mask。

做两条独立请求各自创建Cache；同请求复用，不同请求隔离。
改最后一个输入观察前缀不变，故意重置offset看数值错误。
失败用例用独立Cache，避免污染后续参考。

## 4. 调试与优化顺序

1. 先CPU FP32小shape，记录每层旧长、新长、request ID。
2. full/cached比较全部支持logits，不只末位置或top1。
3. 检查上下文越界拒绝后旧Cache还能读取，重复请求不串历史。
4. 再分析重复torch.cat造成的复制；分页是后续布局优化，不先改算法。
5. 低精度单独记录误差，不以FP32逐位相等要求BF16。

注意：本实现预检查context/batch等错误，不保证任意运行中异常都会原子回滚所有层。
生产异常恢复需额外设计，不能从一个越界测试推断通用事务性。

## 5. 自测与参考答案

**问：Decode只输入新token，还需要历史Attention吗？**
答：需要，当前Q仍匹配全部可见历史K；省去的是重投影历史K/V。

**问：清张量但不清位置会怎样？**
答：新Q/K可能使用错误RoPE位置，shape仍合法而数值错。

**问：两用户能共用一个可变Cache吗？**
答：不能无隔离地共用；权重可共享，历史状态需请求身份和所有权。

**问：3+2+3的三张score形状是什么？**
答：省略B/H后为3×3、2×5、3×8，Mask需按绝对位置构造。

## 6. 验收与资料

- [ ] 同权重验证至少两种split与多步增长。
- [ ] 记录拒绝前后长度和每层状态，说明运行中异常边界。
- [ ] 能解释请求级位置与Mask，而不是背“只输入1个”。

深入：[Hugging Face Cache解释](https://huggingface.co/docs/transformers/v4.57.0/en/cache_explanation)；
[Day 4真实权重误差记录](DAY04_KV_CACHE.md)。
下一课：[Day 13](DAY13_BLOCK_BACKWARD.md)。
