# Day 11：GQA/MQA，让多个问题共享资料，不共享提问

前置：[Day 10](DAY10_MULTIHEAD.md)；维度与别名见 [统一符号](NOTATION.md)。

## 1. 图解与数字例子

固定Hq=4，把独立KV头数从4改成2或1：

```text
MHA： Q0→KV0，Q1→KV1，Q2→KV2，Q3→KV3
GQA： Q0/Q1→KV0，Q2/Q3→KV1
MQA： Q0/Q1/Q2/Q3→KV0
```

共享资料柜不意味着四个问题答案相同：各Q不同，权重也可不同。
固定L=2、B=1、T=8、D=8、FP32：
紧凑总KV为 `2*L*B*T*Hkv*D*4` bytes，Hkv=4/2/1分别为4096/2048/1024 bytes。
这只是KV数组，不是总参数或完整进程峰值。

## 2. 概念与计算契约

普通同D投影：Q宽Hq*D，K/V各宽Hkv*D；共享比例Hq/Hkv。
本实现要求Hq能被Hkv整除，缓存紧凑 `[B,Hkv,T,D]`。
教学计算用repeat_interleave映射到Q heads，物化临时副本可能抵消部分计算阶段内存收益；
不能把临时扩展后的张量当成实际应该永久保存的Cache。

本日每种结构的随机权重不同。改变Hkv会改变K/V参数shape，
不能要求不同结构输出相同，也不能无条件加载同一checkpoint。

## 3. Jupyter与源码路线

打开 [Day 11 Notebook](../learning/day11_course.ipynb)，读取variants中每种Hkv的Cache shape与bytes。
参考 `_cache_lab` 用**同一结构自己的同一套权重**做full/cached对齐。
定位Mini `Attention` 的k/v投影、紧凑update和repeat_interleave先后顺序。

先预测Hkv=1/2/4的紧凑头轴，再运行；之后只改T或dtype，重新手算线性增长。
复用 [两个头共享KV手算](ATTENTION_HAND_CALC.md) 解释不同Q的不同输出。

## 4. 调试与优化顺序

1. 打印Cache更新位置，确认保存的是扩展前KV。
2. 检查Q head到KV head的分组映射。
3. 多步Decode对齐，排除offset和请求身份错误。
4. 区分权重大小、KV大小和临时副本，分别记字节数。
5. 在支持的kernel中再看读取带宽；不能只改配置宣称模型质量或性能不变。

## 5. 自测与参考答案

**问：四个Q共享一个KV，Query参数也变少吗？**
答：Hq和D不变时Q投影宽度不变，改变的是K/V生成与缓存头数。

**问：Hkv减半是否模型总内存减半？**
答：不是，仅相同前提下的紧凑KV部分减半，其他权重和buffer另计。

**问：为什么不拿三种模型的logits直接相等作为验收？**
答：结构和权重不同；应各自做full/cached一致性，再另行评估质量。

**问：GQA等同MLA吗？**
答：不是，MLA改变联合latent表示与计算路径，不只是减少KV头数。

## 6. 验收与资料

- [ ] 画4个Q对应Hkv=1/2/4的映射，算4096/2048/1024 bytes。
- [ ] 说明紧凑持久Cache与临时扩展的区别。
- [ ] 按结构各自对齐，不预填质量结论。

深入：[GQA原论文](https://arxiv.org/abs/2305.13245)、[MLA图解](MLA_INTUITION.md)。
下一课：[Day 12](DAY12_REQUEST_CACHE.md)。
