# Day 10：多头Attention，拆分的是特征，不是用户或时间

前置：[单头Attention](DAY09_SINGLE_ATTENTION.md)、[统一符号](NOTATION.md)。

## 1. 图解与数字例子

把每个头理解成同一段资料上的不同查询方式，不是把一句话分给几个人各看一截。

```text
输入每位置E=8个数
   ↓ Q投影：8 → Hq*D=2*6=12
每位置12个特征 → 2个头，每头6个
   ↓ 各自计算
2张T_q×T_kv匹配表
   ↓ 各头汇集V
合并为12维 → 输出投影回8维 → 接回残差
```

若B=1、T=3，score是 `[1,2,3,3]`，共18个数，
不是把两个头串成一张6×6的跨头大表。
Hq*D不必等于E；**先Linear计算新特征，后view重新分组**。

## 2. 概念与计算契约

```text
[B,T,Hq*D] → view → [B,T,Hq,D] → transpose → [B,Hq,T,D]
各头输出 [B,Hq,T,D] → transpose → [B,T,Hq,D] → reshape → [B,T,Hq*D]
```

view/transpose无可学习参数；输出投影有参数且可跨头混合。
transpose通常改变stride；reshape在需要时可能复制，不能看到shape正确就宣称零拷贝。
合并时先让同一位置的各头相邻，不能在多T情形直接reshape掩盖错位。

## 3. Jupyter与源码路线

打开 [Day 10 Notebook](../learning/day10_course.ipynb)，定位Mini `Attention.forward` 中
`.view(...).transpose(1,2)`和末尾合并。参考实验检查因果前缀及SDPA对齐，
但没有自动替你验证每个stride；应另外打印shape、stride、is_contiguous。
复习 [手算Notebook](../learning/day05_attention_handcalc.ipynb) 的两Query布局负例。

先用标签数 `[[头0位置0,头0位置1],[头1位置0,头1位置1]]` 追踪合并顺序，
再换随机浮点输入。修改E与Hq*D令二者不同，验证out投影仍回到E。

## 4. 调试与优化顺序

1. 每次view前检查元素总数，别把Linear扩维和reshape混为一谈。
2. 每次transpose后记录stride，用标签值检查实际顺序。
3. 故意去掉合并前transpose，构造T>1错误，避免T=1掩盖问题。
4. 同权重对齐后检查隐式contiguous/复制，再决定是否优化布局。
5. 性能比较固定E/H/D/T与dtype；不把“头更多”当作加速或质量证明。

## 5. 自测与参考答案

**问：2个头、每头6维，E=8合法吗？**
答：合法，只要投影为12维并在输出时投影回8维；D=E/Hq只适用于额外规定投影同宽的设计。

**问：为什么是两张3×3表？**
答：每个头在自己的投影空间匹配位置，不让头0的Query直接参加头1的softmax。

**问：输出投影只是降维吗？**
答：还有混合不同头特征的作用，即使输入输出同宽也能保留。

**问：transpose是否等于拷贝所有数据？**
答：一般是视图与stride变化，后续操作是否复制需查实际布局与算子。

## 6. 验收与资料

- [ ] 写出拆头和合头的每一步shape/stride。
- [ ] 用标签而非随机数证明排列正确，保留漏transpose负例。
- [ ] 说明投影参数与布局操作的区别。

深入：[PyTorch Tensor Views](https://docs.pytorch.org/docs/2.10/tensor_view.html)；
[Linear](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Linear.html)。
下一课：[Day 11](DAY11_GQA_MQA.md)。
