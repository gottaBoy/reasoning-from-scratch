# Day 13：Block与反向传播，两次残差不是都加最初的X

前置：[Day 8](DAY08_RMSNORM_ROPE.md)、[Day 12](DAY12_REQUEST_CACHE.md)。

## 1. 图解与数字例子

把Attention看成“查资料后更新”，FFN看成“对更新结果再加工”：

```text
X ── Norm → Attention → A ── + ← X
                            ↓
                           Y
Y ── Norm → SwiGLU   → U ── + ← Y
                            ↓
                           Z
```

只用标量记连接：X=10、A=2，则Y=12；U=3则Z=15。
若第二次错加原X会得到13，计算虽然能运行，语义已不同。
真实向量A/U由模型计算；这个例子只是连线，不是Attention的数值模型。

## 2. 概念与计算契约

```text
Y = X + Attention(norm1(X))
Z = Y + down(SiLU(gate(norm2(Y))) * up(norm2(Y)))
gate/up: E→F；down:F→E
```

gate/up两套权重，不是同一矩阵调用两遍；SiLU不局限0..1。
无bias的SwiGLU三矩阵共有3EF参数。
训练要保留autograd图；`eval()`控制模块行为，不等于关闭梯度。
`no_grad/inference_mode`是另外的契约，不能在推理模式里期待自动得到训练backward。

## 3. Jupyter与源码路线

打开 [Day 13 Notebook](../learning/day13_course.ipynb)，定位Mini `Block/FFN`。
参考实验在明确启用梯度的范围里算小随机模型cross-entropy，
检查全部参数grad存在且有限；只是一次梯度链检查，不代表训练收敛。

自己用hooks捕获Y/FFN输入，核对第二次残差。
按真实训练任务构造target时，明确当前位置logits对应哪个下一个token；
不要把能得到一个loss当作标签方向正确。
暂不把推理Cache引入训练，先做无Cache的小序列。

## 4. 调试与优化顺序

1. 打印requires_grad与grad_fn，检查图是否被detach/推理上下文截断。
2. loss必须有限；backward后检查每个参数，而不是只看loss下降一次。
3. 重复更新前用zero_grad，区分梯度累积与单步梯度。
4. 用独立手算/小模块定位残差与门控，必要时FP64 gradcheck适用模块。
5. 之后才考虑激活重算/混合精度，分别验证梯度与内存。

本课未实现完整优化器训练、AMP、训练收敛或融合算子的backward。

## 5. 自测与参考答案

**问：第二次加的是X还是Y？**
答：是第一次Attention残差后的Y。

**问：SiLU gate等于MoE router吗？**
答：不是，前者调节通道，后者选择完整expert，作用层级不同。

**问：eval后为何仍有梯度？**
答：eval不关闭autograd；是否记录计算图由梯度上下文和requires_grad决定。

**问：所有grad有限就证明模型学会了吗？**
答：不能，只证明这次梯度路径存在且未产生非有限值，还需优化、数据与评测。

## 6. 验收与资料

- [ ] 能用10→12→15解释两次残差，并用hooks确认真实接口。
- [ ] 独立跑一次loss/backward，记录参数梯度与模式。
- [ ] 说明3EF的三份来源及未验证的训练边界。

深入：[PyTorch Autograd](https://docs.pytorch.org/docs/2.10/notes/autograd.html)；
[SwiGLU图解](ROPE_SWIGLU_INTUITION.md)。
下一课：[Day 14](DAY14_MODEL_BASELINE.md)。
