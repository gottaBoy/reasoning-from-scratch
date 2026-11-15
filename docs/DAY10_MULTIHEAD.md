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

<a id="tensor-stride"></a>

### 2.1 stride究竟是什么：索引每走一步，去货架哪里找数？

本节讨论普通、非空的dense strided Tensor，不涉及稀疏或其他特殊布局。
先分清：**shape管索引范围，stride管寻址距离，二者不是同一个东西。**

把storage想成一排编号格子，把三维Tensor想成“箱子/排/格”的目录：

```text
shape=[2,3,4]：有2箱，每箱3排，每排4格。

                    第0格 第1格 第2格 第3格
箱0  第0排的编号：      0     1     2     3
     第1排的编号：      4     5     6     7
     第2排的编号：      8     9    10    11
箱1  第0排的编号：     12    13    14    15
     第1排的编号：     16    17    18    19
     第2排的编号：     20    21    22    23
```

这是索引目录，不表示物理内存真的分成三维货架。FP32的底层storage仍是一维字节区，
上图按每4字节一个元素编号；例子刻意让元素值等于编号，方便验算。

#### 定义：固定其他索引，只让某条轴的索引加1

`stride[i]`是**第i条轴的索引增加1时，在底层storage中前进的元素数**，
单位是“元素”，不是“字节”，也不是该轴本身有多少个元素。

例如上面的默认连续布局：

| 固定其他索引，只改变 | 示例 | 存储编号变化 | 对应stride |
| --- | --- | --- | --- |
| 格号+1 | `[0,0,0] -> [0,0,1]` | `0 -> 1` | `stride[2]=1` |
| 排号+1 | `[0,0,0] -> [0,1,0]` | `0 -> 4` | `stride[1]=4` |
| 箱号+1 | `[0,0,0] -> [1,0,0]` | `0 -> 12` | `stride[0]=12` |

所以shape是`[2,3,4]`，stride却是`[12,4,1]`：
前者回答“有几箱/几排/几格”，后者回答“换箱/换排/换格时跳多远”。

#### 为什么默认连续布局要把右边各维乘起来？

创建普通行优先（row-major）连续布局时，最后一轴的相邻元素挨着放，
所以最后一个stride是1。前一轴每走一步，必须越过右侧的一个完整数据块：

```text
shape=[2,3,4]
最后一轴：1个元素                         -> stride[2]=1
排轴：    一整排有4个元素                 -> stride[1]=4
箱轴：    一整箱有3排，每排4个元素        -> stride[0]=3*4=12

一般的默认连续布局：
stride[-1] = 1
stride[i]  = shape[i+1] * stride[i+1]
           = 右侧所有轴长度的乘积
```

**这个递推是默认连续布局的构造规则，不是对任意Tensor由shape求stride的通用公式。**
转置、切片、expand都可能改变stride，必须结合产生它的操作，或直接查`.stride()`。
stride不是模型学出来的参数，也通常不需要你手动设置：PyTorch在创建布局、
执行view、转置或切片时维护它；`.stride()`读取的就是当前实际的寻址规则。
大小为1的轴不发生有效索引移动，它的stride也不能仅由shape唯一确定；
空Tensor和channels-last等布局另看具体定义，不强套这个口诀。

#### 计算一个元素在哪个位置

普通strided Tensor的storage元素编号为：

```text
storage_index = storage_offset + Σ(index[i] * stride[i])

x[1,2,3] -> offset + 1*12 + 2*4 + 3*1
          = 0 + 12 + 8 + 3
          = storage第23号元素
```

从storage起始地址算字节地址时才乘元素字节数：

```text
byte_address = storage_base_address
             + storage_index * element_size
```

例子使用FP32，`element_size=4`，所以第23号元素距离storage开头`23*4=92`字节。
公式不是CPU/GPU实际执行的逐步指令描述，而是普通Tensor的索引到存储映射。

```python
import torch

base = torch.arange(24, dtype=torch.float32)
x = base.reshape(2, 3, 4)
print(x.shape, x.stride())               # [2,3,4], (12,4,1)
assert x[1, 2, 3].item() == base[23].item() == 23
```

#### view与transpose到底怎样确定新stride？

`view`不是必须先连续，而是新形状必须能用原storage和兼容stride表示。
在这个例子中，`view`把每个4维特征拆成`H=2、D=2`：

```python
split = x.view(2, 3, 2, 2)               # [B,T,H,D]
heads = split.transpose(1, 2)            # [B,H,T,D]
print(split.shape, split.stride())        # [2,3,2,2], (12,4,2,1)
print(heads.shape, heads.stride())        # [2,2,3,2], (12,2,4,1)
```

| 张量 | 轴含义 | shape | stride |
| --- | --- | --- | --- |
| `x` | `[B,T,H*D]` | `[2,3,4]` | `[12,4,1]` |
| `split` | `[B,T,H,D]` | `[2,3,2,2]` | `[12,4,2,1]` |
| `heads` | `[B,H,T,D]` | `[2,2,3,2]` | `[12,2,4,1]` |

拆头时，一个head内部相邻特征跳1格，换head跳2格，换时间仍跳4格。
转置时**shape和stride的第1、2项一起交换**，不是把数据搬成新的一维顺序。
`heads`最后一轴仍连续，但整个张量不是默认连续布局，不能看到最后一个stride为1就说连续。

```text
heads[1,1,2,0]：
  第1批、第1个head、第2个时间、第0个特征
  storage_index = 1*12 + 1*2 + 2*4 + 0*1 = 22
```

```python
assert heads[1, 1, 2, 0].item() == base[22].item() == 22
assert heads.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
assert not heads.is_contiguous()
```

别把`is_contiguous()`误认为“数据是否在一个storage里”；它检查当前shape/stride
是否符合指定的连续内存格式，默认是`torch.contiguous_format`。

#### storage_offset与“同一storage但data_ptr不同”

只取每批后两排，数据仍留在原storage里：

```python
tail = x[:, 1:, :]
print(tail.shape, tail.stride(), tail.storage_offset())
# [2,2,4], (12,4,1), 4
assert tail[0, 0, 0].item() == base[4].item() == 4
assert tail.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
assert tail.data_ptr() == x.data_ptr() + 4 * x.element_size()
```

`shape`变小了，但换批仍要跳原来的一整箱12个元素，不是`2*4=8`。
`tail.storage_offset()`是4，表示逻辑起点从原storage第4号元素开始。
**`tensor.data_ptr()`是这个Tensor第一个逻辑元素的地址，不一定是storage的基址。**
本例`tail/x`共享storage而`data_ptr()`不同；在这些非空普通张量的实验里，
比较`untyped_storage().data_ptr()`来观察是否指向同一storage，并同时打印offset。
不要把`data_ptr()`相等当作所有Tensor共享关系的完整判定器。

```python
every_other = base[::2]
print(every_other.shape, every_other.stride())   # [12], (2,)
assert every_other[1].item() == base[2].item() == 2
```

这个反例说明：一维Tensor的stride也不保证是1；步长切片保留了取数间隔。
`expand`则可能产生stride为0的轴，即多个逻辑位置读同一个存储元素；
不要对这种重叠视图随意原地写入。

#### reshape与contiguous不会替你理解时间轴和head轴

正确合头先换回轴顺序，再合并：

```python
merged = heads.transpose(1, 2).reshape(2, 3, 4)
wrong = heads.reshape(2, 3, 4)
packed = heads.contiguous()
assert torch.equal(merged, x)
assert not torch.equal(wrong, x)
assert packed.is_contiguous()
assert torch.equal(packed, heads)
assert not torch.equal(packed.reshape(2, 3, 4), x)
print(wrong[0])   # [[0,1,4,5], [8,9,2,3], [6,7,10,11]]
```

`reshape`按**当前逻辑索引顺序**重排形状，兼容时返回view，否则复制；
`contiguous()`按当前逻辑顺序生成连续布局，需要复制时才复制。
两者都不会替你推断“我其实想先把同一位置的各头拼到一起”。
直接reshape的错误是轴语义不对，不是加一次`contiguous()`就能修复。

| 操作 | 是否生成新内容 | 数据/布局行为 |
| --- | --- | --- |
| `arange` | 是，创建数列 | 本例分配连续storage；dtype显式用FP32 |
| `Linear`投影 | 是，矩阵乘加 | `y=x@W.T+b`，不是把原数字重新分组 |
| `view` | 否 | 共享storage；不兼容就报错，不偷偷复制 |
| `transpose` | 否 | 普通strided Tensor共享storage；交换两轴的shape/stride |
| `reshape` | 否，不做特征乘加 | 返回view或复制，取决于原布局与目标形状 |
| `contiguous` | 否，不做特征乘加 | 已符合目标格式时返回原Tensor，否则复制为目标连续布局 |

例如`x=[2,1]`、`W=[[1,0],[0,1],[1,1]]`且无bias，投影得到`[2,1,3]`：
数值/元素个数都可能改变；`view`不能凭空把2个元素变成3个。

在Jupyter顺序运行上述单元格，再打印一张总表：

```python
for name, tensor in (("base", base), ("x", x), ("split", split),
                     ("heads", heads), ("tail", tail), ("merged", merged),
                     ("wrong", wrong), ("packed", packed), ("every_other", every_other)):
    print(name, "shape=", tuple(tensor.shape),
          "stride=", tensor.stride(), "offset=", tensor.storage_offset(),
          "element_bytes=", tensor.element_size(),
          "contiguous=", tensor.is_contiguous(),
          "storage_ptr=", tensor.untyped_storage().data_ptr(),
          "first_ptr=", tensor.data_ptr())
```

观察地址相同/不同的关系，不背每次运行都会变化的绝对地址。
本实验是布局与正确性验证，不是GPU带宽或算子加速证明。

#### 自测与答案

- shape是`[2,3,4]`，stride一定是`[12,4,1]`吗？不一定，这只是默认连续布局。
- stride是“某一轴的长度”吗？不是，是该轴索引加1时的元素寻址距离。
- 转置后需要重新按右侧乘积算stride吗？不需要，交换对应stride项即可。
- stride的单位是字节吗？不是；字节偏移还需乘`element_size()`。
- 连续数据是否就能随意view？不是，新形状还必须元素数一致；
  非连续张量也不是一概不能view，关键是新旧布局是否兼容。
- `contiguous()`能修复合头漏transpose吗？不能，它不改变当前逻辑轴语义。

定义与边界对照PyTorch 2.10官方文档：
[stride](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.stride.html)、
[Storage](https://docs.pytorch.org/docs/2.10/storage.html)、
[Tensor Views](https://docs.pytorch.org/docs/2.10/tensor_view.html)、
[view兼容条件](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.view.html)、
[expand与零stride](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.expand.html)。

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
