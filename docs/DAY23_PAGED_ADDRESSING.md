# Day 23：分页地址翻译，目录不是数据，逻辑顺序不是物理顺序

前置：[Day 7抽屉图解](DAY07_KV_LIFECYCLE.md)，这里开始自己验证地址结构。

## 1. 图解与数字例子

每个请求拿目录找自己的抽屉，而不是要求资料在大柜子里连成一段：

```text
block_size=4，table=[7,2,9]
请求位置0..3 → 物理块7
请求位置4..7 → 物理块2
请求位置8..  → 物理块9
```

逻辑位置6：`logical=6//4=1`，`offset=6%4=2`，
物理块为table[1]=2。不能把“逻辑块1”直接当“物理块1”。
位置编号从0开始，位置6其实是第7个位置。

## 2. 概念与计算契约

教学storage轴顺序为 `[physical_block,layer,K_or_V,Hkv,offset,D]`。
请求表保存table、capacity、每层有效lengths与active；free队列记录可借物理块。
没有Batch轴不是没有请求隔离：请求身份由目录映射，读取后可gather为 `[1,Hkv,T,D]`。

本池按最大capacity预留，分配粒度固定，不代表当前vLLM所有布局。
`get(layer)`按table逻辑顺序拼KV；它是显式gather与torch.cat，
不是直接按页读取的生产GPU attention kernel。

## 3. Jupyter与源码路线

打开 [Day 23 Notebook](../learning/day23_course.ipynb)，查看block_table与logical_mapping。
参考流程释放A再申请C，使C目录非连续；定位 `PagedRequest.locate/get`。
进一步运行 [Day 7真实存取](../learning/day07_kv_lifecycle.ipynb)，
写各不相同的数值标记，证明拼回来的不仅地址对、数据顺序也对。

自己画table=[7,2,9]的位置0/3/4/6/8映射，再与divmod核对。
注意图里随便列的块号不代表当前小池一定有那些物理块。

## 4. 调试与优化顺序

1. 核对position范围，capacity不是已写入长度。
2. 检查logical block、table项和offset，别按物理编号排序读取。
3. 逐层对齐连续KV oracle，再检查相同物理块的复用。
4. 先验证所有权后再测gather/copy成本，明确成本来自教学路径。
5. 扫block_size时固定真实长度分布，分别记录页尾浪费、表项和kernel性能。

7个位置、页宽4要2页容量8，仍浪费1槽；分页不等于零碎片或压缩KV维度。

## 5. 自测与参考答案

**问：table=[7,2,9]读取顺序是2/7/9吗？**
答：不是，必须按逻辑目录7/2/9，还原请求历史顺序。

**问：目录能代替KV向量吗？**
答：不能，目录只是地址，数值在物理storage中。

**问：页越小越好吗？**
答：不一定，页尾浪费降低但表项和访问复杂度增加，需工作负载与硬件实测。

**问：这个分页器就是PagedAttention kernel吗？**
答：不是，管理器和读取/计算kernel是协作但不同的模块。

## 6. 验收与资料

- [ ] 用非连续目录定位并读回不同数值，不只检查shape。
- [ ] 解释capacity、有效length与物理容量的区别。

深入：[PagedAttention论文](https://arxiv.org/abs/2309.06180)；
[vLLM分页设计](https://docs.vllm.ai/en/latest/design/paged_attention/)。
下一课：[Day 24](DAY24_SCHEDULER_FAIRNESS.md)。
