# 学习进度

用户确认的当前学习位置：**Day 7：请求生命周期与调度地图**。
更新时间：2026-11-07（America/Los_Angeles）。用户要求完善Day 6教程并继续下一天。

| 维度 | 状态 |
| --- | --- |
| 学习位置 | Day 7，学习中 |
| Day 1–4 | 已讲解并准备/验证实验，个人自测结果仍由学习者填写 |
| Day 5 | 已完成本轮讲解，用户决定进入下一天；个人独立自测结果尚未登记 |
| Day 6 | 本轮讲解与教程补充收尾，用户决定进入下一天；个人独立验收尚未登记 |
| Day 7 | 已开始；按请求状态、调度轨迹、分页映射顺序学习，个人验收待填写 |
| Day 8–30 | 后续材料已准备，尚未登记个人验收；参考实验结果见验证记录，Day 22 外部服务未配置 |
| 掌握程度 | 不因助手写完参考答案或测试通过而自动提升 |

此前已按用户要求自动准备后续材料；2026-11-07用户先后明确推进到Day 6、Day 7，
不是仅因材料准备或参考实验通过而自动推进。先预测，再执行，再记录自己的结果。
执行入口和实际覆盖范围见 [后续课程](COURSE_DAY06_30.md)。
每天的知识点与面试主问/追问见 [知识索引](KNOWLEDGE_INDEX.md)，
测试与执行证据见 [验证记录](COURSE_VALIDATION.md)。

## Day 7 起步

今日记录：[Day 7学习记录](DAY07_STUDY_LOG.md)。
先读 [请求生命周期与调度地图](DAY07_SCHEDULER_MAP.md)，再打开
[Day 7交互式Notebook](../learning/day07_vllm_map.ipynb)，
观察A/B/C请求在sequential/static/continuous三种策略下的轨迹。
本日使用CPU模拟，不部署真实vLLM，不将round换算为GPU性能。
调度图看完后继续 [四个角色、抽屉结构与生命周期](DAY07_KV_LIFECYCLE.md) 和
[真实CPU页池/小模型实验](../learning/day07_kv_lifecycle.ipynb)。
新增实际KV写入、非连续页复用与异常回收，仍不代表生产PagedAttention或GPU性能验证。

## Day 6 学习回顾

[Day 6学习记录](DAY06_STUDY_LOG.md)保留个人验收待填项，并增加收尾总结。
已补充 [Online Softmax图解](DAY06_ONLINE_SOFTMAX.md)、
[两块小数字Notebook](../learning/day06_online_softmax.ipynb)：
m/l/o含义、换基准手算、向量Value、数值对齐、数组大小和FlashAttention收益边界。
参考验证不等于个人已完成独立实现或生产GPU kernel验收。

## Day 5 学习回顾

2026-11-01 补充 [MLA 直观解释](MLA_INTUITION.md) 与
[Jupyter 对齐实验](../learning/day05_mla_intuition.ipynb)：
先用两个数生成不同头的 K/V，再验证展开计算与 latent 计算的等价关系。
包括逐 token 状态、GQA 区别、矩阵吸收、位置 Key 和面试参考答案。
材料与实验准备完成，不代表个人已通过 MLA 自测。

2026-11-07进一步讲解并补充：

- E/F/H/D及T_q/T_kv的英文名称、直观意义与相互关系。
- 先线性投影、后拆头；D不一定等于E/Hq。
- 两次残差分叉图、GQA共享KV、Attention加权汇集与输出投影的手算。
- [统一符号与发音规范](NOTATION.md)、[Attention逐步手算](ATTENTION_HAND_CALC.md)
  以及已验证的对应Notebook。

这些是讲解与材料记录；个人独立复现、口述和失败实验仍由学习者记录。

## 个人验收

每完成一天，在自己的笔记中填写：

```text
Day：
我能独立解释：
我能独立复现：
我做过的失败实验：
与参考答案不同的地方：
仍不理解：
是否进入下一天：
```

外部框架安装失败、GPU 项被跳过、教学模拟与真实服务测试必须分别记录。
不要用“读过”“代码存在”“助手跑通”代替个人验收。
