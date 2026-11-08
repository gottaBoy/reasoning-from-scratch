# 学习进度

用户确认：**概念课程已全部阅读，进入本地实战**。
当前关卡：**P1-01：自写RMSNorm/RoPE的CPU与CUDA验收**（对应原Day 8实现内容）。
更新时间：2026-11-08（America/New_York）。概念阅读完成为用户自述，独立实现与性能验收分别记录。

| 维度 | 状态 |
| --- | --- |
| 学习位置 | 本地实战P1-01，学习中 |
| 概念阅读 | 用户确认全部看完，不自动等于全部模块可独立实现 |
| Day 1–4 | 已讲解并准备/验证实验，个人自测结果仍由学习者填写 |
| Day 5 | 已完成本轮讲解，用户决定进入下一天；个人独立自测结果尚未登记 |
| Day 6 | 本轮讲解与教程补充收尾，用户决定进入下一天；个人独立验收尚未登记 |
| Day 7 | 本轮教程与实验已完成准备，个人独立验收尚未登记 |
| Day 8实现内容 | 作为P1-01第一关，自己的candidate代码尚未验收 |
| Day 9–30实现内容 | 概念已读，参考材料已备齐；按实战关卡独立实现，Day 22真实服务仍未配置 |
| 掌握程度 | 不因助手写完参考答案或测试通过而自动提升 |

此前已按用户要求自动准备后续材料；2026-11-07用户先后明确推进到Day 6、Day 7、Day 8，
不是仅因材料准备或参考实验通过而自动推进。先预测，再执行，再记录自己的结果。
执行入口和实际覆盖范围见 [后续课程](COURSE_DAY06_30.md)。
每天的知识点与面试主问/追问见 [知识索引](KNOWLEDGE_INDEX.md)，
测试与执行证据见 [验证记录](COURSE_VALIDATION.md)。
用户另要求一次准备全部剩余教程，详细入口见
[Day 9–30逐日教程](COURSE_DAY09_30_DETAILS.md)。
当时准备和助手验证没有自动推进个人进度；本次用户明确选择转入本地实战，
按 [反复练习路线](LOCAL_PRACTICE.md) 与 [实战日志](LOCAL_PRACTICE_LOG.md) 推进，
不自动勾选后续独立实现或生产服务验收。

## 剩余课程准备与提交记录

- 教程主体提交：`149d7f2 Prepare detailed Day 9-30 learning guides`，已推送到`origin/main`。
- 范围：Day 9–30共22份详细教程，配套Notebook补充图解和自测参考答案；
  原代码、输出、单元格ID和个人记录保留，并已同步到现有Spark Jupyter。
- 参考执行：22份Notebook均执行结束，21项实验通过；
  Day 22为`external_service_not_configured`，不计通过。
- 最近普通回归：`347 passed / 27 skipped / 1 warning`；
  GB10提示保留，测试与执行记录详见 [验证报告](COURSE_VALIDATION.md)。
- 上述历史补记未重复执行全部实验，当时保留Day 8位置；现在按用户新决定转入本地实战。

## 当前：本地实战起步

先打开 [实战P1-01 Notebook](../learning/practice01_start.ipynb)：
CPU参考基线 → CUDA参考基线 → 你自己的 `learning/my_ops.py` → candidate验收。
验收器：[local_practice.py](../learning/local_practice.py)。
未传candidate时只验证参考实现；传文件也不自动证明独立创作或精通。
本地第一关不需要重建镜像、下载模型或升级现有依赖。

## Day 8 参考入口

今日记录：[Day 8学习记录](DAY08_STUDY_LOG.md)。
先读 [RMSNorm与RoPE教程](DAY08_RMSNORM_ROPE.md)，再运行
[逐步Notebook](../learning/day08_rmsnorm_rope.ipynb)。
重点是独立实现契约、边界输入、offset和误差，不加载Qwen3，不以文本输出评价正确性。

## Day 7 学习回顾

Day 7记录：[Day 7学习记录](DAY07_STUDY_LOG.md)和
[四角色、存储与生命周期图解](DAY07_KV_LIFECYCLE.md)。
已补充抽屉账本、真实CPU小模型调用、异常回收和官方深入链接；
个人独立口述和性能验收仍由学习者填写。
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
