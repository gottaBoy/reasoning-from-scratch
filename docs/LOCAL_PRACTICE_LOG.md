# 本地实战日志

开始日期：2026-11-08（America/New_York）。
用户确认：概念课程全部阅读，转入本地实战。
当前关卡：**P1-02，单头因果Attention计算核心**。
教程：[本地反复练习路线](LOCAL_PRACTICE.md)。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 概念阅读 | 用户自述已全部看完，不等于逐模块独立实现 |
| 已有环境 | 复用Spark Docker/Jupyter，不重建、不升级依赖 |
| 验收入口 | `python -m learning.local_practice --task attention`；默认任务仍为norm-rope，可传自己的candidate文件 |
| 参考CPU/CUDA基线 | 本次均14/14通过，implementation_role为reference_baseline |
| P1-01用户candidate | `/workspace/learning/my_ops.py`，支持范围内CPU/CUDA均14/14通过 |
| P1-02用户candidate | 尚未指定/验收，当前只准备模板与参考验收 |
| 后续模型/算子/引擎/服务 | 按关卡推进，未自动通过；真实vLLM仍未配置 |

## 已执行的参考证据

2026-11-08在现有DGX Spark容器执行，未升级或重建环境：

- NVIDIA GB10、PyTorch 2.10.0+cu130；实际矩阵乘法和Attention前后向自检通过。
- P1-01参考实现CPU 14/14、CUDA 14/14通过；未传用户candidate文件。
- 新Notebook三个代码单元格完整执行；原始证据保存在以下容器路径。
- 全仓库普通回归 `355 passed / 27 skipped / 1 warning`；
  保留GB10 capability提示，不把它隐藏或当作所有扩展兼容证明。

```text
/workspace/logs/practice/environment-20261108T075408094557Z.json
/workspace/logs/practice/p1-01-cpu-20261108T075728451905Z.json
/workspace/logs/practice/p1-01-cuda-20261108T075728827959Z.json
/workspace/logs/practice/practice01-executed-20261108T075729344831Z.ipynb
```

以上是助手运行的环境与参考代码证据。你的独立实现、调错记录和性能实验仍待填写。

## P1-01用户文件验收：2026-11-08（America/Los_Angeles）

继续练习时读取到用户已经填写RMSNorm与RoPE。首先尝试容器内验收，
发现candidate文件不存在；这是宿主机与named volume源码未同步，不是数值计算失败。
确认容器目标不存在后，仅复制my_ops.py，未改算法或占位注释。

```text
implementation_role = candidate_file
source = /workspace/learning/my_ops.py
source_sha256 = 70b81c835cd769c7799595b9db44e051bae5b9df741fe4ea7174c2ec2af6bff9
CPU：14 passed / 0 failed
CUDA：14 passed / 0 failed
```

实际报告：

```text
/workspace/logs/practice/p1-01-20261108T161905605802Z.json
/workspace/logs/practice/p1-01-20261108T161922365171Z.json
```

两份报告源码hash一致。代表该文件通过已定义FP32数值、边界、位置及有限梯度检查，
不代表所有dtype/layout、训练质量、性能或独立创作已验证。
源码中的TODO/return后不可达占位仍可由学习者清理，本次不替换或自动提交该文件。

下一关：[P1-02步骤与模板](PRACTICE_P1_02_ATTENTION.md)。
当前任务是自己的 `causal_attention(q,k,v,start=0)`；不把参考baseline记成用户已完成。

## P1-02参考验收：2026-11-08（America/Los_Angeles）

新增A1–A5待填模板、逐步Notebook与 `--task attention` 验收入口。
参考CPU/CUDA各14/14通过，implementation_role为reference_baseline；
用户尚未指定my_attention.py，不登记为个人实现已通过。
本次只验证单头FP32核心，Q/K/V已给定，不含投影、RoPE或Cache管理。

```text
/workspace/logs/practice/p1-02-cpu-20261108T162801132622Z.json
/workspace/logs/practice/p1-02-cuda-20261108T162801698385Z.json
/workspace/logs/practice/practice02-attention-executed-20261108T162802215225Z.ipynb
```

全仓库普通回归364 passed / 27 skipped / 1 warning；保留GB10提示。
源码、文件路径和报告的变化要按各自的hash与时间记录，不覆盖P1-01证据。

## 本轮文档归档决定

本轮确认：P1-02不另建`my_attention.py`，直接在
`learning/my_attention_template.py`完成A1–A5。模板集中放置变量说明、shape推导、
17.5手算例、Mask规则、验收命令和待填写代码；解释、文件职责、失败模式和验收边界
统一记录在 [P1-02教程](PRACTICE_P1_02_ATTENTION.md)。

`practice02_attention.ipynb`只作为逐格观察工具，`practice_attention.py`保持独立oracle
与负例验收，不能为让测试通过而修改。`learning/my_ops.py`属于已完成的P1-01个人文件，
本轮不覆盖、不重写。后续每个实战关卡沿用同一归档方式：

```text
一个可填写的实现模板
→ 一份解释与边界文档
→ 一个独立验收器
→ 一份个人运行日志
```

## 每轮填写

```text
关卡：
源码路径 / source_sha256：
实现来源（自己编写 / 参考改写 / 直接调用参考）：
预测：
CPU报告：
CUDA报告：
第一个失败case：
错误原因：
单变量修改：
新输入验证：
性能是否实测、口径：
当前不支持：
下一轮：
```

每轮使用新报告文件；未通过项保留失败记录，不以删除case或放宽阈值制造通过。
