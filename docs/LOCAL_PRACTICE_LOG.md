# 本地实战日志

开始日期：2026-11-08（America/New_York）。
用户确认：概念课程全部阅读，转入本地实战。
当前关卡：**P1-01，自己的RMSNorm/RoPE代码验收**。
教程：[本地反复练习路线](LOCAL_PRACTICE.md)。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| 概念阅读 | 用户自述已全部看完，不等于逐模块独立实现 |
| 已有环境 | 复用Spark Docker/Jupyter，不重建、不升级依赖 |
| 验收入口 | `python -m learning.local_practice`；默认测参考，可传自己的candidate文件 |
| 参考CPU/CUDA基线 | 本次均14/14通过，implementation_role为reference_baseline |
| 用户candidate文件 | 尚未指定，不替用户登记为完成 |
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
