# Day 22：真实模型服务，先打通正确请求，不先追求大并发

前置：[Day 7四角色](DAY07_KV_LIFECYCLE.md)。本日有外部依赖，默认不自动安装或启动服务。

## 1. 图解与数字例子

把服务分成前台与车间，不是HTTP返回成功就代表所有内层优化正确：

```text
客户端 → API校验/鉴权 → 引擎/调度 → 模型加载与执行 → 返回输出
                         ↘ KV管理/实际Attention后端
```

探针发送temperature=0、max_tokens=8的小请求；
得到HTTP 200和文本，只证明这次协议/加载/推理链路可达。
它不证明分页kernel、请求取消、首token延迟或大规模容量。

## 2. 概念与计算契约

仓库 `.pth` 是自定义模型实现的checkpoint，不能直接当成通用HF目录。
服务模型必须有目标框架支持的配置、tokenizer、权重格式与实际架构。
Spark是ARM64/GB10，安装前核对目标版本支持矩阵、CUDA/驱动和扩展；
不要在现有已验证torch环境里直接pip upgrade来试运气。

本探针仅GET models、POST non-streaming completions；
使用返回的第一个模型ID，没有自动鉴权key传递，不测SSE/TTFT。
无授权服务时状态是external_service_not_configured，**不是passed**。

## 3. Jupyter与源码路线

打开 [Day 22 Notebook](../learning/day22_course.ipynb)。
没有服务可直接学习mock测试与错误状态；不为跑满课程擅自部署重量级框架。
准备实际部署时，单独镜像/虚拟环境、独立volume/端口，保留旧工作区和回滚入口。
具体安装参数按所选版本官方资料核对，本课不固定不可靠的“latest兼容组合”。

只有服务已启动且获授权，才在Notebook设置包含/v1的地址：

```python
os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
```

不要把token/密码写入笔记或Git，不把开发固定Jupyter token、公网0.0.0.0监听当生产安全方案。
探针如果遇到鉴权服务，先按其协议实现凭证管理，不能去掉服务鉴权绕过。

## 4. 调试与优化顺序

1. GPU注入、实际CUDA张量与适用kernel，先于模型并发。
2. 支持的模型/格式/tokenizer、上下文与精度，先于HTTP性能。
3. 单请求输入输出及服务日志，检查model ID和请求schema。
4. 区分连接错误、4xx请求/鉴权、5xx加载/执行，不把一切归结网络。
5. 稳定后才按Day 27定义工作负载、尾延迟与吞吐，再调预算。

本课程不修改现有教学镜像，不自动接触未知远程端点。

## 5. 自测与参考答案

**问：没配置服务，Notebook跑到末尾就算通过吗？**
答：不算，报告仍是external_service_not_configured，外部集成尚未验证。

**问：总请求耗时能直接叫TTFT吗？**
答：不能，non-streaming只有整次完成时长，没有首token到达时刻。

**问：为什么先隔离环境？**
答：框架可能要求不同torch/CUDA/Triton，避免破坏已验证环境并方便回滚。

**问：API成功证明PagedAttention更快吗？**
答：不能，还需实际后端、工作负载、正确性与计时证据。

## 6. 验收与资料

- [ ] 写清是否有真实服务、版本、模型身份、授权、日志和回滚方式。
- [ ] 未运行的部署、鉴权、流式、性能均单列状态。

深入：[vLLM安装](https://docs.vllm.ai/en/latest/getting_started/installation/)；
[兼容服务接口](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)；
[本课探针](../learning/vllm_probe.py)。
下一课：[Day 23](DAY23_PAGED_ADDRESSING.md)。
