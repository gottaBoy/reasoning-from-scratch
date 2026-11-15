# P1-07：模型基线与 full/cached 对齐

P1-06 已确认 Block、SwiGLU 和梯度路径。现在固定同一权重、同一输入和同一
dtype/设备，比较 full forward 与 cached forward；正确性先于速度。

## 参考验收

```bash
python -m learning.local_practice --task baseline --device cpu
python -m learning.local_practice --task baseline --device cuda
```

覆盖：

1. 全部 logits 的 full/cached 对齐，不只比较最后一个位置；
2. 修改未来输入不会改变前缀 logits；
3. compact KV Cache 的 shape 和最终长度；
4. warm-up、重复次数、CUDA 同步和计时口径。

计时报告只表示同步端到端 wall-clock，不直接宣称 Cache 加速。
CPU 的 `peak_allocated_bytes=None` 也不等于内存占用为零。
一次 full forward 与分块 cached forward 的调用次数和计算路径不同；
这里检查相同输入位置的计算结果和计时链路，不用两者延迟相除计算生成加速比。

## 固定工作量

参考输入长度为 8，分块为：

```text
prefill 5 + decode 1 + decode 1 + decode 1
```

full 与 cached 必须使用同一份模型权重。每次 cached workload 都要新建 Cache，
不能把前一次请求的历史带进下一次计时。

## 个人模型组装

打开 `learning/my_model.py`，复用已经完成的 `my_block.py`、`my_cache.py`
和 `my_ops.py` 中的RMSNorm。构造函数和上下文预检查已备好，
本次只填 `forward` 的 M1–M3：

```text
M1：token IDs [B,T] → embedding → [B,T,E]
M2：逐层 Block；共享本次起始 position/cache，分别传入 layer 索引
M3：final norm → LM head → logits [B,T,vocab]
```

`position` 必须在第一层更新Cache之前读取，不能每一层重新读取。
LM head输出未经softmax的分数；不要返回概率或argmax。
Attention内部仍复用课程参考，不把这一关说成全部模型算子从零重写。

```bash
python -m learning.local_practice --task baseline --device cpu \
  --candidate learning/my_model.py
```

个人文件未填写时应明确失败；CPU通过后再把 `cpu` 换成 `cuda`。
上面命令需要当前工作区源码可见；宿主机与Jupyter卷不会自动同步，
在容器里运行前先确认是最新文件，不覆盖其他个人文件。

## 个人记录

记录以下字段：

```text
seed、Config、输入 token、dtype、device
full/cached 最大绝对误差
warmup、repeats、是否同步
prefill 时间、decode 时间或明确的端到端时间
peak allocated（仅 CUDA）
```

不同 prompt 长度、不同输出预算可以另行记录，但不能与固定工作量的优化结论混合。
小随机模型的对齐和计时不代表语言质量或生产服务吞吐。
