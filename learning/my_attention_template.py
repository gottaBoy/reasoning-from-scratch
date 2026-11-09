"""P1-02：单文件练习入口，直接在本模板填写A1–A5即可。

本关目标
--------
已给定Q/K/V，自己完成：匹配 → 缩放 → 屏蔽未来 → softmax → 汇集V。
暂不加入Linear投影、RoPE、Cache分配或输出投影；保留已完成的my_ops.py。
本文件可直接导入，不会在顶层调用尚未实现的函数。

先认变量
--------
q / k / v：Query查询向量 / Key匹配向量 / Value待汇集内容。
B：Batch Size，序列数；头数固定为1。
T_q：Query Sequence Length，本次Query位置数。
T_kv：Key/Value Sequence Length，已有K/V位置数。
D_k：Query/Key Head Dimension，每头Q/K特征数。
D_v：Value Head Dimension，每头V特征数，可以不同于D_k。
start：第一个Query在完整序列中的绝对位置，不是张量切片起点。

手算例子
--------
Q=[0, sqrt(2)*ln(3)]，K位置0=[1,0]、位置1=[0,1]。
V位置0=[10]、位置1=[20]；start=1，两个Key都可见。
缩放后的分数=[0,ln(3)] → 权重=[1/4,3/4] → 输出=17.5。
ln是自然对数；所有数字仅用于手算，不代表文字语义。

start与可见范围
---------------
完整Prefill：start=0、T_q=T_kv=3，三行分别允许前1/2/3个Key。
已有4个历史位置：start=4、T_q=3、T_kv=7，允许前5/6/7个Key。
本函数只接收已经准备好的K/V，不实现Cache保存或追加。
禁止未来位置应在softmax前填负无穷，不能填0，因为exp(0)=1。

填写与验收
----------
先手算，再完成下面A1–A5，并用return output替换末尾NotImplementedError。
在现有Jupyter Terminal内执行，无须另建my_attention.py：

    cd /workspace
    python -m learning.local_practice --task attention --device cpu \\
        --candidate learning/my_attention_template.py
    python -m learning.local_practice --task attention --device cuda \\
        --candidate learning/my_attention_template.py

先CPU通过再CUDA；未填完时失败是预期，不删测试、不先放宽阈值。
未传candidate只测参考基线；报告保留在logs/practice，不自动证明性能或作者身份。
宿主机与Jupyter卷是两份文件，验收前确认修改已在/workspace对应路径。

反复练习
--------
14项检查包括手算、方形/矩形Mask、T=1、未来扰动、零Q均匀权重、
输入不被修改、梯度、非法shape/start拒绝。
通过后每次只故意改错一项：去sqrt缩放、未来填0、softmax轴错、忽略start。
记录失败case再修复，最后另选一组B/T_q/T_kv/D_k/D_v检验。

可选深入阅读（不必打开才能完成本关）
----------------------------------
docs/PRACTICE_P1_02_ATTENTION.md
https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import torch


def causal_attention(q, k, v, start=0):
    """同设备FP32输入；q:[B,1,T_q,D_k]，k/v长度相同，输出:[B,1,T_q,D_v]。"""
    # 入口校验已提供。下面只实现计算核心，不修改q/k/v本身。
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Use [B,1,T,D] tensors.")
    if (min(*q.shape, *k.shape, *v.shape) < 1
            or q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1
            or q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]
            or q.shape[-1] != k.shape[-1]):
        raise ValueError("Use compatible nonempty single-head inputs.")
    if any(x.device != q.device or x.dtype != torch.float32 for x in (q, k, v)):
        raise ValueError("Same-device FP32 inputs required.")
    if not isinstance(start, int) or isinstance(start, bool) or not 0 <= start <= k.shape[2] - q.shape[2]:
        raise ValueError("Invalid query start position.")

    # TODO A1：匹配并缩放，结果命名scores。
    # q @ k.transpose(-2, -1)，再除以sqrt(D_k)，D_k=q.shape[-1]。
    # [T_q,D_k] @ [D_k,T_kv] -> [T_q,T_kv]，特征D_k轴经点积求和消失。
    # 保留Batch/head轴，scores完整shape为[B,1,T_q,T_kv]。

    # TODO A2：构造query_positions和key_positions，均在q.device。
    # Query位置：start .. start+T_q-1；Key位置：0 .. T_kv-1。
    # 用[:, None]和[None, :]广播成Query行、Key列，不把两条轴写反。

    # TODO A3：blocked[i,j] = key_positions[j] > query_positions[i]。
    # blocked为[T_q,T_kv]，可广播到[B,1,T_q,T_kv]。
    # 本变量True表示禁止；用masked_fill把对应scores填为-torch.inf。
    # 注意：SDPA的bool attn_mask中True表示允许，与blocked语义相反。

    # TODO A4：probabilities = softmax(scores, dim=-1)，沿Key位置归一化。
    # 每个Query这一行的有效概率和为1，不是在Query轴或head轴求和。
    # shape仍为[B,1,T_q,T_kv]；本关没有dropout。

    # TODO A5：output = probabilities @ v，然后return output。
    # [T_q,T_kv] @ [T_kv,D_v] -> [T_q,D_v]，Key位置轴经加权求和消失。
    # 完整输出shape为[B,1,T_q,D_v]；T_q来自Q，最后特征宽度来自V。
    # 保留autograd：不要detach，不要在此函数内进入no_grad。
    # 不原地修改输入，不导入参考causal_attention代替自己的实现。
    raise NotImplementedError("Fill A1-A5 here, then return output.")
