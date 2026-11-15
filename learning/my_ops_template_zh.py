"""统一练习模板：中文图解版。看这里的提示，只在 learning/my_ops.py 中实现。

这不是另一份作业，而是 my_ops_template.py 的中文对照版：
函数名、参数、校验、TODO 编号都一样，只增加中文直觉和手算。
模板故意没有完成计算，导入不会报错，调用未完成部分会抛 NotImplementedError。
不要把整个模板覆盖到 my_ops.py，否则会丢失已经写好的实现。

先记住三个动作
--------------
RMSNorm：像调音台，先调整一行特征的整体强度，再由各通道旋钮分别调节。
RoPE：把每对特征看成一根二维指针，按位置和转速旋转，帮助 Q/K 匹配位置。
Attention：拿问题查资料，先比较索引，再把资料内容按权重混合。
这些只是帮助理解的数据流比喻，不表示真实通道一定对应音量、指针或资料语义。

字母速查
--------
B = Batch Size，一批序列的数量。
T = Sequence Length，序列的位置数，不是 Tensor 的缩写。
E = Embedding/Residual Dimension，每个位置在主路径上的特征数。
H = Number of Attention Heads，头数；本关单头 Attention 固定 H=1。
D = Head Dimension，一个头内部的特征数。
T_q / T_kv = Query / Key-Value Sequence Length，本轮问题数 / 可查资料数。
D_k / D_v = Query-Key / Value Head Dimension，匹配用的特征数 / 内容特征数。
shape 表示张量各轴的长度；dtype 是数值类型，device 是 CPU 或 CUDA 设备。

怎样使用
--------
1. RMSNorm 看 R1-R4，RoPE 看 P1-P7，Attention 看 A1-A5。
2. 先在纸上算下面的小例子，再在 my_ops.py 中实现对应接口。
3. 只新增本关函数，保留已完成的函数；不要导入参考答案代替自己的实现。
4. 在现有 Jupyter Terminal 的 /workspace 目录验收自己的文件：

    python -m learning.local_practice --task norm-rope --device cpu \\
        --candidate learning/my_ops.py
    python -m learning.local_practice --task attention --device cpu \\
        --candidate learning/my_ops.py
    python -m learning.local_practice --task layout --device cpu \\
        --candidate learning/my_ops.py

CPU 通过后，把 cpu 改成 cuda 再跑。三个任务共用 my_ops.py，分别检查各自接口。
未传 --candidate 只是在测参考基线，不是验收你的代码。
宿主机文件与 Jupyter 卷不是自动同步的，运行前确认你正在验收哪份文件。
本模板不是生产算子，也不自动证明 GPU 性能、全部 dtype 支持或学习者已精通。
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm：Root Mean Square Normalization，均方根归一化。

    直觉：一行特征是一排推子，先统一调小/调大，再各自调节
    ------------------------------------------------------
    x 的最后一轴是一行特征；[B,T,E] 中每个位置有 E 个数。
    先为这一行计算一个共同缩放系数，再乘 gamma 的 E 个独立通道权重。
    不是每个数除以自己，也不是把所有 token 混起来求一个平均值。

    手算：暂忽略很小的 eps
    ---------------------
    输入 x                    = [3, 4]
    平方                      = [9, 16]
    mean_square               = (9+16)/2 = 12.5
    RMS                       = sqrt(12.5) = 3.5355...
    inv_rms                   = 1/RMS = 0.2828...
    normalized                = [0.8485..., 1.1314...]
    若 gamma / weight         = [1, 2]
    最终 output               = [0.8485..., 2.2627...]

    共同缩放像“总音量”，weight 像“每通道旋钮”；这不是说特征真的是声音。
    注意第二个值可以大于 1：RMSNorm 不保证输出落在 [0,1] 或 [-1,1]。
    它不先减均值，不能写成 LayerNorm；乘 gamma 后也不保证均方根仍为 1。

    公式：output = weight * x / sqrt(mean(x*x, dim=-1, keepdim=True) + eps)
    eps 在根号里面，用于避免全零输入时除以零；实际代码不能忽略它。
    输入输出同 shape、dtype、device，可用于 [..., width]，不限于三维。
    """

    def __init__(self, width, eps=1e-6):
        super().__init__()
        if width < 1 or eps <= 0:
            raise ValueError("Positive width and epsilon required.")
        self.eps = eps
        # weight 就是可学习的 gamma，shape=[width]，每个通道一个权重。
        # 在 __init__ 中只创建一次，才能让优化器持续更新同一组参数。
        # 验收接口要求参数名为 weight，不要自行改成 scale。
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.weight.numel():
            raise ValueError("The last dimension must match width.")
        input_dtype = x.dtype
        x_fp32 = x.float()

        # input_dtype：保存输入类型，最后恢复；输入并不一定原本就是 FP32。
        # x_fp32：明确选择 FP32 计算，方便低精度输入的归约；device 不变。
        # 这是本练习的精度策略，不是所有 RMSNorm 实现都必须这样写。

        # TODO R1：对 x_fp32 先平方，再沿最后一轴求 mean_square。
        # 例如 x=[B,T,E] -> mean_square=[B,T,1]：每行得到一个“整体强度”。
        # keepdim=True 是保留末尾的 1，后面才能明确地广播回这一行的 E 个数。
        # 不写 dim 会把不同序列/位置混起来；不要先减均值。

        # TODO R2：inv_rms = rsqrt(mean_square + self.eps)。
        # rsqrt(z)=1/sqrt(z)，是共同缩放系数，不是 RMS 本身。

        # TODO R3：normalized = x_fp32 * inv_rms。
        # [B,T,E] * [B,T,1] -> [B,T,E]：一行 E 个数共享一个系数。

        # TODO R4：normalized 乘 self.weight 得到 output，再恢复 input_dtype 并返回。
        # [E] 权重广播到各行，每个通道用自己的 gamma，不改变输入形状。
        # 不要 detach，不要在 forward 内进入 no_grad，不要每次新建 Parameter。
        _ = x_fp32, input_dtype
        raise NotImplementedError("Fill R1-R4 in your own implementation.")


def rope(x, start=0, theta=10_000.0):
    """RoPE：Rotary Position Embedding，旋转位置编码。

    直觉：不是给向量追加一个“位置号码”，而是把里面的特征对转一转
    ----------------------------------------------------------------
    输入/输出都是 [B,H,T,D]，元素个数不变，最后一维 D 必须为正偶数。
    把 D 个数配成 D/2 根二维指针，同一位置的各根指针使用不同转速。
    在本课程的模型路径中，对投影后的 Q/K 分别调用，不对 V 调用。

    本关固定 split-halves：前后两半对应配对，不是相邻两个数配对
    -----------------------------------------------------------
    x=[x0,x1 | x2,x3] -> 两根指针分别是 (x0,x2)、(x1,x3)。
    例如 x=[1,2 | 3,4]，指针是 (1,3) 和 (2,4)。
    为看清规则，假设两对都转 90 度：(a,b) -> (-b,a)：
        (1,3) -> (-3,1)，(2,4) -> (-4,2)
        拼回前后两半：[-3,-4 | 1,2]
    “都转 90 度”只是演示配对/拼回；真实各对角度按下面频率计算。

    theta 不是本次直接旋转的角度
    ---------------------------
    frequency[i] = theta ** (-2*i/D)，i=0..D/2-1
    angle[p,i]   = position[p] * frequency[i]
    D=4、theta=10000 -> 频率 [1,0.01]；绝对位置 2 -> 角度 [2,0.02]。
    这里单位是弧度，torch.sin/cos 不接收“度数”。
    同一根指针转一圈确实会回去；多对不同频率共同表达位置关系，
    但不能据此宣称任意长上下文里绝不会重合或能无误外推。

    start 与当前位置
    ----------------
    T=3、start=5 -> 绝对位置 [5,6,7]，不能每个新块都重新从 0 计数。
    start 改变角度，不改变 T；RoPE 本身也不保存历史 K/V。
    旋转帮助 Q/K 的匹配包含相对位置，causal mask 才负责禁止读取未来。
    """
    if x.ndim != 4:
        raise ValueError("Expected [B,H,T,D].")
    dim = x.shape[-1]
    if dim < 1 or dim % 2 or start < 0 or theta <= 0:
        raise ValueError("Positive even D, nonnegative start, positive theta required.")

    input_dtype = x.dtype
    x_fp32 = x.float()
    half = dim // 2

    # TODO P1：创建特征对索引 i=0..half-1，指定 x.device 和 FP32。
    # dim 是原特征数，half 是指针数；D=4 时就是 i=[0,1]。

    # TODO P2：frequencies[i] = theta ** (-2*i/dim)，shape=[half]。
    # 可理解为各根指针的转速，不是每一对都乘同一个 theta 角度。

    # TODO P3：positions 从 start 到 start+T-1，shape=[T]，放在 x.device。
    # 绝对位置跨块连续；不能把本块下标 [0,1,...] 当成历史位置。

    # TODO P4：angles = positions[:,None] * frequencies[None,:]。
    # [T,1] * [1,half] -> [T,half]：每个位置，给每根指针一个角度。
    # angles 能广播到所有 B/H；同一位置的各 Batch/head 用同一组位置角。

    # TODO P5：left=x_fp32[...,:half]，right=x_fp32[...,half:]。
    # 两者均为 [B,H,T,half]；left[...,i] 与 right[...,i] 是一对。

    # TODO P6：使用同一组 angles 的 cos/sin 旋转两半，不能先覆盖 left 再计算 right。
    # new_left  = left*cos(angles) - right*sin(angles)
    # new_right = left*sin(angles) + right*cos(angles)
    # 指针换方向，长度保持；不是重新生成一组无关特征。

    # TODO P7：沿最后一轴拼回 new_left/new_right，再恢复 input_dtype 并返回。
    # [B,H,T,half] + [B,H,T,half] -> [B,H,T,D]；这里“+”指拼接，不是数值加法。
    # 不原地修改 x，不把 CUDA 张量搬回 CPU，不截断梯度。
    _ = x_fp32, input_dtype, half
    raise NotImplementedError("Fill P1-P7 in your own implementation.")


def causal_attention(q, k, v, start=0):
    """单头因果 Attention：拿问题查资料，不是直接生成文字。

    直觉：问题 -> 比较资料索引 -> 分配查阅权重 -> 混合资料内容
    -------------------------------------------------------
    Q / Query：问题用什么特征去找资料？
    K / Key：各份资料用什么特征供匹配？
    V / Value：匹配后实际取回、参与混合的内容。
    K/V 的位置一一对应：第 j 个索引对应第 j 份内容。
    这是“加权混合”，不是只挑最高分的一份，也不是把 K 直接当内容返回。
    Q/K/V 已由调用方给定，本关不实现投影、RoPE、Cache 存储或输出投影。

    先把 shape 看成一张“问题行 x 资料列”的表
    ---------------------------------------
    q=[B,1,T_q,D_k]，k=[B,1,T_kv,D_k]，v=[B,1,T_kv,D_v]
    T_q 是本轮的问题数，T_kv 是本轮可以查询的资料位置数，头数固定为 1。
    D_k 是比较问题/索引时的共同特征数；D_v 是内容宽度，可以不同于 D_k。

        Q [T_q,D_k] @ K.T [D_k,T_kv]
                    -> scores [T_q,T_kv]
                    -> 屏蔽未来，再沿资料列 softmax
                    -> probabilities [T_q,T_kv]
        probabilities @ V [T_kv,D_v]
                    -> output [T_q,D_v]

    第一遍乘法消去 D_k：每个问题与每份索引做点积求和。
    第二遍乘法消去 T_kv：每个问题把各份内容按权重求和。
    B/head 两轴始终保留；最终完整 shape=[B,1,T_q,D_v]。

    A1 拆开看：k.transpose(-2,-1) 到底转了什么？
    ------------------------------------------
    -1 是最后一条轴 D_k，-2 是倒数第二条轴 T_kv，不是最后两个元素。
    transpose 交换这两条轴，前面的 B/head 不变：
        k：[B,1,T_kv,D_k] -> k转置：[B,1,D_k,T_kv]
    可以记成“原来一份资料占一行，转置后一份资料占一列”。

    省略 B/head，假设有 3 份资料，每份 Key 有 2 个特征：
        原来 K [3,2]              转置后 K [2,3]
             特征0 特征1                 资料0 资料1 资料2
        资料0   1     2             特征0    1     3     5
        资料1   3     4             特征1    2     4     6
        资料2   5     6

    现在让一个 Query=[1,2] 与转置后的各列做点积：
        对资料0：1*1 + 2*2 = 5
        对资料1：1*3 + 2*4 = 11
        对资料2：1*5 + 2*6 = 17
        [1,2] @ [[1,3,5],[2,4,6]] -> [5,11,17]
    原来形状 [1,2] @ [3,2] 的内侧长度 2/3 不匹配；
    转置后是 [1,2] @ [2,3] -> [1,3]，一行里就是对三份资料的分数。
    此处只演示未缩放点积；完整 A1 还要除 sqrt(2)，后面再做 mask/softmax。
    transpose 本身不计算分数、不改值，也不会原地修改原来的 k。
    它不是 reshape：reshape 只按元素顺序重新解释形状，不等于交换行列。
    上面的 K.T 只是二维数学记号；四维代码明确写 k.transpose(-2,-1)。

    多维时不要把每一维都想成“又一张新表”
    ---------------------------------------
    当前练习的分数是 [B,1,T_q,T_kv]：
        B    = 一批里有多少条序列
        1    = head 数，本关固定单头
        T_q  = 表格的行数，问题数
        T_kv = 表格的列数，资料数
    访问一个格子可以写成 scores[b, 0, query_i, key_j]。
    真正决定“未来还是过去”的只有 query_i 和 key_j，
    所以 blocked 只需要 [T_q,T_kv]，不需要为每个 batch/head 重写一份规则。

    广播是“把同一张小地图套到每个大表”，不是一定真的复制内存：

        scores  ：[B, 1, T_q, T_kv]
        blocked ：[   T_q, T_kv]
        对齐后 ：[B, 1, T_q, T_kv]
                   ↑  ↑
                缺的前导轴按 1 处理，1 可扩展到 B 或 head 数

    例如 B=2、H=2、T_q=3、T_kv=7 时：
        scores 形状 = [2,2,3,7]，共有 2*2=4 张“问题×资料”表；
        blocked 形状 = [3,7]，同一张因果规则表套在这 4 张表上。
    第 0 批第 1 个头的第 2 行第 5 列读作 scores[0,1,2,5]；
    它对应的规则只看 blocked[2,5]，与 batch/head 无关。
    如果每个 batch 有不同 padding，mask 才需要扩大为 [B,1,T_q,T_kv]；
    如果每个 head 也不同，才会使用 [B,H,T_q,T_kv]。

    代码中可以显式写出“补两个前导轴”：
        blocked_4d = blocked[None, None, :, :]  # [1,1,T_q,T_kv]
        scores.masked_fill(blocked_4d, -torch.inf)
    但 masked_fill 会自动广播，所以直接传 [T_q,T_kv] 就够了。
    记忆顺序：先看最后两轴（问题×资料），再看前面的 batch/head 复制层。

    四种模态的同一套“查资料”故事
    --------------------------------
    下面只是把 token 换成不同模态的“基本单位”；Attention 公式没有变。

    1) 文本：一个个 token 排成队
       “猫  坐  在  垫子  上”
       Query = 当前 token 想找上下文；Key = 每个 token 的索引特征；
       Value = 每个 token 真正提供的内容。T 是 token 数。
       自回归语言模型通常只能看左边和自己，所以会使用未来 Mask。

    2) 图片：把图片切成 patch 小方块
       [patch0][patch1][patch2]...
       Query = 某个方块要找相关区域；Key = 每个方块的视觉索引；
       Value = 方块里的颜色/纹理/形状特征。
       例如图片切成 14×14 个 patch，序列长度 N=196；
       Attention map 可以是 [B,H,196,196]，N 扮演文本 T。
       普通图像理解多使用双向注意力，不一定遮挡“未来 patch”；
       自回归图像生成才可能按 patch 顺序遮挡未来。

    3) 音频：把声音切成连续时间帧
       |帧0|帧1|帧2|帧3|...
       Query = 当前帧想参考哪些声音片段；Key = 各帧的匹配特征；
       Value = 各帧的音色、音素或声学内容。T 是帧数，不是采样点总数。
       流式语音通常只能读过去帧，所以 start、因果 Mask 和 KV Cache 很重要。

    4) 视频：时间帧里再切空间 patch
       第0帧：[patch0][patch1]...     第1帧：[patch0][patch1]...
       一个视频 token 可能是某一帧的小方块，也可能是 tubelet。
       若有 F 个视频帧、每帧 P_h×P_w 个 patch，序列长度 N=F×P_h×P_w；
       Attention map 是 [B,H,N_q,N_kv]。建议用 F 表示帧数、N 表示时空 token 数，
       不要把公式中的 T 误读成 Tensor 或只代表视频帧。
       时间因果 Mask 可以禁止未来帧；空间是否遮挡取决于模型设计。

    四种模态都可用同一句话检查 shape：
        最后两轴 = “当前问题单位 × 可查询资料单位”
        更前面的轴 = batch、head，或其他并行分组
    token、patch、音频帧、视频时空 patch 只是“资料卡片”的不同载体；
    Value 不是原始文字、像素、声音或视频文件，而是模型内部的特征向量。

    stride：索引走一步，底层storage要跳多少个元素？
    ------------------------------------------------
    shape 是“有多少箱/排/格”，stride 是“换箱/换排/换格时跳多远”。
    其他索引不变时，第 i 条轴的索引加 1，storage元素编号增加 stride[i]。
    它的单位是元素，不是字节，也不等于这条轴本身的长度。

    默认连续行优先布局，shape=[2,3,4]：2箱，每箱3排，每排4格。
        换一格：跳1个数                -> stride[2]=1
        换一排：跳一整排4个数         -> stride[1]=4
        换一箱：跳一整箱3*4=12个数    -> stride[0]=12
    因此 stride=[12,4,1]；从右向左乘，是因为要跳过右侧的一整个块。
    这只是默认连续布局的规则，转置/切片后不能仅凭shape照算。
    stride不是模型学习的参数，通常由PyTorch在创建/转换布局时维护，.stride()读取实际结果。
    大小为1的轴也不靠shape唯一决定stride；以实际 .stride() 为准。

    对普通strided Tensor，元素寻址公式是：
        storage_index = storage_offset + sum(index[i] * stride[i])
        byte_address = storage_base_address + storage_index * element_size()
    shape=[2,3,4]、stride=[12,4,1]、offset=0时：
        x[1,2,3] -> 1*12 + 2*4 + 3*1 = storage第23号元素。
    用FP32时一个元素4字节，第23号元素距离storage基址92字节。

    用标签数验证多头拆合，不用猜：
        base = torch.arange(24, dtype=torch.float32)
        x = base.reshape(2,3,4)          # [B,T,H*D], stride=(12,4,1)
        split = x.view(2,3,2,2)          # [B,T,H,D], stride=(12,4,2,1)
        heads = split.transpose(1,2)     # [B,H,T,D], stride=(12,2,4,1)
    拆头把4维解释成2个head、每头2维；转置只交换两轴的shape/stride。
    heads[1,1,2,0] -> 1*12+1*2+2*4+0*1=22，值就是base[22]。
    多维不神秘：每条轴只贡献“索引乘自己的跳距”，最后相加。

    切片说明为什么还需要storage_offset：
        tail = x[:,1:,:]                # shape=(2,2,4), stride=(12,4,1), offset=4
    第一格值是base[4]；换batch仍跳12格，不是新shape右侧乘积8格。
    tail和x共享storage，但tail.data_ptr()比x.data_ptr()晚4个FP32元素，即16字节。
    data_ptr()指第一个逻辑元素，不一定是storage起始地址；
    在这些非空普通张量里用 untyped_storage().data_ptr() 对比storage基址。

    view不复制，要求新旧形状/stride兼容；非连续不等于一概不能view。
    reshape兼容时返回view，不兼容时复制；不要假定必然共享storage。
    contiguous按当前逻辑顺序生成连续布局，已符合格式时无需复制。
    正确合头：heads.transpose(1,2).reshape(2,3,4)。
    直接heads.reshape(2,3,4)会混错时间和head；先contiguous再直接reshape也不会修好语义。
    切片base[::2]的stride为2；expand还可产生stride为0的轴，并非最后一个stride总是1。
    完整货架图、Jupyter逐步实验与官方API见 docs/DAY10_MULTIHEAD.md 第2.1节。
    本段只解释布局，不新增已完成算子；你的实现仍只放在my_ops.py。

    小数字：两份资料的内容是 10 和 20
    --------------------------------
    当前 Query 在绝对位置 1，K/V 位置是 0、1，都允许读取。
    Q=[0,sqrt(2)*ln(3)]，K0=[1,0]，K1=[0,1]，D_k=2。
    点积后除 sqrt(2) 得到 scores=[0,ln(3)]。
    exp 后是 [1,3]，除以总和 4 -> probabilities=[1/4,3/4]。
    混合内容：1/4*10 + 3/4*20 = 17.5，不是返回分数或挑选 20。
    ln 是自然对数；这个例子只说明数值，不是在解释真实词语语义。

    因果规则：只能看过去和当前位置，不能偷看未来
    -------------------------------------------
    位置 0 的问题只看位置 0；位置 1 看 0、1；位置 2 看 0、1、2。
    分数表每行是一个问题，列是资料，未来资料在 softmax 前填 -inf。
    exp(-inf)=0，所以未来权重为零；填 0 不行，因为 exp(0)=1。

    已有 4 个历史位置，新来 3 个 Query 时：
        start=4，T_q=3，T_kv=7
        问题行的绝对位置 = [4,5,6]
        资料列的位置     = [0,1,2,3,4,5,6]
        三行分别允许前 5、6、7 份资料。
    start 不是切掉前 4 份资料，而是告诉每行“你处于哪一个绝对位置”。
    Decode 只来一个问题时也可能查很多历史资料：T_q=1，T_kv 不一定为 1。
    """
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

    # TODO A1：匹配并缩放，得到 scores=[B,1,T_q,T_kv]。
    # q @ k.transpose(-2,-1)，再除 sqrt(D_k)，D_k=q.shape[-1]。
    # transpose 交换末尾 T_kv/D_k：[B,1,T_kv,D_k] -> [B,1,D_k,T_kv]。
    # 让每份 Key 从一行变成一列，Q 才能逐列打分；不是交换两个元素。
    # 像给每个“问题-资料”组合打分；缩放避免随特征维增加，典型点积幅度过大。
    # 这不是除以资料份数 T_kv；分数此时还不是概率。

    # TODO A2：在 q.device 创建 query_positions 与 key_positions。
    # 问题行：start..start+T_q-1，资料列：0..T_kv-1。
    # [:,None] 让问题编号竖着排，[None,:] 让资料编号横着排。
    # [T_q,1] 和 [1,T_kv] 可以逐行逐列广播比较。

    # TODO A3：blocked[i,j] = key_positions[j] > query_positions[i]。
    # blocked=[T_q,T_kv]；True 表示“这份资料来自未来，禁止看”。
    # scores=[B,1,T_q,T_kv]，PyTorch 从右向左对齐，把同一张 [T_q,T_kv]
    # 规则表广播到每个 batch/head；这通常是逻辑扩展，不是手动复制数据。
    # 用 masked_fill 把这些 scores 改为 -torch.inf；不会修改 q/k/v。
    # 注意 SDPA 的 bool attn_mask 恰好相反：True 表示允许。

    # TODO A4：probabilities = softmax(scores, dim=-1)，shape 不变。
    # 沿最后一轴的资料列归一化：每个问题单独分配一份总和为 1 的权重。
    # 不沿问题轴归一化；不要自己裸算 exp(大分数)，torch.softmax 会做稳定处理。
    # 本练习不加 dropout。

    # TODO A5：output = probabilities @ v，返回 [B,1,T_q,D_v]。
    # 一行权重乘所有对应 Value，再按资料位置求和，每个问题得到一份混合内容。
    # 不返回 probabilities，不做 argmax，也不用 K 代替 V。
    # 保留梯度；不 detach，不进 no_grad，不原地修改输入，不导入参考实现代做。
    # 完成后按模板开头命令验收 my_ops.py，先 CPU，再 CUDA。
    raise NotImplementedError("Fill A1-A5 in your own implementation.")


def split_heads(x, heads, head_dim):
    """P1-03：把 [B,T,H×D] 拆成 [B,H,T,D]，只做布局操作。"""
    if x.ndim != 3:
        raise ValueError("Expected [B,T,H*D].")
    if (not isinstance(heads, int) or isinstance(heads, bool) or heads < 1
            or not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1):
        raise ValueError("Positive integer heads and head_dim required.")
    if x.shape[-1] != heads * head_dim:
        raise ValueError("Last dimension must equal heads * head_dim.")
    if x.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    if not x.is_contiguous():
        raise ValueError("split_heads requires contiguous input.")

    # TODO S1：view 成 [B,T,heads,head_dim]。
    # 只是重新解释连续存储，不做 Linear 投影，不改变数值。

    # TODO S2：交换 T 和 heads 轴，得到 [B,heads,T,head_dim]。
    # 不要在这里 contiguous；先观察转置后的 stride。
    raise NotImplementedError("Fill S1-S2 in your own implementation.")


def merge_heads(heads):
    """P1-03：把 [B,H,T,D] 合回 [B,T,H×D]，只做布局操作。"""
    if heads.ndim != 4:
        raise ValueError("Expected [B,H,T,D].")
    if min(*heads.shape) < 1:
        raise ValueError("Nonempty [B,H,T,D] input required.")
    if heads.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")

    # TODO M1：先把 head 和 T 换回 [B,T,H,D]。
    # 只有同一时间位置的多个 head 相邻，才可以安全合并最后两轴。

    # TODO M2：reshape 最后 H、D 两轴 -> [B,T,H×D]。
    # 直接对 [B,H,T,D] reshape 是故意保留的错误实验。
    raise NotImplementedError("Fill M1-M2 in your own implementation.")


def expand_kv_heads(kv, query_heads):
    """P1-04：把紧凑 KV [B,H_kv,T,D] 扩展成 [B,H_q,T,D]。"""
    if kv.ndim != 4:
        raise ValueError("Expected [B,H_kv,T,D].")
    if min(*kv.shape) < 1:
        raise ValueError("Nonempty [B,H_kv,T,D] input required.")
    if not isinstance(query_heads, int) or isinstance(query_heads, bool) or query_heads < 1:
        raise ValueError("Positive integer query_heads required.")
    if kv.dtype != torch.float32:
        raise ValueError("This task requires FP32 input.")
    if query_heads % kv.shape[1] != 0:
        raise ValueError("query_heads must be divisible by kv_heads.")

    # TODO G1：group_size = query_heads // kv.shape[1]。
    # 一个紧凑 KV head 服务 group_size 个 Query heads。

    # TODO G2：expanded = kv.repeat_interleave(group_size, dim=1)。
    # 输出 [B,query_heads,T,D]；不修改紧凑 kv。
    # 扩展是计算阶段的临时对齐，不代表 Cache 应保存扩展副本。
    raise NotImplementedError("Fill G1-G2 in your own implementation.")
