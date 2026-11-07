# Day 8学习记录：RMSNorm与RoPE独立实现

开始日期：2026-11-07（America/New_York）。
状态：**学习中**。用户要求提交Day 7后继续下一天。
当前学习位置见 [进度记录](LEARNING_PROGRESS.md)。

## 今日核心问题

```text
RMSNorm如何稳定一行特征的数值？
RoPE如何在不改变shape的情况下给Q/K加入位置？
```

入口：

- [Day 8教程](DAY08_RMSNORM_ROPE.md)
- [Day 8基础Notebook](../learning/day08_course.ipynb)
- [Day 8逐步Notebook](../learning/day08_rmsnorm_rope.ipynb)
- [Mini Transformer参考实现](../learning/mini_transformer/model.py)

## 学习顺序

| 步骤 | 行动 | 个人完成状态 |
| --- | --- | --- |
| 1 | 手算 `[3,4]` 的RMS、RMSNorm；说明没有减均值 | 待记录 |
| 2 | 观察全零、常数、大值、小值的RMSNorm行为 | 待记录 |
| 3 | 对一个二维特征对画/说出位置旋转 | 待记录 |
| 4 | 改RoPE offset=0/5，检查shape不变、数值改变 | 待记录 |
| 5 | 检查偶数D、非整齐T、FP32/BF16误差 | 待记录 |
| 6 | 解释RoPE与causal mask、KV Cache的职责区别 | 待记录 |
| 7 | 将参考代码遮住，另建自己的RMSNorm/RoPE实现，再与oracle对齐 | 待记录 |

FP64公式只作高精度oracle；现有Mini RMSNorm内部使用FP32，不因`.double()`改变。
RoPE固定split-halves布局，4维配对为(0,2)/(1,3)，theta_base=10000。
将offset和布局明确后再比较数值，不用放宽阈值来掩盖配对错误。

## 验收模板

```text
我手算的RMSNorm：
我对RMSNorm与LayerNorm的区别：
我观察到的RoPE shape：
offset改变后的误差/数值现象：
我做过的失败实验：
仍不理解：
是否进入Day 9：
```

个人运行与口述结果：**尚未登记**。
