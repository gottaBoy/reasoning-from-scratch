# Day 25：内存所有权与OOM，借不够就不要借一半

前置：[Day 23](DAY23_PAGED_ADDRESSING.md)、[Day 24](DAY24_SCHEDULER_FAIRNESS.md)。

## 1. 图解与数字例子

把free list看成可借抽屉账本：

```text
free=[3]，新请求要2块
    ↓ 先检查
拒绝：free仍是[3]，不登记半个请求，不动已有A/C的数据
```

请求B完成后，把它的块号放回free，旧B借阅证active=False。
C可以重用这些物理块，但C有效长度从0开始，
不能因为数组还残留旧数值就把它当成自己的历史。

## 2. 概念与计算契约

`reserve`先算ceil(capacity/block_size)，检查可用量后才取页；
这是该分配入口的原子拒绝契约，不是任意运行时故障的数据库事务。
`release`移除request、让handle失效、页归还队列。
**回收页所有权不等于释放整个pool张量，也不等于安全擦除旧数据。**

本引擎按最大预算提前预留，简化运行中增长；没有外部取消、抢占/offload、
引用计数/COW或多线程安全保证。CUDA异步还需确保旧读者完成后才能安全复用。

## 3. Jupyter与源码路线

打开 [Day 25 Notebook](../learning/day25_course.ipynb)，看atomic_oom与reclaimed_blocks。
本统一实验验证分配/回收元数据，不包含真实KV数值测试；
更细步骤运行 [Day 7生命周期Notebook](../learning/day07_kv_lifecycle.ipynb)。
定位reserve检查、active检查、release以及Engine.run的finally。

自己保存OOM前free/requests/storage快照，触发失败后逐项比较。
对旧handle读取应报错；不要直接绕过接口读取原始storage来声称隔离失效或已安全隔离。
模拟forward异常，确认登记的页都归还，不隐藏原异常。

## 4. 调试与优化顺序

1. 每个页唯一活跃所有者，free与live不交叠，总页数守恒。
2. 错误输入和不足容量先拒绝，不能取完一部分页才报错。
3. 完成/异常出口统一清理，注意重复release和遗留引用。
4. 分别观察free数、pool大小、allocated/reserved、系统统计。
5. 先查泄漏与过度预留，再调容量/页宽；逐token empty_cache不是修复所有权的办法。

## 5. 自测与参考答案

**问：请求结束显存没降一定泄漏吗？**
答：不一定，pool仍存在；先看页是否归还、live请求是否清掉。

**问：OOM前后最少比较什么？**
答：free、request表、已有状态及必要的实际storage，不只异常类型。

**问：有finally就等于生产取消完整了吗？**
答：不是，还需取消协议、pending工作、异步stream/引用生命周期与跨进程故障处理。

**问：旧值没清零可直接交给新请求吗？**
答：只有有效长度/写后读/所有权都正确才不会逻辑读旧值；这不保证安全擦除与多租户隔离。

## 6. 验收与资料

- [ ] 保留OOM原子拒绝和旧handle失效证据。
- [ ] 解释请求生命周期、页池生命周期与allocator缓存三者。

深入：[PyTorch CUDA内存管理](https://docs.pytorch.org/docs/2.10/notes/cuda.html)；
[抽屉图解](DAY07_KV_LIFECYCLE.md)。
下一课：[Day 26](DAY26_BATCHED_ENGINE.md)。
