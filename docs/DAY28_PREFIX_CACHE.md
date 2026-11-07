# Day 28：Prefix Cache，借旧草稿但不能把原稿改脏

前置：[Day 12请求Cache](DAY12_REQUEST_CACHE.md)、[Day 25所有权](DAY25_OWNERSHIP_OOM.md)。

## 1. 图解与数字例子

公共系统提示像一份已计算的草稿，不必每次从头算：

```text
前缀[1,2,3] → 首次prefill → 缓存KV与末位置logits
再次[1,2,3] → hit → 拿一份隔离副本继续输入4
原前缀仍长3，当前分支变长4
```

本实验只识别**完整精确token前缀**。
`[1,2,3,4]`不会自动查找已存的`[1,2,3]`最长部分命中。
“文字意思相近”不是token前缀相同；不同tokenizer也不能直接共享。

## 2. 概念与计算契约

PrefixStore绑定一个model，key为token ID tuple，value为逐层Cache及logits快照。
miss后存clone，hit后返回clone；这是隔离正确性方案，
不是共享物理页、引用计数、COW或内存节省算法。

同模型权重/adapter、位置/RoPE配置等才允许复用。
当前key没有完整包含这些身份，权重或配置改变应显式clear。
只存末logits最多能选首个续写token，后续Attention仍需要历史KV。
没有自动驱逐/容量管理，不适合无上限长期服务。

## 3. Jupyter与源码路线

打开 [Day 28 Notebook](../learning/day28_course.ipynb)，看first_hit=False、
second_hit=True、continuation_alignment和isolation。
定位PrefixStore.prefill、Cache.clone、clear。
先预测追加suffix后当前长度，再读取同前缀验证原snapshot仍长3。

自己改前缀一个token应miss；对model参数变动先clear，不期待当前实现自动识别。
注意report中“no weight invalidation”表示无自动失效，不能误读成无需失效处理。

## 4. 调试与优化顺序

1. 先验证token身份与同权重full continuation对齐。
2. 原快照不被续写污染；检查clone不是浅拷贝。
3. 设计模型/adapter/位置/精度/租户等缓存身份，不把tuple当完整生产key。
4. 再测查表、复制、预填充节省与容量占用，命中率高不保证净收益。
5. COW、共享页、驱逐各自新增所有权契约与失败测试，当前未实现的不宣称完成。

## 5. 自测与参考答案

**问：为什么只缓存logits不够？**
答：继续生成时新Q还要读历史KV，logits不能重建那些状态。

**问：clone为什么可能不省内存？**
答：它复制快照与返回状态，优先保证隔离，不等于物理页共享。

**问：权重变了仍能命中相同token key吗？**
答：当前代码可能照样命中，所以必须主动clear；生产应做更完整身份与失效策略。

**问：是否实现自动最长前缀匹配？**
答：没有，本例只匹配完整精确前缀。

## 6. 验收与资料

- [ ] 保存miss/hit、续写对齐及原snapshot不变证据。
- [ ] 列出必须失效条件与未实现的容量/共享机制。

深入：[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；
[本仓库PrefixStore](../learning/mini_vllm/engine.py)。
下一课：[Day 29](DAY29_REPRODUCIBLE_REPORT.md)。
