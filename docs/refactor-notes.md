# 架构收敛记录（阶段 2）

> 从 README 移出的重构记录。README「工程化实践」末尾保留两句摘要。

---

## 重构明细

除 bug 之外，这一轮还做了两处「消除重复 / 恢复依赖方向」的重构，全部经基线验证 Δ = 0：

| 项 | 改动 | 为什么值得 |
|------|------|---------|
| **检索器构造收敛到门面** | `app/retrieval/__init__.py` 暴露 `build_retriever` / `build_hybrid_retriever` / `build_reranked_retriever`；`rag_engine` 不再直接依赖 vector/hybrid/reranked 具体类 | 「VectorRetriever + HybridRetriever(权重取自 settings)」原本在 `rag_engine` 里**重复 3 次**（hybrid 分支 / reranked 分支 / eval_retriever 属性）。改一处忘一处，就会让**生产检索与评估检索用不同权重**，评估结论对不上生产行为，且无任何报错 |
| **file_lock 下沉到 utils** | 新增 `app/utils/files.py`；`knowledge/store.py` 不再反向 import `app/evaluation/utils.py` | 依赖方向原本是倒的（底层 → evaluation），核心层无法脱离评估层存在。现在恢复为 `utils ← evaluation` |
| **架构规则化** | `test_layering.py` 用 AST 扫描 `app/**`，禁止低层 import 高层（**函数内的延迟导入同样能扫到** —— 倒置常藏在这里） | 把「分层」从口头约定变成会变红的规则。阴性对照精准报出 `vectorstore.py:10 → app.evaluation.utils` |
| **领域规则库拆包** | 757 行单文件（其中 471 行是纯数据表）拆为 `domain_boost/` 包：`relations` / `concepts` / `source_map` 纯数据 + `lookup` 逻辑，`__init__.py` 门面再导出全部名字 | 62% 是常量表且与逻辑交错 —— 加一条伴生关系要滚过 200 行 `CONCEPT_ARTICLE_MAP`。**调用点一行未改**（门面透明接管），拆分前后四张数据表与六个函数源码经比对**逐字节一致** |
| **法条编号原语收敛** | `app/utils/law_refs.py` 统一承载「法条编号正则」与「中文数字 ↔ 阿拉伯数字」两类原语；`constants` / `helpers` / `domain_boost` 改为再导出 | 正则曾散落 7 处（2 处带病），中文数字解析曾散落 6 处（语义互不一致，Bug 6 即由此而来）。现由**身份断言**（`is` 同一对象）守卫，再抄一份立刻变红 |
