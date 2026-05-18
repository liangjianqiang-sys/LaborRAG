---
description: 开发进度追踪 - 记录当前完成状态，对话丢失后快速恢复
---

# LaborRAG 开发进度

## 当前版本: V3 CRAG高级版
## 当前阶段: V3 全部代码完成，端到端测试已通过，准备V4
## 最后更新: 2026-05-08

---

## V1 进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| 阶段1: 项目初始化与后端骨架 | ✅ 已完成 | 目录结构、requirements、config、schemas、.env.example |
| 阶段2: RAG核心引擎 | ✅ 已完成 | embeddings/document_loader/vectorstore/retriever/generator/rag_engine |
| 阶段3: API接口 | ✅ 已完成 | routes.py(6个API) + main.py(FastAPI入口+lifespan+CORS) |
| 阶段4: 劳动法数据准备 | ✅ 已完成 | 6部法律UTF-8文本已放入data/labor_laws/ |
| 阶段5: 前端界面 | ✅ 已完成 | React+TS+TailwindCSS |
| 阶段6: 联调与测试 | ✅ 已完成 | 修复FAISS中文路径+API兼容问题，问答测试通过 |

## V2 进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| 阶段1: Hybrid Search | ✅ 已完成 | bm25.py + hybrid.py + RRF融合 |
| 阶段2: Reranker重排序 | ✅ 已完成 | reranked.py + BGE-Reranker-v2-m3 |
| 阶段3: 法条结构化切分 | ✅ 已完成 | text_splitter.py + LawArticleSplitter |
| 阶段4: RAGAS评估体系 | ✅ 已完成 | eval_dataset + eval_runner + eval_report + API |
| 阶段5: 对比实验 | ⬜ 待测试 | 需运行三组对比(vector/hybrid/reranked) |
| 阶段6: 前端更新 | ✅ 已完成 | EvalPanel + 评估API |
| 阶段7: 前端重构 | ✅ 已完成 | 多页面布局(问答/知识库/评估/设置) + React Router |
| 阶段8: Prompt强化 | ✅ 已完成 | 严格忠实于检索内容，禁止编造 |
| 阶段9: 参数调优 | ✅ 已完成 | CHUNK_SIZE=800, TOP_K=8, SCORE_THRESHOLD=0.2 |
| 阶段10: 评估修复 | ✅ 已完成 | ragas 0.4.x兼容(LLM+Embedding传入) |
| 阶段11: 评估完善 | 🔜 V4后 | 需增加纠错成功率/Agent准确率等指标 |

## V3 进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| 阶段1: CRAG框架 | ✅ 已完成 | langgraph依赖 + crag_graph.py + RAG_MODE配置 + RAGEngine集成 |
| 阶段2: 检索质量评估 | ✅ 已完成 | _grade_documents节点 + 条件路由 |
| 阶段3: Query改写 | ✅ 已完成 | _rewrite_query节点 + 上下文感知改写 |
| 阶段4: 回答质量评估 | ✅ 已完成 | _grade_answer节点 + 忠实度判断 |
| 阶段5: 多轮对话 | ✅ 已完成 | conversation.py + conversation_id + 上下文注入CRAG |
| 阶段6: CRAG流程可视化 | ✅ 已完成 | ChatResponse.crag_steps + 前端CragStepsPanel |
| 阶段7: 对比实验代码 | ✅ 已完成 | run_v3_comparison.py + 前端评估页更新 |
| 端到端测试 | ✅ 已完成 | CRAG工作流实际测试通过 |
| 对比实验运行 | ⏸ 延后V4后 | 需V4完成后统一运行四版对比 |

## V4 进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| 全部 | ⬜ 未开始 | 需先完成V3 |

## 技术决策记录
- 2026-04-23: 确定使用LangChain 1.0 + LCEL + LangGraph路线
- 2026-04-23: 确定渐进式开发：V1→V2→V3
- 2026-04-23: Embedding选用BGE-M3，LLM选用DeepSeek/GLM-4 API
- 2026-04-25: LLM改用百炼qwen3.6-flash，Embedding用本地BGE-M3
- 2026-04-25: FAISS不支持中文路径，向量库改存D:/LaborRAG_data/vector_store
- 2026-04-25: 修复FAISS旧版similarity_search_with_relevance_score不存在问题
- 2026-04-28: V2混合检索采用BM25+jieba中文分词+RRF融合方案
- 2026-04-28: V2重排序选用BGE-Reranker-v2-m3 Cross-Encoder
- 2026-04-28: V2切分策略新增law_article法条结构化切分，默认启用
- 2026-04-28: V2评估采用RAGAS框架，4指标：faithfulness/answer_relevancy/context_precision/context_recall
- 2026-04-28: RAG引擎根据RETRIEVER_TYPE自动选择检索器，支持vector/hybrid/reranked三种策略
- 2026-04-28: 确定V4多Agent架构：Router/Retriever/Calculator/Validator四个Agent协作
- 2026-04-28: 前端重构为多页面布局(React Router)，4页面：问答/知识库/评估/设置
- 2026-04-28: Prompt强化，忠实度从0.37提升至0.95
- 2026-04-28: 参数调优CHUNK_SIZE=800/TOP_K=8，召回率从0.5提升至1.0
- 2026-04-28: 修复ragas 0.4.x兼容性(EvaluationResult对象+LLM/Embedding传入)
- 2026-04-28: 评估体系待V4后完善，需增加纠错成功率/Router准确率/计算准确率/对话连贯性等指标
