---
description: 开发进度追踪 - 记录当前完成状态，对话丢失后快速恢复
---

# LaborRAG 开发进度

## 当前版本: V4 Agentic RAG多Agent版
## 当前阶段: V4 核心代码完成 + 评估系统完善，待端到端测试
## 最后更新: 2026-05-23

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
| 阶段4: RAGAS评估体系 | ✅ 已完成 | eval_dataset(24题+question_type) + eval_runner(支持rag_mode/分类型统计) + eval_report + API |
| 阶段5: 对比实验 | ⬜ 待测试 | 需运行四组对比(vector/reranked/crag/agent) |
| 阶段6: 前端更新 | ✅ 已完成 | EvalPanel + 评估API |
| 阶段7: 前端重构 | ✅ 已完成 | 多页面布局(问答/知识库/评估/设置) + React Router |
| 阶段8: Prompt强化 | ✅ 已完成 | 三个生成器(simple/crag/agent)统一4条严格规则，Validator增加"法条是否在资料中"检查 |
| 阶段9: 参数调优 | ✅ 已完成 | CHUNK_SIZE=800, TOP_K=8, SCORE_THRESHOLD=0.2 |
| 阶段10: 评估修复 | ✅ 已完成 | ragas 0.4.x兼容(LLM+Embedding传入) |
| 阶段11: 评估完善 | ✅ 已完成 | question_type字段+分类型统计+每题详情+异步后台执行+前端轮询+V4实验配置 |

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
| 对比实验代码 | ✅ 已完成 | run_v3_comparison.py支持V1-V4四组实验 |
| 对比实验运行 | ⏸ 延后 | 需统一运行四版对比 |

## V3 升级优化进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| 层次化父子分块 | ✅ 已完成 | LawArticleSplitter父块+子块+parent_id，VectorStoreManager子块索引+parent_store持久化 |
| 父子分块检索提升 | ✅ 已完成 | promote_children_to_parents()子块→父块替换+去重+最高分排序 |
| 元数据过滤 | ✅ 已完成 | extract_article_ref()法条编号提取，FAISS filter+BM25后置过滤 |
| HyDE查询增强 | ✅ 已完成 | 替换CRAG rewrite_query为HyDE，短查询(≤15字)条件触发 |
| 上下文压缩 | ✅ 已完成 | retriever/compressor.py EmbeddingsFilter，余弦相似度<0.5丢弃，CRAG新增compress节点 |
| Adaptive RAG | ✅ 已完成 | CRAGGraph→AdaptiveRAGGraph，复杂度分类(simple/medium/complex)+三路路由 |
| Bug修复 | ✅ 已完成 | import os移到顶部、score_threshold=0.0 falsy修复、Hybrid RRF去重修复 |
| 参数调优 | ✅ 已完成 | VECTOR_WEIGHT=0.7/BM25_WEIGHT=0.3/SCORE_THRESHOLD=0.3/COMPRESSION_THRESHOLD=0.5 |

## V4 进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| AgenticRAGGraph主框架 | ✅ 已完成 | agent_graph.py, 5个Agent节点+LangGraph状态图 |
| Router Agent | ✅ 已完成 | LLM意图分类(retrieve/calculate/compare) |
| Calculator Agent | ✅ 已完成 | 5种劳动法计算+缺参时返回公式框架而非放弃 |
| Comparator Agent | ✅ 已完成 | 双组检索+LLM结构化对比 |
| Validator Agent | ✅ 已完成 | 合法性+幻觉验证(含"法条是否在资料中"检查)，未通过追加警告 |
| rag_engine集成 | ✅ 已完成 | RAG_MODE=agent，带降级兜底 |
| config/schemas更新 | ✅ 已完成 | RAG_MODE支持simple|crag|agent |
| 端到端测试 | ⬜ 待测试 | 需实际运行验证 |
| 代码重构 | ✅ 已完成 | _first_match/_parse_cn_salary提取, _mean_scores合并重复逻辑, routes简化 |
| Prompt统一强化 | ✅ 已完成 | simple/crag/agent三路生成prompt统一4条规则, Validator增加资料内法条检查 |
| auto_calculate改进 | ✅ 已完成 | 缺参时返回公式框架+缺失参数提示，不再直接放弃 |
| 评估异步化 | ✅ 已完成 | 后台线程执行+GET /evaluation/status轮询+前端轮询模式+页面恢复状态 |
| 对比实验运行 | ⏸ 延后 | 需V1-V4四版统一对比 |

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
- 2026-05-19: 实现层次化父子分块(PARENT_CHILD_ENABLED)，子块索引+父块存储+promote提升
- 2026-05-19: 实现元数据过滤(extract_article_ref)，FAISS filter+BM25后置过滤，法条编号精准查询
- 2026-05-19: HyDE替换CRAG rewrite_query，短查询(≤15字)条件触发，成本不变精度提升
- 2026-05-19: 修复score_threshold=0.0 falsy bug，Hybrid RRF子块级别融合去重
- 2026-05-19: 参数调优VECTOR_WEIGHT=0.7/BM25_WEIGHT=0.3/SCORE_THRESHOLD=0.3
- 2026-05-20: 实现上下文压缩EmbeddingsFilter(compressor.py)，CRAG retrieve→compress→grade_documents
- 2026-05-20: 实现Adaptive RAG，CRAGGraph→AdaptiveRAGGraph，复杂度分类+三路路由(simple/medium/complex)
- 2026-05-20: config.py新增COMPRESSION_THRESHOLD=0.5，RAG_MODE注释更新crag=Adaptive RAG
- 2026-05-20: 实现V4 Agentic RAG，agent_graph.py五Agent协作(Router/Retriever/Calculator/Comparator/Validator)
- 2026-05-20: Calculator Agent支持5种劳动法计算(加班费/经济补偿金/赔偿金/双倍工资/年休假工资)
- 2026-05-20: rag_engine.py集成RAG_MODE=agent，带降级兜底，config RAG_MODE支持simple|crag|agent
- 2026-05-23: 代码重构：agent_graph提取_first_match/_parse_cn_salary简化参数提取；eval_runner提取_mean_scores合并重复指标解析；routes用**result解包简化返回
- 2026-05-23: Prompt统一强化：simple/crag/agent三路生成prompt统一4条严格规则(禁止编造/必须出现在资料中)；VALIDATOR_PROMPT新增"法条是否在参考资料中"检查维度
- 2026-05-23: auto_calculate改进：缺参时返回公式框架+缺失参数提示，不再直接返回"未能识别"
- 2026-05-23: 评估系统完善：eval_dataset增加question_type字段(24题)；eval_runner支持rag_mode/question_type参数+分类型统计+每题详情；routes评估异步化(后台线程+状态轮询)；前端轮询模式+页面恢复状态；run_v3_comparison.py添加V4实验配置
