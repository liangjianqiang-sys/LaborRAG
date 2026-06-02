# Agentic RAG 分层架构设计

## 阶段一：依赖盘点与状态契约

### 1. 外部依赖（import）

| 来源 | 引用项 | 用途 |
|------|--------|------|
| Python 标准库 | `re`, `math` | 正则、数学计算 |
| typing | `List`, `Tuple`, `Literal` | 类型注解 |
| typing_extensions | `TypedDict` | AgentState 定义 |
| langchain_core.documents | `Document` | 文档数据结构 |
| langchain_core.prompts | `ChatPromptTemplate` | Prompt 模板 |
| langchain_openai | `ChatOpenAI` | LLM 客户端 |
| langgraph.graph | `StateGraph`, `END` | 工作流图 |
| app.config | `settings` | 全局配置 |
| app.core.generator.prompts | `ANSWER_STYLE_RULES`, `RETRIEVAL_EVALUATION_PROMPT`, `MULTI_QUERY_EXPANSION_PROMPT`, `HYDE_PROMPT` | Prompt 模板 |
| app.core.generator.guardrails | `AnswerGuardrails` | 护栏 |
| app.core.retriever.reranked | `RerankedRetriever` | isinstance 判断 |
| app.core.retriever.law_graph | `inject_related_articles`, `_extract_law_article_from_doc`, `concept_lookup`, `get_related_articles`, `_article_to_cn`, `_exact_lookup_from_parent_store`, `_LAW_SOURCE_MAP`, `_CN_NUM` | 知识图谱 |

### 2. 内部共享资源（类属性）

| 资源 | 类型 | 消费节点 |
|------|------|---------|
| `self.retriever` | BaseRetriever | retrieve_docs, retrieve_for_compare, calculate, simple_generate |
| `self.grader_llm` | ChatOpenAI | rewrite_query, route_intent, expand_queries, generate_hyde, check_retrieval_sufficiency |
| `self.gen_llm` | ChatOpenAI | simple_generate, generate_from_retrieval, calculate, compare |
| `_COMPLEX_KEYWORDS` | frozenset | classify_complexity |
| `_CALCULATE_KEYWORDS` | frozenset | classify_complexity |
| `_SIMPLE_KEYWORDS` | frozenset | classify_complexity |
| `_CN_DIGIT_MAP` | dict | normalize_article → validate |

### 3. AgentState 状态契约

```python
class AgentState(TypedDict, total=False):
    question: str                              # [入口写入] 原始问题
    rewritten_question: str                    # [rewrite_query写入] → 所有人读
    conversation_context: str                  # [入口写入] 对话上下文
    intent: str                                # [route_intent写入] → decide_intent读
    complexity: str                            # [classify_complexity写入] → decide_after_classify读
    context_docs: List[Tuple[Document, float]] # [retrieve/retrieve_for_compare写入] → generate/calculate/compare/validate读
    context_docs_b: List[Tuple[Document, float]] # [retrieve_for_compare写入] → compare读
    calculation_result: str                    # [calculate写入] → run()输出
    answer: str                                # [generate/calculate/compare/simple_generate写入] → validate读/覆盖 → run()输出
    steps: List[str]                           # [所有节点读写]
```

**已删除字段**：
- `comparison_result` — 从未被写入或读取
- `validation_passed` — 被写入但从未被读取（validate后直接END）

### 4. 节点读/写矩阵

| 节点 | 读 | 写 |
|------|----|----|
| rewrite_query | question, conversation_context | rewritten_question, steps |
| classify_complexity | rewritten_question∥question | complexity, steps |
| _decide_after_classify | complexity | *(路由)* |
| simple_generate | rewritten_question∥question, conversation_context, complexity | answer, context_docs, steps |
| route_intent | rewritten_question∥question, conversation_context | intent, steps |
| _decide_intent | intent | *(路由)* |
| retrieve_docs | rewritten_question∥question, complexity | context_docs, steps |
| retrieve_for_compare | rewritten_question∥question | context_docs, context_docs_b, steps |
| generate_from_retrieval | rewritten_question∥question, context_docs, conversation_context | answer, steps |
| calculate | rewritten_question∥question, conversation_context, context_docs | answer, calculation_result, steps |
| compare | rewritten_question∥question, context_docs, context_docs_b | answer, steps |
| validate | answer, context_docs | answer(覆盖), steps |

### 5. 图结构

```
rewrite_query → classify_complexity
  ├─ simple_path → simple_generate → END
  └─ agent_path → route_intent
       ├─ retrieve_path → retrieve_docs → generate_from_retrieval → validate → END
       ├─ calculate_path → retrieve_for_calc(=retrieve_docs) → calculate → validate → END
       └─ compare_path → retrieve_for_compare → compare → validate → END
```

### 6. 目标目录结构

```
app/core/generator/
├── __init__.py
├── state.py              # AgentState TypedDict
├── constants.py          # 关键词集合、数字映射、计算常量
├── prompts.py            # 所有Prompt模板（已有，补充移入）
├── guardrails.py         # 护栏（已有）
├── helpers.py            # 纯函数：_format_docs, _normalize_article, _strip_unsolicited_sections
├── calculators.py        # 计算器引擎：LABOR_CALCULATORS + auto_calculate + 辅助函数
├── nodes/
│   ├── __init__.py
│   ├── classify.py       # classify_complexity, decide_after_classify
│   ├── rewrite.py        # rewrite_query
│   ├── router.py         # route_intent, decide_intent
│   ├── retrieve.py       # retrieve_docs, retrieve_for_compare
│   ├── generate.py       # simple_generate, generate_from_retrieval
│   ├── calculate.py      # calculate
│   ├── compare.py        # compare
│   └── validate.py       # validate
└── graph.py              # 编排器：_build_graph + run + __init__（< 100行）
```
