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
| app.core.retriever.law_graph | `inject_related_articles`, `concept_lookup`, `_extract_law_article_from_doc`, `get_related_articles`, `_exact_lookup_from_parent_store`, `_LAW_SOURCE_MAP`, `_SOURCE_LAW_MAP` | 知识图谱（包，`__init__.py` 为门面） |

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

### 6. 实际落地的目录结构（2026-09-20）

> 下面是**实施后**的结构，与本节最初的提案（`app/core/generator/`）不同 ——
> 落地时按「能力分层」而非「技术类型」组织，最终归入 `app/agent/`，
> 且 `helpers` 的内容收敛到了 `app/utils/`（跨层通用原语应放最底层）。

```
app/
├── core/            config.py  logging.py                     最底层（配置、日志）
├── schemas/         chat.py                                   API DTO
├── utils/           law_refs.py  files.py  text.py            零依赖原语
├── knowledge/       loader.py  splitter.py  embeddings.py  store.py
├── retrieval/       base / vector / bm25 / hybrid / reranked / query_enhance
│                    domain_boost/  law_graph/  __init__.py（门面）
├── memory/          conversation.py
├── tools/           labor_calculator.py                       自含计算常量
├── agent/           graph.py  state.py  prompts.py  guardrails.py  constants.py
│   └── nodes/       classify  rewrite  router  retrieve
│                    generate  calculate  compare  validate
├── services/        rag_engine.py
├── evaluation/      datasets/  metrics/  runner.py  persistent.py
│                    report.py  utils.py
└── api/             routes.py（仅聚合）  deps.py  auth.py
                     chat.py  knowledge.py  evaluation.py
```

**分层规则（有 `tests/test_layering.py` 用 AST 扫描兜底，含函数内延迟导入）：**

```
core / schemas / utils  →  knowledge / retrieval / memory / tools
                        →  agent  →  services  →  evaluation  →  api
```

`app/main.py` 是唯一入口，允许依赖任意层。

#### 6.1 api 层的拆分（2026-09-17）

`routes.py` 原本 12814B，拆为「聚合器 + 三个子路由 + 共享 deps」：

| 文件 | 职责 |
|---|---|
| `routes.py` | **仅聚合**（679B）：把三个子路由 include 进一个 router |
| `deps.py` | 引擎单例引用 + `SUPPORTED_EXT` + `require_engine` |
| `auth.py` | `verify_bearer`（`AUTH_SECRET` 留空则直接放行） |
| `chat.py` | `/chat` |
| `knowledge.py` | `/knowledge-base/*`、`/documents/*` |
| `evaluation.py` | `/evaluation/*` |

**两条容易踩的约定：**

1. **鉴权必须挂在每个子路由上。** `include_router` **不会**让子路由继承聚合器的
   `dependencies`，所以三个子路由各自声明
   `APIRouter(dependencies=[Depends(verify_bearer)])`。
   漏挂新子路由 = 静默鉴权绕过 → 由 `tests/test_api_auth_wiring.py` 结构守卫兜底。
2. **引擎必须用模块属性访问** `deps.engine`，不能
   `from app.api.deps import engine` 后读本地名 —— 后者是 import 时的快照，
   lifespan 注入 / 测试 monkeypatch 替换引擎时不会跟着变。

**端点清单（22 个，拆分前后逐条核对无增减）：**
`/health`（挂在 `app` 上，免鉴权）、`/api/v1/chat`、
`/api/v1/knowledge-base/{status,build}`、`/api/v1/documents/{list,upload,{filename}}`、
`/api/v1/evaluation/{status,report,bad-cases,observability,run}`、
`/api/v1/evaluation/persistent/{create,start,resume,restart,retry-failed,force-stop,progress,tasks,report,delete}`

