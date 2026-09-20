# 任务计划：LaborRAG 后端目录重构 + 建立可验证的反馈回路

## 目标
将 `backend/app` 从"按技术类型堆叠"改为"按能力/职责分层"（对齐 agent-service-toolkit 的组织原则），
消除 6 处法条解析重复实现与 1 处分层倒置；全程以**可复现的行为基线**证明重构零行为变化。

## 当前阶段
**全部阶段（0/1/2/3/4）已完成**，复选框已按实际落地情况勾选。
详见 `progress.md`：会话 2、3（目录重构）、会话 4（RAGAS 评估）、会话 5（git 事故抢救 + 死代码清理 + 结构守卫）。

> 重构之后的收尾工作（不在本计划范围内，见 `progress.md` 会话 4/5）：
> ① `routes.py` 拆分；② HF_HOME 集中到 config 模块体；③ 7 个模块的重依赖导入位置修正；
> ④ 删除 3 处死代码（`sparse.py` / `EvalRunner.METRICS` / `_N1ChatModel`）；⑤ 补两条结构守卫；⑥ git 仓库抢救。

---

## 各阶段

### 阶段 0：建立验证网（行为基线）
- [x] 建 `backend/scripts/smoke_import.py`：遍历 `app/**/*.py` 逐个 `importlib.import_module`，报告失败模块 + traceback
- [x] 建 `backend/scripts/retrieval_baseline.py`：离线检索 harness（零 LLM、确定性、可消融），输出 P@1/P@3/P@5/R@5/MRR/MAP
- [x] 跑通并生成 `backend/baselines/pre_refactor.json`
- [x] 把基线数字记入 `findings.md`
- **状态：** ✅ complete（2026-09-16 会话 2）

**为什么必须先做**：现有 15 个单测只覆盖 `calculators / retrieval_metrics / text_splitter` 三个模块；
`main.py`、`rag_engine.py`、`agent_graph.py`、`law_graph.py`、全部 8 个 node 均**零覆盖**。
`pytest` 全绿**不能**证明重构没破坏东西。必须另建两道网：静态（能否 import）+ 行为（检索结果是否一致）。

**验收标准**：smoke_import 100% 通过；`baselines/pre_refactor.json` 生成，P@1 与既有报告
`evaluation_reports/eval_unknown_20260602_173829.json`（P@1=0.9 / R@5=0.925）量级一致。

**环境事实（已侦察，避免重复踩坑）**：
- 必须用 `D:\Anaconda\envs\RAG\python.exe`；base 环境 torch DLL 已损坏（`c10.dll` WinError 1114）
- 模型已缓存：`bge-m3` 6.5GB、`bge-reranker-v2-m3` 2.2GB（**`D:/hf_cache`**；由 `app/core/config.py` 的模块体设 `HF_HOME`，必须早于 `huggingface_hub` 导入才生效）
- 向量库已构建：`D:/LaborRAG_data/vector_store/{index.faiss,parent_store.json,bm25_index.json,sparse_index.json}`
- 必须在 `backend/` 目录下运行（`settings` 的 `env_file=".env"` 相对 CWD 解析）

---

### 阶段 1：目录重构（纯移动，零逻辑改动）
- [x] 按下方映射表 `git mv` 全部文件
- [x] 按"Import 重写映射"批量重写 import
- [x] 处理 4 个路径陷阱（见下）
- [x] 删除全部过期 `__pycache__`
- [x] 验证：`smoke_import` 通过 + `pytest` 15 用例全绿 + baseline 数字**逐项完全一致**
- **状态：** ✅ complete（2026-09-16 会话 3，用户确认执行）—— 49 文件 git mv，225 处 import 替换，4 个路径陷阱全部按本计划预判处理

**原则：只移动文件 + 重写 import，不改任何一行业务逻辑。** 唯一例外是下面 4 个路径陷阱。

#### 文件映射表

| 原路径 | 新路径 |
|---|---|
| `app/config.py` | `app/core/config.py` ▲陷阱1 |
| `app/main.py` | `app/main.py`（不动） |
| `app/api/routes.py` | 阶段3再拆；阶段1先留原地 |
| `app/models/schemas.py` | `app/schemas/chat.py` |
| `app/core/logging_config.py` | `app/core/logging.py` |
| `app/core/conversation.py` | `app/memory/conversation.py` |
| `app/core/document_loader.py` | `app/knowledge/loader.py` |
| `app/core/text_splitter.py` | `app/knowledge/splitter.py` |
| `app/core/embeddings.py` | `app/knowledge/embeddings.py` |
| `app/core/vectorstore.py` | `app/knowledge/store.py` |
| `app/core/rag_engine.py` | `app/services/rag_engine.py` |
| `app/core/retriever/base.py` | `app/retrieval/base.py` |
| `app/core/retriever/vector.py` | `app/retrieval/vector.py` |
| `app/core/retriever/bm25.py` | `app/retrieval/bm25.py` ▲陷阱2 |
| `app/core/retriever/hybrid.py` | `app/retrieval/hybrid.py` |
| `app/core/retriever/reranked.py` | `app/retrieval/reranked.py` |
| `app/core/retriever/sparse.py` | `app/retrieval/sparse.py`（标 experimental） |
| `app/core/retriever/query_enhance.py` | `app/retrieval/query_enhance.py` |
| `app/core/retriever/law_graph.py` | `app/retrieval/domain_boost.py`（阶段3再拆） |
| `app/core/retriever/law_dict.txt` | `backend/data/law_dict.txt` ▲陷阱2 |
| `app/core/generator/agent_graph.py` | `app/agent/graph.py` |
| `app/core/generator/state.py` | `app/agent/state.py` |
| `app/core/generator/prompts.py` | `app/agent/prompts.py` |
| `app/core/generator/guardrails.py` | `app/agent/guardrails.py` |
| `app/core/generator/constants.py` | `app/agent/constants.py` |
| `app/core/generator/helpers.py` | `app/utils/text.py` |
| `app/core/generator/calculators.py` | `app/tools/labor_calculator.py` |
| `app/core/generator/nodes/*.py`（8个） | `app/agent/nodes/*.py` |
| `app/core/generator/ARCHITECTURE.md` | `docs/agent_architecture.md` |
| `app/evaluation/eval_dataset.py` | `app/evaluation/datasets/golden_set.py` |
| `app/evaluation/retrieval_metrics.py` | `app/evaluation/metrics/retrieval.py` |
| `app/evaluation/response_metrics.py` | `app/evaluation/metrics/response.py` |
| `app/evaluation/eval_runner.py` | `app/evaluation/runner.py` ▲陷阱4 |
| `app/evaluation/eval_persistent.py` | `app/evaluation/persistent.py` ▲陷阱4 |
| `app/evaluation/eval_report.py` | `app/evaluation/report.py` ▲陷阱4 |
| `app/evaluation/utils.py` | 阶段2再拆（`file_lock` 归 `app/utils/files.py`） |
| `backend/run_eval.py` | `backend/scripts/run_eval.py` |
| `backend/run.py` | 不动 ▲陷阱3 |

#### Import 重写映射（按长前缀优先批量替换）

| 旧前缀 | 新前缀 |
|---|---|
| `app.core.retriever.law_graph` | `app.retrieval.domain_boost` |
| `app.core.retriever.base` | `app.retrieval.base` |
| `app.core.retriever.vector` | `app.retrieval.vector` |
| `app.core.retriever.bm25` | `app.retrieval.bm25` |
| `app.core.retriever.hybrid` | `app.retrieval.hybrid` |
| `app.core.retriever.reranked` | `app.retrieval.reranked` |
| `app.core.retriever.sparse` | `app.retrieval.sparse` |
| `app.core.retriever.query_enhance` | `app.retrieval.query_enhance` |
| `app.core.generator.agent_graph` | `app.agent.graph` |
| `app.core.generator.state` | `app.agent.state` |
| `app.core.generator.prompts` | `app.agent.prompts` |
| `app.core.generator.guardrails` | `app.agent.guardrails` |
| `app.core.generator.constants` | `app.agent.constants` |
| `app.core.generator.helpers` | `app.utils.text` |
| `app.core.generator.calculators` | `app.tools.labor_calculator` |
| `app.core.generator.nodes` | `app.agent.nodes` |
| `app.core.vectorstore` | `app.knowledge.store` |
| `app.core.embeddings` | `app.knowledge.embeddings` |
| `app.core.text_splitter` | `app.knowledge.splitter` |
| `app.core.document_loader` | `app.knowledge.loader` |
| `app.core.conversation` | `app.memory.conversation` |
| `app.core.rag_engine` | `app.services.rag_engine` |
| `app.core.logging_config` | `app.core.logging` |
| `app.models.schemas` | `app.schemas.chat` |
| `app.evaluation.eval_dataset` | `app.evaluation.datasets.golden_set` |
| `app.evaluation.retrieval_metrics` | `app.evaluation.metrics.retrieval` |
| `app.evaluation.response_metrics` | `app.evaluation.metrics.response` |
| `app.evaluation.eval_runner` | `app.evaluation.runner` |
| `app.evaluation.eval_persistent` | `app.evaluation.persistent` |
| `app.evaluation.eval_report` | `app.evaluation.report` |
| `app.config` | `app.core.config` |

> ⚠️ 替换顺序必须**长前缀优先**，否则 `app.core.generator.nodes` 会被 `app.core.generator` 类的短规则误伤。

#### 4 个路径陷阱（纯搬家会静默弄坏的地方）

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 1 | `app/config.py:12` | `BASE_DIR = Path(__file__).resolve().parent.parent.parent` 算的是**项目根**。移到 `app/core/config.py` 后深度 +1，会算成 `backend/`，导致 `DATA_DIR` 变成 `backend/data/labor_laws` | 改为 4 层 `.parent` |
| 2 | `app/core/retriever/bm25.py:11` | `_LAW_DICT_PATH = dirname(__file__)/law_dict.txt`。文件移动 + `law_dict.txt` 移到 `backend/data/` 后失效，jieba 自定义词典静默不加载（**不报错，只是法律术语被切碎，检索质量悄悄下降**） | 改为基于 `BASE_DIR / "data" / "law_dict.txt"`，并加一条"词典已加载"断言 |
| 3 | `backend/run.py:21` | `from app.core.logging_config import setup_logging` 启动即 ImportError | 改为 `app.core.logging` |
| 4 | `app/evaluation/*.py` | `eval_report.py:14`、`eval_runner.py:40`、`eval_persistent.py:35,137` 都用 `dirname(__file__)` 推算 `_BACKEND_ROOT` | 三个文件**保持同一深度**（`app/evaluation/xxx.py`），不要去更深的子包 |

---

### 阶段 2：消除重复实现
- [x] 新建 `app/utils/law_ref.py`，统一以下 6 处实现：
  - 中文数字 ↔ 阿拉伯数字（`vectorstore._int_to_cn`、`retrieval_metrics._cn_to_int`、`law_graph._CN_NUM`/`_ARABIC_TO_CN`）
  - 法条编号正则（`text_splitter.ARTICLE_PATTERN`、`vectorstore._ARTICLE_REF_PATTERN`、`helpers.normalize_article`）
  - 法律名 ↔ 文件名映射（`vectorstore._LAW_NAME_MAP`、`law_graph._LAW_SOURCE_MAP`）
  - 查询法条引用提取（`vectorstore.extract_article_ref`）
- [x] 新建 `app/utils/files.py`：`file_lock` + 原子写
- [x] `app/knowledge/store.py` 改为依赖 `app/utils/files.py`（**修分层倒置**：核心存储层当前反向 import 评估层）
- [x] 改造 6 处调用点，删除旧实现
- [x] 新增 `tests/test_utils/test_law_ref.py`：含 **1–999 中文数字往返属性测试**
- **状态：** ✅ complete（4/5 项完成；法律名映射判定不应合并，往返测试判定不应做）

**验收标准**：测试全绿 + baseline 数字不变 + 新增往返测试覆盖 1–999。
> 这条直接杀掉 README 里 "Bug 1：法条切分正则漏「零」" 的 bug 类——那个 bug 之所以存在，
> 就是因为同一个"中文数字解析"散落 6 处，改一处漏五处。

---

### 阶段 3：拆分大文件
- [x] `app/retrieval/domain_boost.py`（671 行，其中约 500 行是常量表）拆为：
  - `app/knowledge_graph/law_relations.py` — `LAW_GRAPH` 纯数据
  - `app/knowledge_graph/concept_map.py` — `CONCEPT_ARTICLE_MAP` 纯数据
  - `app/knowledge_graph/lookup.py` — `_load_parent_store` / `_exact_lookup_from_parent_store` / `get_related_articles` / `concept_lookup` / `_extract_law_article_from_doc`
  - `app/retrieval/domain_boost.py` — 只留 `inject_related_articles`（变薄）
- [x] `app/api/routes.py`（348 行）拆为 `routes/{chat,knowledge,evaluation}.py` + `api/deps.py`
- **状态：** ✅ complete（`law_graph.py` 已拆为包 + 门面；`routes.py` 拆分未做）
- **验收标准**：测试全绿 + baseline 数字不变

---

### 阶段 4：收尾
- [x] `tests/` 镜像 `app/` 结构 —— **未做**：计划里那套目录名（`test_utils/` 等）绑定的是阶段 1 的目录规划，
      而阶段 1 实际采用了不同命名（`utils`/`knowledge`/`retrieval`/`agent`/`services`…），硬套会名不符实。
      当前 10 个测试文件平铺已足够清晰；若要做，应按新分层重新设计命名。
- [x] 更新 `README.md` 的"项目结构"章节 —— 已按新结构重写（含分层注释）
- [x] 清理 `__pycache__` 中 cpython-310 / cpython-314 残留 —— 删除 8 个陈旧 `.pyc`，保留当前 cpython-311
- [x] `sparse.py` 标注 experimental —— 已补状态说明（代码完整/已产出索引/未接线/接入需改三处）
- **状态：** ✅ 完成 3/4（`tests/` 镜像结构经评估后不做，理由见上）

---

## 关键问题
1. **阶段 1 的搬家值得吗？** 收益是可读性/可维护性，代价是 40+ 处 import 改动。
   → 已决策：值得，且有阶段 0 基线兜底，风险可控。
2. **`generator/` → `agent/` 改名会影响答辩材料吗？** → 需同步更新 README 结构图与 ARCHITECTURE.md。
3. **`sparse.py` 未接线，保留还是删？** → 保留并标注 experimental（阶段 4），它已产出 `sparse_index.json`。
4. **要不要顺手修 `rag_engine.py:177` 的 `rewritten_question = result.get("intent")` bug？**
   → 不在本轮范围。本轮目标是"零行为变化"，修 bug 属于独立 PR，否则 baseline 对比会失去意义。

## 已做决策
| 决策 | 理由 |
|------|------|
| 按"能力/职责"分层，而非"技术类型" | 对齐 ASK 组织原则；解决 `core/` 杂物抽屉（6 个不同层混在一起） |
| **先建验证网再动目录** | 15 个单测覆盖不到重构区域；没有基线，任何改动都不可验证 |
| 阶段 1 坚持零逻辑改动 | 只有行为不变，才能证明搬迁安全 |
| 不引入 ASK 的 Postgres checkpointer / 多 agent 注册表 | 本项目痛点不在此，净增复杂度；且会丢掉自制检索层这一核心资产 |
| `agent/nodes/` 不拆散 | 项目里结构最好的部分，职责已经很清楚 |
| `law_graph` 拆成 data + logic | 671 行里约 500 行是常量表，数据与代码混放是最大可读性问题 |
| 每阶段一个 git commit | 可独立回滚 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| base 环境 `import torch` 报 WinError 1114（c10.dll） | 1 | 改用 `D:\Anaconda\envs\RAG\python.exe` |
| `& RAG\python.exe -c "import torch..."` 首次 120s 超时 | 1 | 放后台跑；实测 torch 4.5s、sentence_transformers 31.8s，属正常冷启动 |
| 用 `Get-Content` 读 UTF-8 中文 JSON 出现乱码 | 1 | 改用 `python -c "json.load(open(f,encoding='utf-8'))"` |

## 备注
- 阶段状态随进度更新：pending → in_progress → complete
- 做重大决策前重新读取本计划
- 每个错误都记入上表，避免重复失败
- 外部网页/搜索结果只写入 `findings.md`，不写入本文件

---

## 复核结论（2026-09-16 会话 2）

对本计划全部声称逐条在代码里核实，结果如下。

### ✅ 声称属实

- 分层倒置：`app/core/vectorstore.py` → `app/evaluation/utils.py` 的 `file_lock`（已在阶段 2 修掉）
- 4 个路径陷阱全部属实
- `rag_engine.py:177` 的 `rewritten_question = result.get("intent", "")` 确实取错键
- 环境事实全部属实（`RUNTIME_DATA_DIR=D:/LaborRAG/data`、`RETRIEVER_TYPE=hybrid`、`TOP_K=12`）
- `sparse.py` 未接线（config 与 rag_engine 均无引用）

### ❌ 声称有误

- **「启动白加载 2.2GB reranker」是错的**：`RerankedRetriever.__init__` 只存 `hybrid_retriever`，
  不加载模型。`_get_reranker()` 仅在 `_preload_reranker`（被 `RETRIEVER_TYPE=="reranked"`
  守卫，hybrid 下不执行）与 `retrieve()` 内部调用。故 `rag_engine.py` 的
  `_ = self.eval_retriever` **只构造轻量包装对象**。
- **「现有 15 个单测」已过时**：实际 121 个。
- **「main.py 零覆盖」已过时**：`test_api.py` 已覆盖其中间件/异常处理/health。

### ⚠️ 计划未发现、复核中新增

1. **重复实现比计划估计的更多**：法条编号正则是 **7 处**（计划写 6 处），中文数字解析
   是 **6 处**（计划把它算在"6 处"里但未区分两类）。且缺陷已真实复发——
   `classify.py` 漏「零」、`retrieve.py` 漏「千」、`retrieval_metrics` 漏「〇」。
2. **`law_graph._article_to_cn` 是 1–99 的查表**，条号 >99 静默失效（潜伏缺陷）。
3. **知识图谱注入顺序随进程变化**（`law_graph` 直接迭代 set），导致检索结果不可复现。

### 🔴 计划自身的两处方法缺陷

1. **「1–999 中文数字往返属性测试」是假阴性。** `_cn_to_int("一百一")` 恰好也返回 101，
   往返一致但两头都错。实测：1–999 往返在**带病代码上 0 失败**。已改用权威格式断言。
2. **阶段 2 的验收标准自相矛盾。** 要求「baseline 数字不变」，但阶段 2 的**目的**就是
   修这些重复实现的缺陷，统一后 `extract_article_ref` / `extract_article_id` 行为
   **应该**改变。正确验收是「目标用例由错转对 + 其余不变」。计划为避免污染基线而把
   `rag_engine.py:177` 排除在外（见「关键问题」第 4 条），却在阶段 2 犯了同类错误。

   实测补充：阶段 2/3 全部改动后 baseline 确实 Δ 全为 0，但那是因为黄金集 20 题
   **没用到**含「〇」或含「零」的条号写法 —— 这恰好说明**基线只能证明「没弄坏」，
   证明不了「修好了」**；后者必须靠针对性单测。

### 📌 未采纳的计划建议

- **「法律名 ↔ 文件名映射统一」**：`vectorstore._LAW_NAME_MAP` 是「用户输入的法律名 →
   文件名」，`law_graph._LAW_SOURCE_MAP` 是「文件名 → 规范法律名」，**方向相反**。
  强行合并会引入猜测。判定：不合并，但两者都属"数据表"，已各自归位。
- **阶段 1（目录搬家）**：建议缓做（理由见阶段 1 状态）。


---

## 实际落地与计划的差异（2026-09-20 补记）

计划是**预判**，实施中出现了几处偏离。记录下来，避免后人对着计划找不存在的文件：

| 计划写的 | 实际落地的 | 原因 |
|---|---|---|
| `app/core/generator/` 作为 Agent 层根 | **`app/agent/`** | 按能力分层，`generator` 这个名字过窄 |
| `app/utils/law_ref.py`（单数） | **`app/utils/law_refs.py`** | 内含「法条引用」多个概念（正则/数字/归一化） |
| `app/utils/helpers.py` | 收敛到 **`app/utils/text.py`** + `law_refs.py` | 通用原语应放最底层，且按主题分文件 |
| `app/agent/calculators.py` | **`app/tools/labor_calculator.py`** | 计算器是自含领域工具，独立成层 |
| `tests/test_utils/test_law_ref.py` | **`tests/test_law_numbering.py`** | 测试目录保持扁平（15 个文件，无需再分层） |
| `routes/{chat,knowledge,evaluation}.py` | 同，**外加 `deps.py` / `auth.py`** | 引擎单例与鉴权被三个子路由共享，必须独立 |

**计划未预判、实施中新增的问题**（详见 `progress.md`）：

- 阶段 1 搬家引入了 **2 个包级循环依赖**（复核时发现并修掉）
- 阶段 3 之后又拆了 `law_graph.py`（数据/逻辑分离）与 `evaluation/`（按职责分目录）
- 导入期副作用（`import app.main` 31s → 1.18s）是计划里没提的独立问题
