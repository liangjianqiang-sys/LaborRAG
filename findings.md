# 发现与决策

> **复核状态（2026-09-16 会话 2）**：本文的每条"发现"已在代码里逐条核实，
> 结论（属实 / 有误 / 新增）记录在 `task_plan.md` 的「复核结论」章节，
> 完成情况见 `progress.md`。以下内容保留原始记录，**未逐条就地修订**，
> 以免失去当时的推理轨迹。

## 需求
- 用户认为本项目目录结构不清晰（对比 `agent-service-toolkit` 的 `tool`/`memory` 分层）
- 诉求：**先重组目录结构，使项目更清晰**
- 背景痛点（前序对话）：检索层缺测试，改一小处无法反馈效果

## 研究发现

### A. agent-service-toolkit（ASK）真实目录结构
来源：<https://github.com/JoshuaC215/agent-service-toolkit>（README + GitHub tree API，2026-09-16 拉取）

```
src/
├── agents/                     # 每个能力一个文件/目录
│   ├── agents.py               #   注册表（dict: name -> agent）
│   ├── chatbot.py
│   ├── rag_assistant.py
│   ├── knowledge_base_agent.py
│   ├── interrupt_agent.py
│   ├── bg_task_agent/          #   能力独占目录：bg_task_agent.py + task.py
│   ├── github_mcp_agent/
│   ├── tools.py                #   工具定义独立成文件
│   └── utils.py
├── core/                       # 只放基础设施，极小
│   ├── settings.py
│   └── llm.py
├── memory/                     # 独立成层，多后端并列
│   ├── postgres.py
│   ├── mongodb.py
│   └── sqlite.py
├── schema/                     # 协议层独立
│   ├── models.py  schema.py  task_data.py
├── service/                    # service.py  agui.py  threads.py  utils.py
├── client/                     # client.py
├── voice/                      # manager.py  providers/  stt.py  tts.py
└── run_{agent,client,service}.py

tests/                          # 镜像 src/ 结构
├── agents/  app/  client/  core/  schema/  service/  smoke/  voice/
```

**可迁移的 4 条组织原则**：
1. `core/` 只放基础设施（settings/llm），保持很小
2. 能力按**目录**切，不按技术类型切（无 `utils/`、`models/` 大杂烩）
3. **数据与逻辑分离**（`memory/` 的三种后端并列；能力目录内放专属子模块）
4. **`tests/` 镜像源码结构**，一眼看出哪个模块有覆盖

**本项目不适用的部分**：多 agent 注册表、LangGraph checkpointer + `thread_id`、Streamlit UI、
HITL `interrupt()`、`Store` 长期记忆、AG-UI 协议、内容审核、语音、Docker+Postgres。
其 RAG agent 自述为 "basic RAG agent using ChromaDB"——远弱于本项目自制检索层。

### B. LaborRAG 现状的 6 个结构问题

| # | 问题 | 证据 |
|---|---|---|
| 1 | `core/` 是杂物抽屉，混了 6 个层 | 基础设施 `logging_config`、模型 `embeddings`、离线 ingestion `document_loader`+`text_splitter`、存储 `vectorstore`、memory `conversation`、应用服务 `rag_engine` |
| 2 | `memory` 语义缺失 | `core/conversation.py` 是 memory 层却被埋最深；项目实有**两种 memory**（对话历史 + `vectorstore.parent_store`）分散两处 |
| 3 | `law_graph.py` 671 行职责爆炸 | 约 500 行是常量表 `LAW_GRAPH` + `CONCEPT_ARTICLE_MAP`，与存储访问、检索逻辑、中文数字工具混放 |
| 4 | **分层倒置** | `app/core/vectorstore.py:308` → `from app.evaluation.utils import file_lock`（核心存储层反向依赖评估层） |
| 5 | **同一逻辑 6 处实现** | 见下方"C. 法条解析重复实现清单" |
| 6 | 数据文件混在代码里 | `retriever/law_dict.txt`（jieba 词典）；`tests/` 扁平 3 文件 |

### C. 法条解析重复实现清单（阶段 2 的收敛目标）

| # | 位置 | 职责 |
|---|---|---|
| 1 | `app/core/text_splitter.py:19` | `ARTICLE_PATTERN` 法条标题正则 |
| 2 | `app/core/vectorstore.py:14` | `_ARTICLE_REF_PATTERN` 法条引用正则 |
| 3 | `app/core/vectorstore.py:52-85` | `_DIGIT_TO_CN` / `_int_to_cn` / `_normalize_article_num` |
| 4 | `app/core/vectorstore.py:18-49` | `_LAW_NAME_MAP` / `_LAW_NAME_PATTERN` 法律名→文件名 |
| 5 | `app/core/retriever/law_graph.py:15-62` | `_LAW_SOURCE_MAP` / `_CN_NUM` / `_ARABIC_TO_CN` / `_article_to_cn` |
| 6 | `app/evaluation/retrieval_metrics.py:23-99` | `_ARTICLE_PATTERN` / `_CN_DIGITS` / `_cn_to_int` / `_normalize_article` / `_LAW_NAME_SHORT` |
| 7 | `app/core/generator/helpers.py` | `normalize_article` |
| 8 | `app/core/retriever/query_enhance.py:11-89` | `_KEYWORD_MAP` 硬编码法条号字符串 |

> **这是 README "Bug 1：法条切分正则漏「零」" 的根因**：同一个"中文数字解析"散落 8 处，
> 改一处漏七处。收敛成单一实现后，这类 bug 从"会复发"变为"不可能发生"。

### D. 环境事实（已实测）

| 项 | 值 |
|---|---|
| Python（可用） | `D:\Anaconda\envs\RAG\python.exe` |
| Python（不可用） | base `D:\Anaconda\python.exe` — `import torch` 报 WinError 1114，`c10.dll` 加载失败 |
| torch | 2.11.0+cpu，import 耗时 4.5s |
| sentence_transformers | import 耗时 31.8s（冷启动正常） |
| 模型缓存 | **`D:/hf_cache/hub/`**（2026-09-20 起，由 `config.py` 模块体设 `HF_HOME`）：`bge-m3` ~6.5GB、`bge-reranker-v2-m3` ~2.2GB、`bert-base-chinese` ~0.4GB。**已不在 `~/.cache/huggingface`** —— 该目录是清理软件的常见目标，实测被整体清空过一次（丢约 8GB），表现为启动时静默重新下载 |
| 向量库 | `D:/LaborRAG_data/vector_store/`：`index.faiss` 5.1MB、`parent_store.json` 528KB、`bm25_index.json` 1.07MB、`sparse_index.json` 1.72MB |
| 缺失依赖（base 环境） | `ragas` 未安装 → 完整 RAGAS 评估在 base 环境跑不了 |
| 运行前提 | 必须在 `backend/` 下执行（`settings` 的 `env_file=".env"` 相对 CWD） |
| `.env` 关键配置 | `RETRIEVER_TYPE=hybrid`（**非** reranked）、`RETRIEVAL_VALIDATION=false`、`TOP_K=12`、`RERANK_SCORE_THRESHOLD=0.3`、`RUNTIME_DATA_DIR=D:/LaborRAG_data` |

### E. 路径陷阱清单（纯搬家会静默弄坏的地方）

由 `grep __file__|parents\[|sys\.path` 得出，涉及 10 处：

| 位置 | 现状 | 搬家后的风险 |
|---|---|---|
| `app/config.py:12` | `BASE_DIR = Path(__file__).resolve().parent.parent.parent` | 移到 `app/core/config.py` 后深度 +1 → `BASE_DIR` 变成 `backend/`，`DATA_DIR` 错位 |
| `app/core/retriever/bm25.py:11` | `_LAW_DICT_PATH = dirname(__file__)/law_dict.txt` | 双重失效（自身移动 + 词典移走）；**静默失败**，jieba 自定义词典不加载，检索质量悄悄下降 |
| `backend/run.py:12-13` | `BACKEND_DIR` + `sys.path.insert` | 不受影响（run.py 不动） |
| `backend/run.py:21` | `from app.core.logging_config import setup_logging` | 启动即 ImportError |
| `app/evaluation/eval_report.py:14` | `dirname ×3` 推算路径 | 保持同深度即可不变 |
| `app/evaluation/eval_runner.py:40` | `_BACKEND_ROOT = dirname/../..` | 同上 |
| `app/evaluation/eval_persistent.py:35,137` | `_BACKEND_ROOT` + `dirname ×3` | 同上 |

### F. 既有评估数据（用于基线对齐）

- `backend/evaluation_reports/eval_unknown_20260602_173829.json`（20 题）：
  `faithfulness=0.9197`、`precision@1=0.9`、`recall@5=0.925`、`hallucination_rate=0.0803`、`completeness=0.8947`
  → 与 README 宣称的 92% / 90% / 92.5% / 8% / 89.5% **完全对应**，数字属实。
- 注意：`precision@5` 恒为 0.2~0.4，因为 `precision_at_k` 分母固定为 k 而每题只有 1–2 条相关法条，
  该指标在本数据集上**不可解释**，不建议作为优化目标。
- `backend/evaluation_reports/persistent/reports/eval_20260530_191438_6b4b34_report.json`（36 题）：
  `faithfulness=0.6105`、`P@1=0.8056` — 规模不同，不可与上面直接比。

### G. 检索层现状（供后续优化参考）

- 交互问答走 `RETRIEVER_TYPE=hybrid` → `HybridRetriever`（**不含 reranker**）
- reranker 仅在评估路径启用：`eval_runner.py:146` 与 `eval_persistent.py:639` 传 `use_reranker=True`
- 但 `rag_engine.py:46` 启动时仍 `_ = self.eval_retriever`，即**加载了 2.2GB reranker 却不用于聊天**
- `app/core/retriever/sparse.py`（BGE-M3 学习型稀疏）已实现且 `sparse_index.json` 已生成，
  但 `RETRIEVER_TYPE` 只支持 `vector|hybrid|reranked` —— 属未接线的实验代码

## 技术决策
| 决策 | 理由 |
|------|------|
| 采用 ASK 的 4 条组织原则，拒绝其基础设施 | 组织原则普适；Postgres/多 agent/Streamlit 与本项目痛点无关 |
| 阶段 1 严格零逻辑改动 | 只有行为不变，才能用基线证明搬迁安全 |
| 先行建立"静态 import 网 + 行为基线网" | 现有 15 个单测只覆盖 3 个模块，绿 ≠ 没坏 |
| 修 bug（如 `rag_engine.py:177`）排除在本轮外 | 混入行为变更会让 baseline 对比失去意义 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| base 环境 torch DLL 损坏 | 改用 conda `RAG` 环境 |
| `Get-Content` 读 UTF-8 中文 JSON 乱码 | 用 Python `json.load(f, encoding='utf-8')` |
| base 环境缺 `ragas` | 完整 RAGAS 评估必须用 `RAG` 环境 |

### 静默缺陷清单（2026-09-17 ~ 09-20 追加）

**共同特征：不报错、不抛异常，只是悄悄丢数据或走错分支。** 这也是为什么
「pytest 全绿」不能作为验收依据 —— 下面每一条在修复前测试都是绿的。

| # | 缺陷 | 形态 | 怎么发现的 |
|---|------|------|-----------|
| 9 | 复杂度分类只看改写结果 | 改写剥掉口语标记 → 分类降级为 medium | 真实 LLM 跑通后对比 |
| 10 | 聚合检索指标缺 5 项 | 只输出 2 个指标，其余算了没带出来 | 实现与 docstring 对照 |
| 11 | RAGAS 按模型名判断是否关思考 | 非 qwen 模型 → 98% 输出 token 是思考 | 读 `runner.py` 时发现 |
| 12 | 对比路径用改写结果做概念提取 | 连接词被剥掉 → 退化成单查询检索 | 日志里一行「对比概念提取失败」 |
| 13 | CLI 硬编码错误指标键名 | `P@1` vs `precision@1` → 检索指标永远显示 N/A | 评估输出 N/A 但 Phase 2 明明算出来了 |
| 14 | 幻觉护栏贪婪正则 | 删幻觉时把前面整段正确内容一起吞掉 | 盲区扫描：validate 零测试 |
| 15 | 修 14 时自己引入的换行塌陷 | `if s.strip()` 吃掉 markdown 段落空行 | **端到端打印真实响应才看到** |
| 16 | 关键词表顺序导致短词遮蔽长词 | 「工伤认定」排在「工伤认定时限」前 → 命错法条 | 客观扫描：O(n²) 查子串遮蔽 |

**Bug 14/15 的教训**：单测证明「删除范围正确」，但坏掉的是「拼接方式」——
**修 A 引入 B 是规则型修复的常见形态，修完之后必须再验一次副作用。**
Bug 15 只在护栏触发时发生（未触发直接返回原答案），所以是隐形的。

### 环境坑（2026-09-20 追加，都实际踩过）

| 坑 | 表现 | 对策 |
|---|---|---|
| `HF_HOME` 设晚了 | 不报错，模型照旧下到 C 盘 `.cache`（被清理软件清空过） | 必须写在 `config.py` **模块体**（`huggingface_hub` 在 import 时固化该路径） |
| Git Bash 转换反斜杠 | heredoc 里 `\n` → `/n`、`\\` → `//`，导致正则「莫名不匹配」 | 写正则/路径**不用反斜杠**，改用 `chr(10)` 或逐行处理 |
| `tar` 把 `D:/x` 当远程主机 | `Cannot connect to D: resolve failed` | 用 POSIX 路径 `/d/x` |
| 本机 `http_proxy` 拦本地请求 | 打 `127.0.0.1:8000` 返回 **502**，像「服务挂了」 | `build_opener(ProxyHandler({}))` 绕过 |
| 后台 uvicorn 脱离 shell | 包装进程退出后**进程仍存活**占端口 | `netstat -ano \| grep :8000` 找 PID 再结束 |
| `.git` 会被移进回收站 | git 报「不是仓库」 | 见 `progress.md` 会话 5 的完整修复记录 |

## 资源
- [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit)
- [RAGAS Experimentation](https://docs.ragas.io/en/stable/concepts/experimentation/)（已在依赖中，可用于后续配置对比）
- [RAGAS Testset Generation](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)（可用于扩充 20 题 Golden Set）
- [LangGraph 本地服务器 `langgraph dev`](https://docs.langchain.com/oss/python/langgraph/local-server)
- [Arize Phoenix](https://arize.com/docs/phoenix/evaluation/llm-evals)

## 视觉/浏览器发现
- 无（本会话未做截图/浏览器操作）

---
*每执行 2 次查看/浏览器/搜索操作后更新此文件*
*外部内容只写本文件，不写 task_plan.md*


---

## Context Recall 0.5667 的归因分析（2026-09-21）

### 问题

smoke（5 题）的 RAGAS `context_recall = 0.5667` 偏低。这需要判断：
**是 golden set 标注的问题，还是检索的问题？** 两者在论文里的写法完全不同 ——
前者要写「评估集偏差」，后者要写「已知局限」。

### 方法：三个口径互相对照（零 LLM 成本）

数据源：`evaluation_reports/cache/answers_smoke_after_refactor_*.json`
（上次 smoke 的真实答案 + 父块上下文 + ground_truth）。

| 口径 | 结果 | 怎么测的 |
|---|---|---|
| 法条引用层面 | **78.6%** | ground_truth 引用的 14 条法条，11 条出现在检索上下文里 |
| 内容词覆盖 | **91.2%** | jieba 分词后逐句比对内容词（排除停用词） |
| RAGAS context_recall | **56.67%** | 上次评估报告 |

### 结论

**检索上下文里确实包含 ground_truth 的内容**（91% 的内容词都在，
法条层面 78.6% 覆盖）——所以低分**主要不是「检索没检到」，也不是标注与语料不符**。

### 已确认的真缺陷（Bug 20）：对比题只评估一半证据

```
retrieve_for_compare  →  {"context_docs": docs_a, "context_docs_b": docs_b}
rag_engine.py:157     →  relevant_docs = result["context_docs"]      ← 只取 A 组
                        sources / full_contexts / child_contexts 全部由它派生
```

这是**结构性的口径不一致**：答案由 A+B 两组生成，评估只看 A 组。
`retrieve_for_compare` 的 `context_docs_b` 除了喂给 `compare` 节点，
在 `rag_engine` 里**再没被任何地方读过**。

**理论上**受影响的是：

- 对比题的检索指标（B 组的命中不计入）
- 对比题的 RAGAS 三项（B 组的法条不在 contexts 里）

### ⚠️ 但在现有 5 题样本里**没有观察到实际影响**

逐题核对「答案引用的法条」是否都能被评估看见：

| 题 | 答案引用 | 评估可见 | 引用但不可见 |
|---|---|---|---|
| 没签合同被辞了 | 5 条 | 7 条 | 无 |
| 加班工资怎么算 | 5 条 | 26 条 | 无 |
| 孕期被公司辞退了 | 3 条 | 14 条 | 无 |
| 月薪8000加班10小时 | 2 条 | 23 条 | 无 |
| 协商解除和公司单方辞退 | 8 条 | 14 条 | 无 |

**0/5 题出现「答案引用了评估看不见的法条」。** 实测那题的 A 组（11 篇）与 B 组（18 篇）
内容有较大重叠，答案引用的第 18/19/24/36/40/41/43/46 条恰好都在 A 组里。

检索指标层面同样未观察到影响：该题标注 {36, 39, 40}，A 组含 36（39 排在第 6 位之后），
而 B 组既不含 39 也不含 40 —— 把 B 组计入也不会改变这道题的召回。

**所以：这是真实的结构缺陷（口径不一致、B 组数据被丢弃），
但它不是 Context Recall 偏低的原因，对现有数字的影响也未观察到。
修它的理由是「口径应当一致」这一原则，而非「改了分数会变好」。**

**修与不修都会改变论文里已有的数字，因此先记录、由作者决定。**

### 91% vs 57% 的剩余差距

现有数据无法定位 —— 评估报告只存了**逐题 faithfulness**，没存逐题 context_recall。
可能的原因（按可能性排序）：

1. **判分模型偏严**（RAGAS 槽 = `deepseek-v4.1-flash`）—— 要求上下文**显式陈述**
   该断言，而不仅是包含相同词汇
2. **RAGAS 用的是 child contexts**，而本次分析用的是 parent contexts（前者是后者的切分）；
   实际 child contexts 未被缓存，无法核对
3. 其他口径问题

**低成本定位手段**：只跑 `context_recall` 一项、只用这 5 题的缓存数据
→ 约 **5 次 LLM 调用 / ~20k token**（对比全量 smoke 约 200 次调用、13 分钟）。
这是能给出逐题分数、直接回答「标注 vs 检索」的最小实验。


---

## Context Recall 0.5667 的归因分析（2026-09-21）

### 问题

smoke（5 题）的 RAGAS `context_recall = 0.5667` 偏低。这需要判断：
**是 golden set 标注的问题，还是检索的问题？** 两者在论文里的写法完全不同 ——
前者要写「评估集偏差」，后者要写「已知局限」。

### 方法：三个口径互相对照（零 LLM 成本）

数据源：`evaluation_reports/cache/answers_smoke_after_refactor_*.json`
（上次 smoke 的真实答案 + 父块上下文 + ground_truth）。

| 口径 | 结果 | 怎么测的 |
|---|---|---|
| 法条引用层面 | **78.6%** | ground_truth 引用的 14 条法条，11 条出现在检索上下文里 |
| 内容词覆盖 | **91.2%** | jieba 分词后逐句比对内容词（排除停用词） |
| RAGAS context_recall | **56.67%** | 上次评估报告 |

### 结论

**检索上下文里确实包含 ground_truth 的内容**（91% 的内容词都在，
法条层面 78.6% 覆盖）——所以低分**主要不是「检索没检到」，也不是标注与语料不符**。

### 已确认的真缺陷（Bug 20）：对比题只评估一半证据

```
retrieve_for_compare  →  {"context_docs": docs_a, "context_docs_b": docs_b}
rag_engine.py:157     →  relevant_docs = result["context_docs"]      ← 只取 A 组
                        sources / full_contexts / child_contexts 全部由它派生
```

后果：

- 对比题的**检索指标**只统计 A 组（B 组的命中不计入）
- 对比题的 **RAGAS faithfulness / context_precision / context_recall 也只看到 A 组**，
  而答案是 A+B 两组共同生成的 —— 来自 B 组的正确主张会被判为「上下文中无依据」，
  于是 **faithfulness 被低估、幻觉率被高估**
- Golden Set 20 题中有 3 题是对比类（15%）

**修与不修都会改变论文里已有的数字，因此先记录、由作者决定。**

### 91% vs 57% 的剩余差距

现有数据无法定位 —— 评估报告只存了**逐题 faithfulness**，没存逐题 context_recall。
可能的原因（按可能性排序）：

1. **判分模型偏严**（RAGAS 槽 = `deepseek-v4.1-flash`）—— 要求上下文**显式陈述**
   该断言，而不仅是包含相同词汇
2. **RAGAS 用的是 child contexts**，而本次分析用的是 parent contexts（前者是后者的切分）；
   实际 child contexts 未被缓存，无法核对
3. 其他口径问题

**低成本定位手段**：只跑 `context_recall` 一项、只用这 5 题的缓存数据
→ 约 **5 次 LLM 调用 / ~20k token**（对比全量 smoke 约 200 次调用、13 分钟）。
这是能给出逐题分数、直接回答「标注 vs 检索」的最小实验。
