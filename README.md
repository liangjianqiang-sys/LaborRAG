# LaborRAG - 劳动法智能问答系统

基于 Agentic RAG（检索增强生成）技术的劳动法领域智能问答系统。用户输入劳动法问题，系统自动检索法条、生成专业回答，并支持金额计算、法条对比、多轮追问。内置三元组评估体系（RAGAS + 检索指标 + 响应指标）量化回答质量。

## 核心评估指标（20题 Full Set）

| 指标 | 得分 | 说明 |
|------|------|------|
| **精确率 P@1** | 90.0% | 首位命中的法条准确率 |
| **召回率 R@5** | 92.5% | 前5条中相关法条的召回率 |
| **忠实度 Faithfulness** | 92.0% | 回答内容与检索资料的一致性 |
| **幻觉率 Hallucination** | 8.0% | 回答中无依据内容的比例 |
| **完整性 Completeness** | 89.5% | 回答对参考答案的覆盖程度 |

## 系统功能

### 💬 智能问答（三种问题类型自动识别）

| 问题类型 | 示例 | 处理方式 |
|---------|------|---------|
| **法条查询** | "加班工资怎么算？" "试用期最长多久？" | 检索法条 → LLM基于资料生成回答 |
| **金额计算** | "月薪8000加班10小时加班费多少？" | 识别参数 → Python公式精确计算 → 法条依据 |
| **对比分析** | "经济补偿金和赔偿金有什么区别？" | 双组检索 → 结构化对比表 |

- **来源追溯**：每个回答附带参考法条来源、相关度分数，可追溯可验证
- **多轮对话**：支持追问，上下文感知查询改写自动补全追问为完整问题
- **对话历史持久化**：前端 localStorage + 后端 JSON 文件双层持久化，后端重启不丢失上下文
- **缺参引导**：计算类问题缺少参数时，返回公式框架+法条依据+缺失参数提示
- **幻觉防护**：统一规则禁止编造法条 + Validator验证法条真实性 + 输出护栏风险提示
- **超纲拒答**：非劳动法领域问题触发拒答，避免强行解释
- **人性化回答**：13条风格规则（先结论→通俗解释→克制补充→逐句标注来源→法条竞合处理等）

### 📚 知识库管理

- **文档上传**：支持 PDF、DOCX、TXT、MD 格式
- **一键构建**：上传后自动法条结构化切分 → 向量化 → 入库
- **法条切分**：按"第X条"切分保持完整性，子块精准检索→自动提升为父块提供完整上下文
- **元数据过滤**：查询含法条编号（如"劳动合同法第47条"）时精准命中，无需搜索全库

### 📊 三元组评估体系

- **RAGAS指标**：faithfulness（忠实度）/ context_precision（上下文精确度）/ context_recall（上下文召回率）
- **检索指标**：P@1 / P@3 / P@5 / R@5 / MRR，基于 relevant_articles 标注匹配
- **响应指标**：ROUGE-L / BLEU / 幻觉率 / 完整性（LLM评估）
- **双轨制测试**：Smoke Set（5题快速验证）+ Full Set（20题完整评估），Smoke ⊂ Full
- **分维度统计**：按 question_type（retrieve/calculate/compare）+ difficulty（easy/medium/hard）分组
- **可观测性面板**：检索质量 / 生成质量 / 业务指标 三维度概览
- **Bad Case追踪**：得分最低题目快速定位问题
- **并发评估**：答案生成阶段并发执行（可配置并发度），显著缩短评估时间
- **持久化断点续评**：每题结果实时保存，支持崩溃/重启后手动续评
- **任务全生命周期管理**：创建/启动/继续/重评/停止/删除
- **超时保护**：任务级超时自动标记失败，避免永久卡住

### 🔧 检索与生成技术栈

| 层级 | 技术 | 作用 |
|------|------|------|
| 混合检索 | 向量(FAISS) + BM25(jieba) + RRF融合 | 语义+关键词双路检索 |
| 重排序 | BGE-Reranker-v2-m3 Cross-Encoder | Top-K精排 |
| 概念注入提升 | 知识图谱概念映射 + 强制置顶 | 规则匹配高置信度法条提升至首位，P@1从20%→90% |
| 查询增强 | HyDE（短查询≤15字触发）+ 上下文感知改写 | 生成假设答案提升检索精度 / 追问自动补全 |
| 自适应路由 | Adaptive RAG | 简单查询省40%+token，复杂查询走完整纠错链路 |
| 自我纠错 | CRAG | 检索评估→HyDE→生成→忠实度评估闭环 |
| 多Agent协作 | Router→Retriever/Calculator/Comparator→Validator | 意图分发+专业处理+质量把关 |
| 降级兜底 | 异常时自动降级到simple模式 | 保证系统可用性 |

### 🖥️ 前端界面

- **四页面布局**：问答 / 知识库 / 评估 / 设置，顶部导航切换
- **历史对话侧边栏**：按日期分组、可折叠、支持新建/切换/删除对话
- **CRAG步骤可视化**：折叠面板实时展示工作流7-10个执行步骤
- **参考来源折叠**：默认收起，点击展开查看详情
- **法律免责声明**：界面明确提示回答仅供参考

## 覆盖法律（17部）

| 法律文件 | 说明 |
|---------|------|
| 《劳动法》 | 劳动关系基本法 |
| 《劳动合同法》 | 劳动合同制度 |
| 《劳动合同法实施条例》 | 劳动合同法实施细则 |
| 《劳动争议调解仲裁法》 | 劳动争议处理 |
| 《社会保险法》 | 社会保障制度 |
| 《工伤保险条例》 | 工伤认定与赔偿 |
| 《职工带薪年休假条例》 | 年休假制度 |
| 《女职工劳动保护特别规定》 | 女职工权益保护 |
| 《最低工资规定》 | 最低工资标准 |
| 《工资支付暂行规定》 | 工资支付规范 |
| 《失业保险条例》 | 失业保险制度 |
| 《职业病防治法》 | 职业病防治 |
| 《住房公积金管理条例》 | 住房公积金管理 |
| 《最高人民法院司法解释（一）》 | 劳动争议审判规则 |
| 《最高人民法院司法解释（二）》 | 劳动争议审判规则 |
| 《职工非因工伤残或因病丧失劳动能力程度鉴定标准》 | 劳动能力鉴定 |
| 《法条适用前提速查表》 | 法条适用条件索引 |

## 技术架构

```
用户问题
    │
    ▼
┌──── Query Rewriter ────┐
│  上下文感知查询改写      │
│  追问补全为完整问题      │
└────────┬──────────────┘
         ▼
┌─────────── Router Agent ───────────┐
│  LLM意图分类: retrieve/calculate/compare  │
└───┬──────────┬──────────┬───────────┘
    │          │          │
    ▼          ▼          ▼
 Retriever   Calculator  Comparator
 (检索+生成)  (公式计算)   (双组对比)
    │          │          │
    └────┬─────┴──────────┘
         ▼
    Validator Agent
    (法条真实性+幻觉验证)
         │
         ▼
       回答

检索链路: 上下文改写 → HyDE查询增强 → Hybrid Search(向量+BM25+RRF)
        → Reranker精排 → 概念注入提升 → 父子分块提升

概念提升: 知识图谱概念映射 → 匹配法条强制置顶 → P@1从20%提升至90%

记忆链路: 前端localStorage(长期) + 后端JSON持久化(长期) + 前端传history(兜底)
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | 百炼 qwen3.7-max（生成）/ qwen3.5-plus（RAGAS评估） | 大语言模型 |
| Embedding | BGE-M3 (本地CPU) | 中文向量模型，文本向量化 |
| Reranker | BGE-Reranker-v2-m3 | Cross-Encoder重排序模型 |
| 向量库 | FAISS | 向量存储与检索 |
| 关键词检索 | BM25 + jieba | 中文分词关键词匹配 |
| 知识图谱 | 法条概念映射 + 伴生法条注入 | 领域知识增强检索 |
| Agent框架 | LangGraph | Agentic RAG 多Agent协作工作流 |
| 评估 | RAGAS + 自研指标 | 三元组评估体系（检索/响应/RAGAS） |
| 对话持久化 | JSON文件 (D:/LaborRAG_data/conversations/) | 后端对话历史持久化 |
| 后端 | FastAPI + Uvicorn | API服务 |
| 前端 | React + TypeScript + TailwindCSS + React Router | 多页面用户界面 |

## 项目结构

```
LaborRAG/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI入口 + 模型预加载
│   │   ├── config.py            # 配置管理
│   │   ├── core/
│   │   │   ├── embeddings.py    # Embedding模型初始化
│   │   │   ├── document_loader.py # 文档加载
│   │   │   ├── vectorstore.py   # FAISS向量库管理
│   │   │   ├── text_splitter.py # 法条结构化切分器
│   │   │   ├── conversation.py  # 对话历史管理器（JSON持久化）
│   │   │   ├── retriever/       # 检索层
│   │   │   │   ├── base.py      # 抽象基类
│   │   │   │   ├── vector.py    # 纯向量检索
│   │   │   │   ├── bm25.py      # BM25关键词检索
│   │   │   │   ├── hybrid.py    # 混合检索（RRF融合）
│   │   │   │   ├── reranked.py  # 重排序检索
│   │   │   │   ├── sparse.py    # 稀疏检索
│   │   │   │   ├── law_graph.py # 知识图谱（概念映射+伴生法条注入）
│   │   │   │   ├── law_dict.txt # 法律词典
│   │   │   │   └── query_enhance.py # 查询增强
│   │   │   ├── generator/       # 生成层
│   │   │   │   ├── agent_graph.py  # Agentic RAG多Agent工作流
│   │   │   │   ├── prompts.py    # 共享回答风格规则
│   │   │   │   ├── guardrails.py # 输出护栏（评估模式跳过）
│   │   │   │   ├── calculators.py # 计算引擎（加班费/经济补偿/赔偿金等）
│   │   │   │   ├── helpers.py    # 生成辅助函数
│   │   │   │   ├── constants.py  # 常量定义
│   │   │   │   ├── state.py     # Agent状态定义
│   │   │   │   └── nodes/       # Agent节点
│   │   │   │       ├── rewrite.py   # 查询改写
│   │   │   │       ├── router.py    # 意图路由
│   │   │   │       ├── retrieve.py  # 检索（含概念注入提升）
│   │   │   │       ├── generate.py  # 生成（含概念注入提升）
│   │   │   │       ├── calculate.py # 计算
│   │   │   │       ├── compare.py   # 对比
│   │   │   │       ├── classify.py  # 分类
│   │   │   │       └── validate.py  # 验证
│   │   │   └── rag_engine.py    # RAG引擎
│   │   ├── evaluation/          # 三元组评估体系
│   │   │   ├── utils.py          # 公共工具
│   │   │   ├── eval_dataset.py  # Golden Set标注数据（20题，双轨制）
│   │   │   ├── eval_runner.py   # 三阶段执行器
│   │   │   ├── eval_persistent.py # 断点续评管理器
│   │   │   ├── eval_report.py   # 评估报告
│   │   │   ├── retrieval_metrics.py  # 检索指标(P@1/P@3/P@5/R@5/MRR)
│   │   │   └── response_metrics.py   # 响应指标(ROUGE-L/BLEU/幻觉率/完整性)
│   │   ├── api/
│   │   │   └── routes.py       # API路由
│   │   └── models/
│   │       └── schemas.py       # Pydantic数据模型
│   ├── run.py                   # 后端启动入口
│   ├── run_eval.py              # CLI评估脚本
│   ├── requirements.txt         # Python依赖
│   └── .env                     # 环境变量（需自行配置）
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # 多页面路由布局
│   │   ├── api.ts               # API客户端
│   │   ├── main.tsx             # React入口
│   │   ├── index.css            # 全局样式+品牌色+动画
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx     # 问答页面
│   │   │   ├── KnowledgePage.tsx # 知识库管理页面
│   │   │   ├── EvalPage.tsx     # 评估页面
│   │   │   └── SettingsPage.tsx # 系统设置页面
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx   # 聊天窗口
│   │   │   ├── ConversationSidebar.tsx # 对话侧边栏
│   │   │   ├── MessageBubble.tsx # 消息气泡
│   │   │   └── InputBar.tsx     # 输入框
│   │   └── hooks/
│   │       └── useConversations.ts # 对话状态管理
│   ├── run.py                   # 前端启动入口
│   ├── package.json             # 前端依赖
│   └── vite.config.ts           # Vite配置
├── data/
│   └── labor_laws/              # 劳动法文档（17部）
├── D:/LaborRAG_data/            # 运行时数据（独立于项目目录）
│   ├── vector_store/            # FAISS向量库
│   ├── conversations/           # 对话历史JSON文件
│   └── evaluation/              # 评估结果持久化
└── README.md
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Conda（推荐）

### 1. 配置后端

```bash
# 创建Conda环境
conda create -n RAG python=3.10
conda activate RAG

# 安装Python依赖
cd backend
pip install -r requirements.txt

# 配置环境变量
copy .env.example .env
# 编辑 .env，填入你的百炼API Key
```

### 2. 配置前端

```bash
cd frontend
npm install
```

### 3. 启动服务

**终端1 - 启动后端：**
```bash
conda activate RAG
python backend/run.py
```

**终端2 - 启动前端：**
```bash
python frontend/run.py
```

### 4. 访问系统

浏览器打开 http://localhost:5173

## API 接口

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/api/v1/chat` | 智能问答（支持多轮对话） |
| GET | `/api/v1/knowledge-base/status` | 知识库状态 |
| POST | `/api/v1/knowledge-base/build` | 构建/重建知识库 |
| POST | `/api/v1/documents/upload` | 上传文档 |
| GET | `/api/v1/documents/list` | 文档列表 |
| DELETE | `/api/v1/documents/{filename}` | 删除文档 |
| POST | `/api/v1/evaluation/run` | 启动评估（后台异步） |
| GET | `/api/v1/evaluation/status` | 查询评估状态 |
| GET | `/api/v1/evaluation/report` | 获取评估报告 |
| POST | `/api/v1/evaluation/persistent/create` | 创建断点续评任务 |
| POST | `/api/v1/evaluation/persistent/start` | 启动任务 |
| POST | `/api/v1/evaluation/persistent/resume` | 断点续评 |
| POST | `/api/v1/evaluation/persistent/restart` | 重新评估 |
| POST | `/api/v1/evaluation/persistent/force-stop` | 强制停止任务 |
| DELETE | `/api/v1/evaluation/persistent/delete` | 删除任务 |
| GET | `/api/v1/evaluation/persistent/tasks` | 任务列表 |
| GET | `/api/v1/evaluation/persistent/report` | 任务评估报告 |

## 检索策略配置

在 `.env` 中通过 `RETRIEVER_TYPE` 切换检索策略：

| 值 | 策略 | 说明 |
|----|------|------|
| `vector` | 纯向量检索 | 语义相似度匹配 |
| `hybrid` | 混合检索 | 向量 + BM25 + RRF融合 |
| `reranked` | 重排序检索 | 混合检索 + Cross-Encoder精排 |

核心配置项：
```env
RETRIEVER_TYPE=hybrid          # 检索策略
VECTOR_WEIGHT=0.4              # 向量检索权重
BM25_WEIGHT=0.6                # BM25检索权重
RRF_K=60                       # RRF融合常数
CHUNK_STRATEGY=law_article     # 切分策略: recursive | law_article
PARENT_CHILD_ENABLED=true      # 父子分块
TOP_K=12                       # 检索返回数量
SCORE_THRESHOLD=0.15           # 相似度阈值
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANK_TOP_K=8                 # 重排序返回数量
RERANK_SCORE_THRESHOLD=0.3     # 重排序分数阈值
RAG_MODE=agent                 # RAG模式: simple | crag | agent
RETRIEVAL_VALIDATION=false     # 两步生成验证
EVAL_MAX_CONCURRENT=2          # 评估并发度
RAGAS_MODEL_NAME=qwen3.5-plus-2026-04-20
RAGAS_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 渐进式开发路线

每版解决上一版的核心问题，逐步提升系统能力：

| 版本 | 解决的问题 | 新增能力 | 状态 |
|------|-----------|---------|------|
| **V1** | 从0到1搭建RAG基础 | 向量检索 + LCEL链式生成 + 知识库管理 | ✅ |
| **V2** | V1纯向量检索精度不足 | 混合检索 + 重排序 + 法条切分 + 三元组评估 | ✅ |
| **V3** | V2无纠错能力，检索错误直接传给LLM | CRAG自我纠错 + HyDE增强 + 上下文压缩 + 多轮对话 | ✅ |
| **V4** | V3无法处理计算/对比类问题 | 多Agent协作 + 上下文感知改写 + 人性化回答 | ✅ |
| **V5** | V4检索P@1仅20%，概念注入被reranker淹没 | 概念注入提升策略 + 知识图谱增强 + 评估体系完善 | ✅ |

### V5 关键改进

- **概念注入提升**：概念匹配的法条强制置顶，即使已被reranker检索到但排序靠后也提升至首位，P@1从20%→90%
- **知识图谱增强**：法条概念映射表 + 伴生法条注入，覆盖50+法律概念关键词
- **评估体系完善**：双轨制测试（Smoke/Full）+ 分维度统计 + P@1/P@3/P@5/MRR多粒度指标
- **Faithfulness修复**：contexts截断修复 + 护栏跳过 + 验证警告移除，幻觉率从61%降至8%

## 注意事项

- FAISS 底层 C++ 不支持中文路径，向量库默认存储在 `D:/LaborRAG_data/vector_store`
- 对话历史默认存储在 `D:/LaborRAG_data/conversations/`，每个会话一个JSON文件
- BGE-M3 模型首次运行需下载约 2GB，后续从本地缓存加载
- BGE-Reranker-v2-m3 首次运行需下载约 560MB，后续从本地缓存加载
- 国内环境建议设置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载
- Agent 模式根据意图不同消耗不同：retrieve≈3000, calculate≈4000, compare≈5000 tokens
- 评估命令：`python run_eval.py --subset smoke`（5题快速验证）或 `python run_eval.py --subset full`（20题完整评估）
- 评估并发度建议设为2-4，过高可能触发API限流（`EVAL_MAX_CONCURRENT`）
- 后端重启后未完成任务自动标记为「已暂停」，需手动点击「继续」恢复
- 本系统回答仅供参考，不构成法律意见，具体问题请咨询专业律师
