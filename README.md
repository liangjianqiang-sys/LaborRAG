# LaborRAG - 劳动法智能问答系统

基于 RAG（检索增强生成）技术的劳动法领域智能问答系统。用户输入劳动法问题，系统自动检索法条、生成专业回答，并支持金额计算、法条对比、多轮追问。内置三元组评估体系（RAGAS + 检索指标 + 响应指标）量化回答质量。

## 系统功能

### 💬 智能问答（三种问题类型自动识别）

| 问题类型 | 示例 | 处理方式 |
|---------|------|---------|
| **法条查询** | "加班工资怎么算？" "试用期最长多久？" | 检索法条 → LLM基于资料生成回答 |
| **金额计算** | "月薪8000加班10小时加班费多少？" | 识别参数 → Python公式精确计算 → 法条依据 |
| **对比分析** | "经济补偿金和赔偿金有什么区别？" | 双组检索 → 结构化对比表 |

- **来源追溯**：每个回答附带参考法条来源、相关度分数，可追溯可验证
- **多轮对话**：支持追问，历史上下文自动注入（如先问"经济补偿金怎么算"，再问"5年月薪6000能拿多少"）
- **缺参引导**：计算类问题缺少参数时，返回公式框架+法条依据+缺失参数提示，而非直接拒绝
- **幻觉防护**：统一5条严格规则禁止编造法条，Validator验证引用法条是否在资料中，未通过自动追加警告

### 📚 知识库管理

- **文档上传**：支持 PDF、DOCX、TXT、MD 格式
- **一键构建**：上传后自动法条结构化切分 → 向量化 → 入库
- **法条切分**：按"第X条"切分保持完整性，子块精准检索→自动提升为父块提供完整上下文
- **元数据过滤**：查询含法条编号（如"劳动合同法第47条"）时精准命中，无需搜索全库

### 📊 三元组评估体系

- **三元组核心**：context_relevancy（上下文相关性）/ faithfulness（忠实度）/ answer_relevancy（答案相关性）
- **检索指标**：Precision@5 / Recall@5 / F1@5 / MRR / MAP，基于 relevant_articles 标注匹配
- **响应指标**：ROUGE-L / BLEU / 幻觉率 / 完整性（LLM评估）
- **RAGAS原始指标**：faithfulness / answer_relevancy / context_precision / context_recall
- **分类型统计**：按 retrieve/calculate/compare 三类分别统计指标
- **可观测性面板**：检索质量 / 生成质量 / 业务指标 三维度概览
- **Bad Case追踪**：得分最低题目快速定位问题
- **每题详情**：展开查看每道题的回答、参考答案、检索/响应指标
- **异步执行**：后台线程运行，前端轮询进度，离开页面再回来可恢复状态
- **对比报告**：多次评估结果自动对比，量化不同策略的改进效果

### 🔧 检索与生成技术栈

| 层级 | 技术 | 作用 |
|------|------|------|
| 混合检索 | 向量(FAISS) + BM25(jieba) + RRF融合 | 语义+关键词双路检索 |
| 重排序 | BGE-Reranker-v2-m3 Cross-Encoder | Top-K精排 |
| 上下文压缩 | EmbeddingsFilter | 过滤低相关文档，Token -20~30% |
| 查询增强 | HyDE（短查询≤15字触发） | 生成假设答案提升检索精度 |
| 自适应路由 | Adaptive RAG | 简单查询省40%+token，复杂查询走完整纠错链路 |
| 自我纠错 | CRAG | 检索评估→HyDE→生成→忠实度评估闭环 |
| 多Agent协作 | Router→Retriever/Calculator/Comparator→Validator | 意图分发+专业处理+质量把关 |
| 降级兜底 | 异常时自动降级到simple模式 | 保证系统可用性 |

### 🖥️ 前端界面

- **四页面布局**：问答 / 知识库 / 评估 / 设置，顶部导航切换
- **CRAG步骤可视化**：折叠面板实时展示工作流7-10个执行步骤
- **参考来源折叠**：默认收起，点击展开查看详情
- **法律免责声明**：界面明确提示回答仅供参考

## 覆盖法律

| 法律文件 | 说明 |
|---------|------|
| 《劳动法》 | 劳动关系基本法 |
| 《劳动合同法》 | 劳动合同制度 |
| 《劳动争议调解仲裁法》 | 劳动争议处理 |
| 《社会保险法》 | 社会保障制度 |
| 《工伤保险条例》 | 工伤认定与赔偿 |
| 《职工带薪年休假条例》 | 年休假制度 |

## 技术架构

```
用户问题
    │
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

检索链路: HyDE查询增强 → Hybrid Search(向量+BM25+RRF)
        → Reranker精排 → 上下文压缩 → 父子分块提升
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | 百炼 qwen3.7-max（生成）/ qwen3.6-plus（评估） | 大语言模型 |
| Embedding | BGE-M3 (本地) | 中文向量模型，文本向量化 |
| Reranker | BGE-Reranker-v2-m3 | Cross-Encoder重排序模型 |
| 向量库 | FAISS | 向量存储与检索 |
| 关键词检索 | BM25 + jieba | 中文分词关键词匹配 |
| RAG框架 | LangChain 1.0 + LCEL | V1/V2 检索生成链式编排 |
| CRAG框架 | LangGraph | V3 状态图自我纠错工作流 |
| Agent框架 | LangGraph | V4 多Agent协作工作流 |
| 评估 | RAGAS + 自研指标 | 三元组评估体系（检索/响应/RAGAS） |
| 后端 | FastAPI + Uvicorn | API服务 |
| 前端 | React + TypeScript + TailwindCSS + React Router | 多页面用户界面 |

## 项目结构

```
LaborRAG/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI入口
│   │   ├── config.py            # 配置管理（含V2检索策略配置）
│   │   ├── core/
│   │   │   ├── embeddings.py    # Embedding模型初始化
│   │   │   ├── document_loader.py # 文档加载
│   │   │   ├── vectorstore.py   # FAISS向量库管理（支持切分策略）
│   │   │   ├── text_splitter.py # V2 法条结构化切分器
│   │   │   ├── retriever/       # 检索层
│   │   │   │   ├── base.py      # 抽象基类
│   │   │   │   ├── vector.py    # V1 向量检索
│   │   │   │   ├── bm25.py      # V2 BM25关键词检索
│   │   │   │   ├── hybrid.py    # V2 混合检索（RRF融合）
│   │   │   │   ├── reranked.py  # V2 重排序检索
│   │   │   │   └── compressor.py # V3 EmbeddingsFilter上下文压缩
│   │   │   ├── generator/       # 生成层
│   │   │   │   ├── base.py      # 抽象基类
│   │   │   │   ├── simple_chain.py # V1/V2 LCEL链
│   │   │   │   ├── crag_graph.py  # V3 Adaptive RAG状态图
│   │   │   │   └── agent_graph.py  # V4 Agentic RAG多Agent
│   │   │   └── rag_engine.py    # RAG引擎（自动选择检索策略）
│   │   ├── evaluation/          # 三元组评估体系
│   │   │   ├── utils.py          # 公共工具(sanitize_floats/strip_per_query)
│   │   │   ├── eval_dataset.py  # 24个标注问答对(含question_type+relevant_articles)
│   │   │   ├── eval_runner.py   # 三阶段执行器(生成→检索指标→RAGAS+响应指标)
│   │   │   ├── eval_report.py   # 评估报告+三维度对比+BadCase+可观测性
│   │   │   ├── retrieval_metrics.py  # 检索指标(P@k/R@k/F1/MRR/MAP)
│   │   │   └── response_metrics.py   # 响应指标(ROUGE-L/BLEU/幻觉率/完整性)
│   │   ├── api/
│   │   │   └── routes.py       # API路由（12个端点，含可观测性+BadCase）
│   │   └── models/
│   │       └── schemas.py       # Pydantic数据模型
│   ├── run.py                   # 后端启动入口
│   ├── requirements.txt         # Python依赖
│   ├── .env.example             # 环境变量模板
│   └── .env                     # 环境变量（需自行配置）
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # 多页面路由布局
│   │   ├── api.ts               # API客户端（含V2评估接口）
│   │   ├── main.tsx             # React入口
│   │   ├── pages/               # 页面组件
│   │   │   ├── ChatPage.tsx     # 问答页面
│   │   │   ├── KnowledgePage.tsx # 知识库管理页面
│   │   │   ├── EvalPage.tsx     # 三元组评估页面(可观测性+BadCase)
│   │   │   └── SettingsPage.tsx  # 系统设置页面
│   │   └── components/          # 通用组件
│   │       ├── ChatWindow.tsx   # 聊天窗口（含CRAG步骤面板）
│   │       ├── MessageBubble.tsx # 消息气泡（参考来源折叠）
│   │       └── InputBar.tsx     # 输入框
│   ├── run.py                   # 前端启动入口
│   ├── package.json             # 前端依赖
│   └── vite.config.ts           # Vite配置
├── data/
│   └── labor_laws/              # 劳动法文档（6部）
├── KNOWLEDGE.md                 # 项目知识文档
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
| POST | `/api/v1/chat` | 智能问答 |
| GET | `/api/v1/knowledge-base/status` | 知识库状态 |
| POST | `/api/v1/knowledge-base/build` | 构建/重建知识库 |
| POST | `/api/v1/documents/upload` | 上传文档 |
| GET | `/api/v1/documents/list` | 文档列表 |
| DELETE | `/api/v1/documents/{filename}` | 删除文档 |
| POST | `/api/v1/evaluation/run` | 启动评估（后台异步，三阶段执行） |
| GET | `/api/v1/evaluation/status` | 查询评估任务状态 |
| GET | `/api/v1/evaluation/report` | 获取三维度对比报告 |
| GET | `/api/v1/evaluation/observability` | 可观测性三维度概览 |
| GET | `/api/v1/evaluation/bad-cases` | Bad Case列表 |

## 检索策略配置

在 `.env` 中通过 `RETRIEVER_TYPE` 切换检索策略：

| 值 | 策略 | 说明 |
|----|------|------|
| `vector` | V1 纯向量检索 | 语义相似度匹配 |
| `hybrid` | V2 混合检索 | 向量 + BM25 + RRF融合 |
| `reranked` | V2 重排序检索 | 混合检索 + Cross-Encoder精排 |

相关配置项：
```env
RETRIEVER_TYPE=reranked     # 检索策略（推荐reranked）
VECTOR_WEIGHT=0.7           # 向量检索权重（法律文本语义匹配更关键）
BM25_WEIGHT=0.3             # BM25检索权重
RRF_K=60                    # RRF融合常数
CHUNK_STRATEGY=law_article  # 切分策略: recursive | law_article
PARENT_CHILD_ENABLED=true   # 层次化父子分块（子块精准检索→父块完整上下文）
CHUNK_SIZE=800              # 切分长度
CHUNK_OVERLAP=100           # 切分重叠
TOP_K=8                     # 检索返回数量
SCORE_THRESHOLD=0.3         # 相似度阈值
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3  # 重排序模型
RERANK_TOP_K=5              # 重排序返回数量
COMPRESSION_THRESHOLD=0.5    # 上下文压缩相似度阈值（低于此值丢弃）
RAG_MODE=agent               # RAG模式: simple | crag | agent
```

## 渐进式开发路线

每版解决上一版的核心问题，逐步提升系统能力：

| 版本 | 解决的问题 | 新增能力 | 状态 |
|------|-----------|---------|------|
| **V1** | 从0到1搭建RAG基础 | 向量检索 + LCEL链式生成 + 知识库管理 | ✅ 通过 |
| **V2** | V1纯向量检索精度不足 | 混合检索 + 重排序 + 法条切分 + 三元组评估 | ✅ 通过 |
| **V3** | V2无纠错能力，检索错误直接传给LLM | CRAG自我纠错 + HyDE增强 + 上下文压缩 + 多轮对话 | ✅ 通过 |
| **V4** | V3无法处理计算/对比类问题 | 多Agent协作（意图路由+专业处理+质量验证） | ✅ 待测试 |

### V3 端到端测试成功标准
✅ CRAG工作流完整闭环
✅ 检索评估准确（precise/vague/irrelevant 分类正确）
✅ HyDE查询增强有效（短查询生成假设答案，检索精度提升）
✅ 父子分块工作正常（子块精准检索→父块完整上下文）
✅ 元数据过滤生效（法条编号查询精准命中）
✅ 忠实度评估工作（检测不忠实回答并重新生成）
✅ 上下文压缩生效（EmbeddingsFilter过滤低相关文档，减少token消耗）
✅ 多轮对话支持（上下文正确注入）
✅ 前端步骤展示（CRAG工作流7-10个步骤可视化）
✅ 参考来源完整（带评分和法律出处）

## V4 端到端测试成功标准
⬜ Router意图分类准确（法条查询/计算类/对比类正确分发）
⬜ Calculator精确计算（加班费/经济补偿金/赔偿金等，结果精确；缺参时返回公式框架）
⬜ Comparator对比分析（双组检索+结构化对比表）
⬜ Retriever路径正常（走V3检索+压缩链路，prompt统一4条规则防幻觉）
⬜ Validator验证生效（检测幻觉+法条不在资料中并修正）
⬜ 多Agent协作完整（Router→专业Agent→Validator全链路）
⬜ Agent降级兜底（异常时自动降级到simple模式）
⬜ 评估系统异步运行（后台线程+前端轮询+页面恢复状态）
⬜ V1-V4四版对比实验

## 注意事项

- FAISS 底层 C++ 不支持中文路径，向量库默认存储在 `D:/LaborRAG_data/vector_store`
- BGE-M3 模型首次运行需下载约 2GB，后续从本地缓存加载
- BGE-Reranker-v2-m3 首次运行需下载约 560MB，后续从本地缓存加载
- 国内环境建议设置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载
- CRAG 模式每次提问约消耗 3000-4000 tokens（含3-4次LLM调用），简单模式约 2500 tokens
- Agent 模式根据意图不同消耗不同：retrieve≈3000, calculate≈4000, compare≈5000 tokens
- 评估对比实验将在V4完成后统一运行（V1 vs V2 vs V3 vs V4 四版对比）
- 本系统回答仅供参考，不构成法律意见，具体问题请咨询专业律师
- 详细技术知识参见 [KNOWLEDGE.md](KNOWLEDGE.md)
