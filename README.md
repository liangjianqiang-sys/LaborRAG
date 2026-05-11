# LaborRAG - 劳动法智能问答系统

基于 RAG（检索增强生成）技术的劳动法领域智能问答系统，采用渐进式开发路线，从基础向量检索到多Agent协作逐步演进。前端采用多页面布局（问答/知识库/评估/设置），各功能模块独立展示。

## 功能特性

### V1 基础版 ✅
- **智能问答**：输入劳动法相关问题，系统自动检索相关法条并由大模型生成专业回答
- **来源追溯**：每个回答附带参考的法律条文来源、相关度分数，可追溯可验证
- **知识库管理**：支持文档上传、列表查看、删除，一键构建/重建向量知识库
- **多格式支持**：支持 PDF、DOCX、TXT、MD 格式文档
- **法律免责声明**：界面明确提示回答仅供参考，降低法律风险

### V2 增强版 ✅
- **Hybrid Search 混合检索**：向量语义检索 + BM25关键词检索 + RRF融合，兼顾语义理解与精确匹配
- **Reranker 重排序**：BGE-Reranker-v2-m3 Cross-Encoder精排，提升Top-K精度
- **法条结构化切分**：按法律条文（第X条）切分，保持法条完整性，避免跨条切割
- **RAGAS 评估体系**：faithfulness / answer_relevancy / context_precision / context_recall 四维量化评估
- **评估对比报告**：不同检索策略（vector / hybrid / reranked）自动对比，量化改进效果
- **多页面布局**：问答/知识库/评估/设置四页面独立展示，顶部导航切换
- **Prompt强化**：严格限制模型忠实于检索内容，禁止编造法律条文

### V3 高级版 ✅
- **LangGraph CRAG 自我纠错**：检索评估 → Query改写 → 生成 → 忠实度评估，闭环纠错
- **检索三分法评估**：precise / vague / irrelevant，vague触发Query改写重新检索
- **Query改写**：模糊问题自动改写为精确专业查询（如"工资怎么算"→"加班工资计算标准"）
- **回答忠实度评估**：判断回答是否忠实于参考资料，不忠实则重新生成
- **多轮对话**：支持上下文追问，代词指代自动解析
- **CRAG降级机制**：工作流失败时自动降级到simple模式，保证可用性
- **CRAG步骤可视化**：前端折叠面板展示工作流每步执行情况
- **参考来源折叠**：默认收起，点击展开，界面更清爽

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
┌─────────────┐     ┌─────────────┐     ┌──────────────────────────────────────┐
│   React     │────▶│   FastAPI    │────▶│           RAG Engine                 │
│  Frontend   │     │   Backend   │     │  ┌─────────────────────────────┐    │
│  + Tailwind │     │  + CORS     │     │  │    Retriever (可替换)        │    │
└─────────────┘     └─────────────┘     │  │  V1: Vector (FAISS)         │    │
                                        │  │  V2: Hybrid (Vector+BM25)   │    │
                                        │  │  V2: Reranked (+CrossEnc)   │    │
                                        │  └────────────┬────────────────┘    │
                                        │  ┌────────────▼────────────────┐    │
                                        │  │    Generator (LLM API)      │    │
                                        │  └─────────────────────────────┘    │
                                        │  ┌─────────────────────────────┐    │
                                        │  │  BGE-M3 Embedding (本地)     │    │
                                        │  └─────────────────────────────┘    │
                                        └──────────────────────────────────────┘
```

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | 百炼 qwen3.5-flash | 大语言模型，生成回答+CRAG评估 |
| Embedding | BGE-M3 (本地) | 中文向量模型，文本向量化 |
| Reranker | BGE-Reranker-v2-m3 | Cross-Encoder重排序模型 |
| 向量库 | FAISS | 向量存储与检索 |
| 关键词检索 | BM25 + jieba | 中文分词关键词匹配 |
| RAG框架 | LangChain 1.0 + LCEL | V1/V2 检索生成链式编排 |
| CRAG框架 | LangGraph | V3 状态图自我纠错工作流 |
| 评估 | RAGAS | RAG系统量化评估 |
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
│   │   │   │   └── reranked.py  # V2 重排序检索
│   │   │   ├── generator/       # 生成层
│   │   │   │   ├── base.py      # 抽象基类
│   │   │   │   ├── simple_chain.py # V1/V2 LCEL链
│   │   │   │   └── crag_graph.py  # V3 CRAG状态图
│   │   │   └── rag_engine.py    # RAG引擎（自动选择检索策略）
│   │   ├── evaluation/          # V2 RAGAS评估
│   │   │   ├── eval_dataset.py  # 20个标注问答对
│   │   │   ├── eval_runner.py   # 评估执行器
│   │   │   └── eval_report.py   # 评估报告+对比
│   │   ├── api/
│   │   │   └── routes.py       # API路由（8个端点，/chat同步线程池运行）
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
│   │   │   ├── EvalPage.tsx     # RAGAS评估页面
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
| POST | `/api/v1/evaluation/run` | 运行RAGAS评估 |
| GET | `/api/v1/evaluation/report` | 获取评估对比报告 |

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
VECTOR_WEIGHT=0.5           # 向量检索权重
BM25_WEIGHT=0.5             # BM25检索权重
RRF_K=60                    # RRF融合常数
CHUNK_STRATEGY=law_article  # 切分策略: recursive | law_article
CHUNK_SIZE=800              # 切分长度
CHUNK_OVERLAP=100           # 切分重叠
TOP_K=8                     # 检索返回数量
SCORE_THRESHOLD=0.2         # 相似度阈值
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3  # 重排序模型
RERANK_TOP_K=5              # 重排序返回数量
RAG_MODE=crag               # RAG模式: simple | crag
```

## 渐进式开发路线

| 版本 | 核心特性 | 状态 | 毕设等级 |
|------|---------|------|---------|
| V1 基础版 | 纯向量检索 + LCEL链式调用 | ✅ 已完成 | 合格 |
| V2 增强版 | Hybrid Search + Reranker + 法条切分 + RAGAS评估 + 多页面布局 | ✅ 已完成 | 良好 |
| V3 高级版 | LangGraph CRAG + 三分法评估 + Query改写 + 多轮对话 + 降级机制 | ✅ 已完成 | 优秀 |
| V4 多Agent版 | Router/Retriever/Calculator/Validator多Agent协作 | ⬜ 待开发 | 优秀+ |

## 注意事项

- FAISS 底层 C++ 不支持中文路径，向量库默认存储在 `D:/LaborRAG_data/vector_store`
- BGE-M3 模型首次运行需下载约 2GB，后续从本地缓存加载
- BGE-Reranker-v2-m3 首次运行需下载约 560MB，后续从本地缓存加载
- 国内环境建议设置 `HF_ENDPOINT=https://hf-mirror.com` 加速模型下载
- CRAG 模式每次提问约消耗 3000-4000 tokens（含3-4次LLM调用），简单模式约 2500 tokens
- 本系统回答仅供参考，不构成法律意见，具体问题请咨询专业律师
- 详细技术知识参见 [KNOWLEDGE.md](KNOWLEDGE.md)
