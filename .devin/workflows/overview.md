---
description: LaborRAG项目总览 - 劳动法智能问答系统渐进式开发路线图
---

# LaborRAG 项目总览

## 项目简介
- **名称**: LaborRAG - 劳动法智能问答系统
- **技术栈**: Python + FastAPI + LangChain 1.0 + React + TailwindCSS
- **核心**: 基于RAG（检索增强生成）的劳动法领域智能问答

## 技术方案
- **LLM**: DeepSeek / GLM-4-Flash (OpenAI兼容API)
- **Embedding**: BGE-M3 (本地中文向量模型)
- **向量库**: FAISS
- **后端**: FastAPI + LangChain LCEL
- **前端**: React + TypeScript + TailwindCSS + shadcn/ui
- **评估**: RAGAS

## 渐进式开发路线

### V1 基础版 (workflow: /v1-basic)
- 纯向量检索 + LCEL链式调用
- 知识库构建/文档管理/智能问答/来源追溯
- 聊天式Web界面
- **毕设等级**: 合格

### V2 增强版 (workflow: /v2-enhanced)
- 在V1基础上: Hybrid Search + Reranker + 结构化切分 + RAGAS评估
- **毕设等级**: 良好

### V3 高级版 (workflow: /v3-advanced)
- 在V2基础上: LangGraph CRAG + Query改写 + 多轮对话
- **毕设等级**: 优秀

### V4 多Agent版 (workflow: /v4-multiagent)
- 在V3基础上: 多Agent协作架构
  - Router Agent: 问题意图识别与路由分派
  - Retriever Agent: 检索+自我纠错(V3 CRAG)
  - Calculator Agent: 赔偿金/补偿金/加班费精确计算
  - Validator Agent: 回答合法性验证(最后一道防线)
- **毕设等级**: 优秀+

## 项目目录结构（规划）
```
LaborRAG/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI入口
│   │   ├── config.py             # 配置管理
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── retriever/        # 检索层（抽象接口）
│   │   │   │   ├── base.py       # BaseRetriever
│   │   │   │   ├── vector.py     # V1: 纯向量检索
│   │   │   │   ├── hybrid.py     # V2: 混合检索
│   │   │   │   └── reranked.py   # V2: 带重排序
│   │   │   ├── generator/        # 生成层（抽象接口）
│   │   │   │   ├── base.py       # BaseGenerator
│   │   │   │   ├── simple_chain.py  # V1: LCEL链
│   │   │   │   └── crag_graph.py    # V3: LangGraph CRAG
│   │   │   ├── embeddings.py
│   │   │   ├── document_loader.py
│   │   │   └── vectorstore.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schemas.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── data/
│   └── labor_laws/              # 劳动法文档
├── .windsurf/
│   └── workflows/
│       ├── overview.md
│       ├── v1-basic.md
│       ├── v2-enhanced.md
│       ├── v3-advanced.md
│       └── progress.md
└── README.md
```

## 数据来源
- 法律条文: 国家法律法规数据库 (flk.npc.gov.cn) / GitHub开源数据
- V1需要6部核心法律: 劳动法、劳动合同法、劳动争议调解仲裁法、社会保险法、工伤保险条例、职工带薪年休假条例
