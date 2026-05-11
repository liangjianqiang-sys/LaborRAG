---
description: V1基础版开发 - 纯向量检索RAG智能问答系统
---

# V1 基础版开发工作流

## 当前状态
- [ ] 未开始

## 开发步骤

### 阶段1: 项目初始化与后端骨架
1. 创建后端目录结构和所有 `__init__.py`
2. 创建 `backend/requirements.txt`，包含依赖:
   - fastapi, uvicorn, python-dotenv, pydantic
   - langchain, langchain-core, langchain-openai, langchain-community
   - faiss-cpu, sentence-transformers
   - pypdf, docx2txt, unstructured
   - python-multipart (文件上传)
3. 创建 `backend/app/config.py` - 配置管理（LLM API Key、模型名、向量库路径等）
4. 创建 `backend/app/models/schemas.py` - Pydantic数据模型（ChatRequest/ChatResponse/SourceDocument等）
5. 创建 `backend/.env.example` - 环境变量模板

### 阶段2: RAG核心引擎
6. 创建 `backend/app/core/embeddings.py` - Embedding模型初始化（BGE-M3本地 / OpenAI兼容API）
7. 创建 `backend/app/core/document_loader.py` - 文档加载器（支持PDF/DOCX/TXT/MD）
8. 创建 `backend/app/core/vectorstore.py` - FAISS向量库管理（构建/检索/保存/加载）
9. 创建 `backend/app/core/retriever/base.py` - BaseRetriever抽象类
10. 创建 `backend/app/core/retriever/vector.py` - V1纯向量检索实现
11. 创建 `backend/app/core/generator/base.py` - BaseGenerator抽象类
12. 创建 `backend/app/core/generator/simple_chain.py` - V1 LCEL简单链实现
13. 创建 `backend/app/core/rag_engine.py` - RAG引擎（整合检索+生成，对外统一接口）

### 阶段3: API接口
14. 创建 `backend/app/api/routes.py` - API路由:
    - POST /api/v1/chat - 智能问答
    - GET /api/v1/knowledge-base/status - 知识库状态
    - POST /api/v1/knowledge-base/build - 构建知识库
    - POST /api/v1/documents/upload - 上传文档
    - GET /api/v1/documents/list - 文档列表
    - DELETE /api/v1/documents/{filename} - 删除文档
15. 创建 `backend/app/main.py` - FastAPI应用入口（CORS、路由注册、启动事件）

### 阶段4: 劳动法数据准备
16. 创建 `data/labor_laws/` 目录
17. 准备6部核心法律文本文件:
    - 劳动法.txt
    - 劳动合同法.txt
    - 劳动争议调解仲裁法.txt
    - 社会保险法.txt
    - 工伤保险条例.txt
    - 职工带薪年休假条例.txt
    数据来源: 国家法律法规数据库 flk.npc.gov.cn 或 GitHub开源数据

### 阶段5: 前端界面
18. 用Vite创建React+TypeScript项目: `npm create vite@latest frontend -- --template react-ts`
19. 安装依赖: tailwindcss, @tailwindcss/vite, lucide-react, react-markdown
20. 创建聊天界面组件:
    - ChatWindow - 对话窗口
    - MessageBubble - 消息气泡（区分用户/AI）
    - SourceCard - 参考来源卡片
    - InputBar - 输入框
    - Sidebar - 侧边栏（知识库状态/文档管理）
21. 创建API调用模块 - 与后端FastAPI通信
22. 创建 `frontend/src/App.tsx` - 主布局

### 阶段6: 联调与测试
23. 启动后端: `cd backend && uvicorn app.main:app --reload --port 8000`
24. 启动前端: `cd frontend && npm run dev`
25. 测试完整流程: 上传文档 → 构建知识库 → 提问 → 查看回答和来源
26. 修复联调问题

## 关键设计决策
- 检索层和生成层使用抽象基类，V2/V3只需新增实现类，不改动接口
- Embedding优先使用BGE-M3本地模型，也支持OpenAI兼容API
- LLM使用OpenAI兼容API（DeepSeek/GLM-4/GPT均可）
- 文档切分使用RecursiveCharacterTextSplitter，中文分隔符优先

## 验收标准
- [ ] 后端API全部可访问，无报错
- [ ] 上传劳动法文档后能成功构建向量库
- [ ] 提问劳动法问题能返回带来源引用的回答
- [ ] 前端界面能正常对话，显示回答和参考来源
- [ ] 知识库状态和文档管理功能正常
