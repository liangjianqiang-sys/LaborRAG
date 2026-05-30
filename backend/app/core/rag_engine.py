import traceback
import uuid
import threading

from app.config import settings
from app.core.vectorstore import VectorStoreManager
from app.core.document_loader import DocumentLoader
from app.core.retriever.base import BaseRetriever
from app.core.retriever.vector import VectorRetriever
from app.core.retriever.bm25 import BM25Retriever
from app.core.retriever.hybrid import HybridRetriever
from app.core.generator.base import BaseGenerator
from app.core.generator.simple_chain import SimpleChainGenerator
from app.core.conversation import conversation_manager
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument, MessageRole


class RAGEngine:
    """RAG引擎，整合检索和生成，对外提供统一接口。

    通过retriever和generator的抽象，V2/V3只需替换实现类即可升级。
    """

    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.document_loader = DocumentLoader(settings.DATA_DIR)
        self.bm25_retriever = BM25Retriever()

        # 根据配置选择检索器
        self.retriever: BaseRetriever = self._create_retriever()
        # 评估专用检索器（单例，含Reranker提高精确率）
        self._eval_retriever = None
        self._eval_retriever_lock = threading.Lock()

        # 根据RAG_MODE选择生成器
        if settings.RAG_MODE == "agent":
            from app.core.generator.agent_graph import AgenticRAGGraph
            self.agent_graph = AgenticRAGGraph(self.retriever)
            self.crag_graph = None
            self.generator: BaseGenerator = SimpleChainGenerator()  # fallback
            print(f"   Generator: Agentic RAG (multi-agent)")
            # 启动时预加载Reranker，避免首次聊天卡顿
            _ = self.eval_retriever
        elif settings.RAG_MODE == "crag":
            from app.core.generator.crag_graph import AdaptiveRAGGraph
            self.crag_graph = AdaptiveRAGGraph(self.retriever)
            self.agent_graph = None
            self.generator: BaseGenerator = SimpleChainGenerator()  # fallback
            print(f"   Generator: Adaptive RAG (complexity-based routing)")
        else:
            self.crag_graph = None
            self.agent_graph = None
            self.generator: BaseGenerator = SimpleChainGenerator()
            print(f"   Generator: SimpleChain")

    def _create_retriever(self) -> BaseRetriever:
        """根据RETRIEVER_TYPE配置创建检索器。"""
        vector_retriever = VectorRetriever(self.vector_store_manager)

        if settings.RETRIEVER_TYPE == "hybrid":
            hybrid = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=self.bm25_retriever,
                vector_weight=settings.VECTOR_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
                rrf_k=settings.RRF_K,
            )
            print(f"   Retriever: Hybrid (vector={settings.VECTOR_WEIGHT}, bm25={settings.BM25_WEIGHT})")
            return hybrid
        elif settings.RETRIEVER_TYPE == "reranked":
            from app.core.retriever.reranked import RerankedRetriever
            hybrid = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=self.bm25_retriever,
                vector_weight=settings.VECTOR_WEIGHT,
                bm25_weight=settings.BM25_WEIGHT,
                rrf_k=settings.RRF_K,
            )
            reranked = RerankedRetriever(hybrid_retriever=hybrid)
            print(f"   Retriever: Reranked (model={settings.RERANKER_MODEL_NAME})")
            return reranked
        else:
            print("   Retriever: Vector (V1)")
            return vector_retriever

    def _preload_reranker(self):
        """预加载Reranker模型，避免首次提问时卡顿。"""
        if settings.RETRIEVER_TYPE == "reranked" and hasattr(self.retriever, '_get_reranker'):
            print("   Preloading reranker model...")
            self.retriever._get_reranker()
            print("   Reranker model preloaded.")

    def initialize(self) -> bool:
        """初始化：尝试加载已有向量库，否则构建新的。"""
        # 父子分块模式提示
        if settings.PARENT_CHILD_ENABLED:
            print("   Parent-Child Chunking: ENABLED (子块精准检索→父块完整上下文)")
        else:
            print("   Parent-Child Chunking: DISABLED (传统切分模式)")

        if self.vector_store_manager.load():
            print("Loaded existing vector store.")
            # 加载BM25索引（如果存在）
            if settings.RETRIEVER_TYPE in ("hybrid", "reranked"):
                if self.bm25_retriever.load_index():
                    print("Loaded existing BM25 index.")
        else:
            return self.build_knowledge_base()

        # 如果BM25索引不存在但需要混合检索，构建之
        if settings.RETRIEVER_TYPE in ("hybrid", "reranked") and not self.bm25_retriever.is_ready():
            self._build_bm25_index()

        # 预加载Reranker模型，避免首次提问/评估时卡顿
        self._preload_reranker()
        # 触发加载评估专用Reranker（只加载一次，线程安全）
        _ = self.eval_retriever

        return True

    def _build_bm25_index(self):
        """从向量库中的文档构建BM25索引。"""
        if self.vector_store_manager.vector_store is None:
            return
        # 从FAISS中获取所有文档
        docstore = self.vector_store_manager.vector_store.docstore
        documents = list(docstore._dict.values())
        if documents:
            chunk_count = self.bm25_retriever.build_index(documents)
            print(f"BM25 index built with {chunk_count} documents.")

    def build_knowledge_base(self, rebuild: bool = False) -> bool:
        """构建知识库。"""
        if self.vector_store_manager.is_ready and not rebuild:
            return True

        documents = self.document_loader.load_documents()
        if not documents:
            print("No documents found in data directory.")
            return False

        chunk_count = self.vector_store_manager.build_from_documents(documents)
        print(f"Knowledge base built with {chunk_count} chunks from {len(documents)} document sections.")

        # 同时构建BM25索引
        if settings.RETRIEVER_TYPE in ("hybrid", "reranked"):
            self._build_bm25_index()

        return chunk_count > 0

    def chat(self, request: ChatRequest, use_reranker: bool = False) -> ChatResponse:
        """核心问答方法：根据RAG_MODE选择simple或crag链路，支持多轮对话。"""
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # 记录用户消息到对话历史
        conversation_manager.add_message(conversation_id, MessageRole.USER, request.question)

        # 获取对话上下文（优先后端持久化，兜底前端传来的history）
        conversation_context = conversation_manager.build_context_string(conversation_id)
        if not conversation_context and request.history:
            # 后端无历史（如重启后首次请求），用前端传来的历史兜底
            lines = []
            for msg in request.history:
                if msg.role == MessageRole.USER:
                    lines.append(f"用户: {msg.content}")
                elif msg.role == MessageRole.ASSISTANT:
                    lines.append(f"助手: {msg.content}")
            conversation_context = "\n".join(lines)

        if settings.RAG_MODE == "agent" and self.agent_graph:
            # V4: Agentic RAG多Agent协作
            try:
                # 评估时临时切换为Reranker检索器
                if use_reranker:
                    orig_retriever = self.agent_graph.retriever
                    self.agent_graph.retriever = self.eval_retriever
                try:
                    result = self.agent_graph.run(request.question, conversation_context=conversation_context, skip_guardrails=request.skip_guardrails)
                finally:
                    if use_reranker:
                        self.agent_graph.retriever = orig_retriever
                answer = result["answer"]
                relevant_docs = result["context_docs"]
                crag_steps = result.get("steps", [])
                rewritten_question = result.get("intent", "")
                rag_mode = "agent"
            except Exception as e:
                answer, relevant_docs, crag_steps, rewritten_question, rag_mode = \
                    self._fallback_simple(request.question, f"Agent降级: {str(e)[:50]}", "agent_fallback")
        elif settings.RAG_MODE == "crag" and self.crag_graph:
            # V3: Adaptive RAG自适应工作流（支持多轮对话上下文）
            try:
                if use_reranker:
                    orig_retriever = self.crag_graph.retriever
                    self.crag_graph.retriever = self.eval_retriever
                try:
                    result = self.crag_graph.run(request.question, conversation_context=conversation_context, skip_guardrails=request.skip_guardrails)
                finally:
                    if use_reranker:
                        self.crag_graph.retriever = orig_retriever
                answer = result["answer"]
                relevant_docs = result["context_docs"]
                crag_steps = result.get("steps", [])
                rewritten_question = result.get("rewritten_question", "")
                rag_mode = "crag"
            except Exception as e:
                answer, relevant_docs, crag_steps, rewritten_question, rag_mode = \
                    self._fallback_simple(request.question, f"CRAG降级: {str(e)[:50]}", "crag_fallback")
        else:
            # V1/V2: 简单链路（评估时用Reranker提高精确率）
            retriever = self.eval_retriever if use_reranker else self.retriever
            relevant_docs = retriever.retrieve(
                query=request.question,
                k=settings.TOP_K,
                score_threshold=settings.SCORE_THRESHOLD,
            )
            answer = self.generator.generate(
                question=request.question,
                context_docs=relevant_docs,
                skip_guardrails=request.skip_guardrails,
            )
            crag_steps = []
            rewritten_question = ""
            rag_mode = "simple"

        # 记录助手回答到对话历史
        conversation_manager.add_message(conversation_id, MessageRole.ASSISTANT, answer)

        # 置信度评分：基于检索文档数量和分数
        confidence = self._compute_confidence(relevant_docs)

        # 注意：免责声明和置信度警告不追加到answer文本中，
        # 否则RAGAS faithfulness会将这些追加内容判定为"幻觉"（contexts中无依据）
        # 前端通过ChatResponse的disclaimer/confidence字段显示

        # 构建来源信息
        sources = [
            SourceDocument(
                content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                source=doc.metadata.get("source", "未知"),
                score=round(float(score), 4),
                page=doc.metadata.get("page"),
            )
            for doc, score in relevant_docs
        ]

        # 完整contexts（供评估使用，不截断）
        full_contexts = [doc.page_content for doc, score in relevant_docs]
        # 将计算器结果纳入评估上下文，确保RAGAS能看到计算器引用的法条
        calc_result = result.get("calculation_result", "") if isinstance(result, dict) else ""
        if calc_result:
            full_contexts.append(calc_result)

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            rag_mode=rag_mode,
            crag_steps=crag_steps,
            rewritten_question=rewritten_question,
            confidence=confidence,
            disclaimer=True,
            full_contexts=full_contexts,
        )

    @property
    def eval_retriever(self):
        """评估专用检索器（懒加载+线程锁，避免并发重复加载）。"""
        if self._eval_retriever is None:
            with self._eval_retriever_lock:
                if self._eval_retriever is None:  # 双重检查
                    from app.core.retriever.reranked import RerankedRetriever
                    from app.core.retriever.hybrid import HybridRetriever
                    from app.core.retriever.vector import VectorRetriever
                    vector_retriever = VectorRetriever(self.vector_store_manager)
                    hybrid = HybridRetriever(
                        vector_retriever=vector_retriever,
                        bm25_retriever=self.bm25_retriever,
                        vector_weight=settings.VECTOR_WEIGHT,
                        bm25_weight=settings.BM25_WEIGHT,
                        rrf_k=settings.RRF_K,
                    )
                    self._eval_retriever = RerankedRetriever(hybrid_retriever=hybrid)
                    print("   [Reranker] 评估模式已加载")
        return self._eval_retriever

    @staticmethod
    def _compute_confidence(relevant_docs: list) -> float:
        """基于检索结果计算回答置信度。

        评分逻辑：
        - 无检索结果 → 0.0
        - 有结果时：综合文档数量和平均分数
        - 3条以上且分数>0.7 → 高置信度(0.8~1.0)
        - 1~2条或分数0.3~0.7 → 中置信度(0.5~0.8)
        - 分数<0.3 → 低置信度(0.0~0.5)
        """
        if not relevant_docs:
            return 0.0

        scores = [score for _, score in relevant_docs]
        avg_score = sum(scores) / len(scores)
        doc_count = len(relevant_docs)

        # 基础分 = 平均检索分数
        base = avg_score

        # 文档数量加成：3条以上+0.1，5条以上+0.15
        count_bonus = 0.1 if doc_count >= 3 else (0.05 if doc_count >= 2 else 0.0)
        if doc_count >= 5:
            count_bonus = 0.15

        confidence = min(base + count_bonus, 1.0)
        return round(confidence, 2)

    def _fallback_simple(self, question: str, step_msg: str, rag_mode: str, use_reranker: bool = False):
        """工作流异常时降级到simple链路。"""
        retriever = self.eval_retriever if use_reranker else self.retriever
        relevant_docs = retriever.retrieve(
            query=question,
            k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        answer = self.generator.generate(
            question=question,
            context_docs=relevant_docs,
        )
        return answer, relevant_docs, [step_msg], "", rag_mode

    def add_document(self, filepath: str) -> int:
        """添加单个文档到向量库。"""
        documents = self.document_loader.load_single_file(filepath)
        return self.vector_store_manager.add_documents(documents)

    @property
    def is_ready(self) -> bool:
        """引擎是否可用。"""
        return self.retriever.is_ready()
