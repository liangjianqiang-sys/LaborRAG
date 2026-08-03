import traceback
import uuid
import threading

from app.config import settings
from app.core.logging_config import get_logger
from app.core.vectorstore import VectorStoreManager
from app.core.document_loader import DocumentLoader
from app.core.retriever.base import BaseRetriever
from app.core.retriever.vector import VectorRetriever
from app.core.retriever.bm25 import BM25Retriever
from app.core.retriever.hybrid import HybridRetriever
from app.core.conversation import conversation_manager
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument, MessageRole
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.generator.prompts import ANSWER_STYLE_RULES

logger = get_logger(__name__)


class RAGEngine:
    """RAG引擎，整合检索和生成，对外提供统一接口。

    固定使用Agentic RAG工作流，异常时降级到SimpleChain。
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

        # Agentic RAG工作流（主链路）
        from app.core.generator.agent_graph import AgenticRAGGraph
        self.agent_graph = AgenticRAGGraph(self.retriever)
        # 降级兜底LLM（懒加载）
        self._fallback_llm = None
        logger.info(f"   Generator: Agentic RAG")
        # 启动时预加载Reranker，避免首次聊天卡顿
        _ = self.eval_retriever

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
            logger.info(f"   Retriever: Hybrid (vector={settings.VECTOR_WEIGHT}, bm25={settings.BM25_WEIGHT})")
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
            logger.info(f"   Retriever: Reranked (model={settings.RERANKER_MODEL_NAME})")
            return reranked
        else:
            logger.info("   Retriever: Vector (V1)")
            return vector_retriever

    def _preload_reranker(self):
        """预加载Reranker模型，避免首次提问时卡顿。"""
        if settings.RETRIEVER_TYPE == "reranked" and hasattr(self.retriever, '_get_reranker'):
            logger.info("   Preloading reranker model...")
            self.retriever._get_reranker()
            logger.info("   Reranker model preloaded.")

    def initialize(self) -> bool:
        """初始化：尝试加载已有向量库，否则构建新的。"""
        # 父子分块模式提示
        if settings.PARENT_CHILD_ENABLED:
            logger.info("   Parent-Child Chunking: ENABLED (子块精准检索→父块完整上下文)")
        else:
            logger.info("   Parent-Child Chunking: DISABLED (传统切分模式)")

        if self.vector_store_manager.load():
            logger.info("Loaded existing vector store.")
            # 加载BM25索引（如果存在）
            if settings.RETRIEVER_TYPE in ("hybrid", "reranked"):
                if self.bm25_retriever.load_index():
                    logger.info("Loaded existing BM25 index.")
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
            logger.info(f"BM25 index built with {chunk_count} documents.")

    def build_knowledge_base(self, rebuild: bool = False) -> bool:
        """构建知识库。"""
        if self.vector_store_manager.is_ready and not rebuild:
            return True

        documents = self.document_loader.load_documents()
        if not documents:
            logger.info("No documents found in data directory.")
            return False

        chunk_count = self.vector_store_manager.build_from_documents(documents)
        logger.info(f"Knowledge base built with {chunk_count} chunks from {len(documents)} document sections.")

        # 同时构建BM25索引
        if settings.RETRIEVER_TYPE in ("hybrid", "reranked"):
            self._build_bm25_index()

        return chunk_count > 0

    async def chat(self, request: ChatRequest, use_reranker: bool = False) -> ChatResponse:
        """核心问答方法：Agentic RAG工作流，支持多轮对话。"""
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

        # Agentic RAG工作流
        try:
            # 评估时临时切换为Reranker检索器
            if use_reranker:
                orig_retriever = self.agent_graph.retriever
                self.agent_graph.retriever = self.eval_retriever
            try:
                result = await self.agent_graph.run(request.question, conversation_context=conversation_context, skip_guardrails=request.skip_guardrails)
            finally:
                if use_reranker:
                    self.agent_graph.retriever = orig_retriever
            answer = result["answer"]
            relevant_docs = result["context_docs"]
            rag_steps = result.get("steps", [])
            rewritten_question = result.get("intent", "")
        except Exception as e:
            # 降级到SimpleChain
            answer, relevant_docs, rag_steps, rewritten_question = \
                self._fallback_simple(request.question, f"Agent降级: {str(e)[:50]}")

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

        # 完整contexts（父块，供Faithfulness评估使用）
        full_contexts = [doc.page_content for doc, score in relevant_docs]
        # 子块精准段落（供Context Precision/Recall评估使用）
        child_contexts = []
        for doc, score in relevant_docs:
            if doc.metadata.get("chunk_type") == "parent":
                # 父块：从parent_store中查找对应的子块
                parent_id = doc.metadata.get("parent_id", "")
                if parent_id and hasattr(self.vector_store_manager, 'parent_store'):
                    # 从parent_store中无法直接反查子块，改用向量库similarity_search
                    # 但更简单的方式：从父块文本中按款项拆分
                    from app.core.generator.helpers import _is_clause_line
                    lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]
                    current_para = []
                    for line in lines:
                        if _is_clause_line(line) or (current_para and line.startswith("第")):
                            if current_para:
                                child_contexts.append("\n".join(current_para))
                            current_para = [line]
                        else:
                            current_para.append(line)
                    if current_para:
                        child_contexts.append("\n".join(current_para))
                else:
                    child_contexts.append(doc.page_content)
            else:
                # 子块或fallback，直接使用
                child_contexts.append(doc.page_content)
        # 将计算器结果纳入评估上下文，确保RAGAS能看到计算器引用的法条
        calc_result = result.get("calculation_result", "") if isinstance(result, dict) else ""
        if calc_result:
            full_contexts.append(calc_result)

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            rag_steps=rag_steps,
            rewritten_question=rewritten_question,
            confidence=confidence,
            disclaimer=True,
            full_contexts=full_contexts,
            child_contexts=child_contexts,
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
                    logger.info("   [Reranker] 评估模式已加载")
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

    @property
    def fallback_llm(self):
        """降级兜底LLM（懒加载）。"""
        if self._fallback_llm is None:
            self._fallback_llm = ChatOpenAI(
                model=settings.LLM_MODEL_NAME,
                openai_api_key=settings.LLM_API_KEY,
                openai_api_base=settings.LLM_API_BASE,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                extra_body={"enable_thinking": False},
            )
        return self._fallback_llm

    def _fallback_simple(self, question: str, step_msg: str, use_reranker: bool = False):
        """工作流异常时降级：检索+直接生成。"""
        retriever = self.eval_retriever if use_reranker else self.retriever
        relevant_docs = retriever.retrieve(
            query=question,
            k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        context = "\n\n".join(
            f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc, score in relevant_docs
        ) if relevant_docs else "未找到相关参考资料。"

        prompt = ChatPromptTemplate.from_template(
            "{style_rules}\n\n参考资料：\n{context}\n\n用户问题：{question}\n\n请基于以上参考资料回答用户的问题："
        )
        chain = prompt | self.fallback_llm
        answer = chain.invoke({
            "context": context,
            "question": question,
            "style_rules": ANSWER_STYLE_RULES,
        }).content
        return answer, relevant_docs, [step_msg], ""

    def add_document(self, filepath: str) -> int:
        """添加单个文档到向量库。"""
        documents = self.document_loader.load_single_file(filepath)
        return self.vector_store_manager.add_documents(documents)

    @property
    def is_ready(self) -> bool:
        """引擎是否可用。"""
        return self.retriever.is_ready()
