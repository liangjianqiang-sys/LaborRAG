import uuid
from typing import List, Optional
from langchain_core.documents import Document

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

        # 根据RAG_MODE选择生成器
        if settings.RAG_MODE == "crag":
            from app.core.generator.crag_graph import CRAGGraph
            self.crag_graph = CRAGGraph(self.retriever)
            self.generator: BaseGenerator = SimpleChainGenerator()  # fallback
            print(f"   Generator: CRAG (self-corrective)")
        else:
            self.crag_graph = None
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

        # 预加载Reranker模型，避免首次提问时卡顿
        self._preload_reranker()

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

    def chat(self, request: ChatRequest) -> ChatResponse:
        """核心问答方法：根据RAG_MODE选择simple或crag链路，支持多轮对话。"""
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # 记录用户消息到对话历史
        conversation_manager.add_message(conversation_id, MessageRole.USER, request.question)

        # 获取对话上下文
        conversation_context = conversation_manager.build_context_string(conversation_id)

        if settings.RAG_MODE == "crag" and self.crag_graph:
            # V3: CRAG自我纠错工作流（支持多轮对话上下文）
            try:
                result = self.crag_graph.run(request.question, conversation_context=conversation_context)
                answer = result["answer"]
                relevant_docs = result["context_docs"]
                crag_steps = result.get("steps", [])
                rewritten_question = result.get("rewritten_question", "")
                rag_mode = "crag"
            except Exception as e:
                print(f"[CRAG] 工作流执行失败，降级到simple模式: {e}")
                import traceback
                traceback.print_exc()
                # 降级到简单链路
                relevant_docs = self.retriever.retrieve(
                    query=request.question,
                    k=settings.TOP_K,
                    score_threshold=settings.SCORE_THRESHOLD,
                )
                answer = self.generator.generate(
                    question=request.question,
                    context_docs=relevant_docs,
                )
                crag_steps = [f"CRAG降级: {str(e)[:50]}"]
                rewritten_question = ""
                rag_mode = "crag_fallback"
        else:
            # V1/V2: 简单链路
            relevant_docs = self.retriever.retrieve(
                query=request.question,
                k=settings.TOP_K,
                score_threshold=settings.SCORE_THRESHOLD,
            )
            answer = self.generator.generate(
                question=request.question,
                context_docs=relevant_docs,
            )
            crag_steps = []
            rewritten_question = ""
            rag_mode = "simple"

        # 记录助手回答到对话历史
        conversation_manager.add_message(conversation_id, MessageRole.ASSISTANT, answer)

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

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            rag_mode=rag_mode,
            crag_steps=crag_steps,
            rewritten_question=rewritten_question,
        )

    def add_document(self, filepath: str) -> int:
        """添加单个文档到向量库。"""
        documents = self.document_loader.load_single_file(filepath)
        return self.vector_store_manager.add_documents(documents)

    @property
    def is_ready(self) -> bool:
        """引擎是否可用。"""
        return self.retriever.is_ready()
