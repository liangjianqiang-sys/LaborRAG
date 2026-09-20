"""检索层门面：包外只依赖这里，不直接依赖具体检索器实现。

分层意图
--------
`retriever` 包内有多个可替换的具体实现（vector / bm25 / hybrid / reranked），
包外只认这个门面。这样"换实现"或"改构造参数"只需改一处 —— 之前
「VectorRetriever + HybridRetriever(权重取自 settings)」这段构造在
`rag_engine.py` 里重复了 3 次（hybrid 分支、reranked 分支、eval_retriever 属性），
任何一次改权重来源都可能只改到其中一处。

构造函数内部一律惰性导入：本包被 `from app.retrieval.base import ...`
这样的语句触发导入，模块级导入会把 jieba 词典、向量库依赖一并拉起。
"""
__all__ = [
    "build_retriever",
    "build_hybrid_retriever",
    "build_reranked_retriever",
]


def build_hybrid_retriever(vector_store_manager, bm25_retriever):
    """构造「向量 + BM25 + RRF 融合」检索器，权重与 rrf_k 取自 settings。"""
    from app.core.config import settings
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.vector import VectorRetriever
    from app.core.logging import get_logger

    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(vector_store_manager),
        bm25_retriever=bm25_retriever,
        vector_weight=settings.VECTOR_WEIGHT,
        bm25_weight=settings.BM25_WEIGHT,
        rrf_k=settings.RRF_K,
    )
    get_logger(__name__).info(
        f"   Retriever: Hybrid (vector={settings.VECTOR_WEIGHT}, bm25={settings.BM25_WEIGHT})"
    )
    return retriever


def build_reranked_retriever(vector_store_manager, bm25_retriever):
    """在混合检索器之上叠加重排层。

    注意：RerankedRetriever 的构造**不加载** CrossEncoder 模型，
    模型在首次 retrieve() 时经模块级单例懒加载。所以这里可以放心调用。
    """
    from app.core.config import settings
    from app.retrieval.reranked import RerankedRetriever
    from app.core.logging import get_logger

    retriever = RerankedRetriever(
        hybrid_retriever=build_hybrid_retriever(vector_store_manager, bm25_retriever)
    )
    get_logger(__name__).info(f"   Retriever: Reranked (model={settings.RERANKER_MODEL_NAME})")
    return retriever


def build_retriever(vector_store_manager, bm25_retriever):
    """按 settings.RETRIEVER_TYPE 构造主链路检索器。"""
    from app.core.config import settings
    from app.retrieval.vector import VectorRetriever
    from app.core.logging import get_logger

    if settings.RETRIEVER_TYPE == "hybrid":
        return build_hybrid_retriever(vector_store_manager, bm25_retriever)
    if settings.RETRIEVER_TYPE == "reranked":
        return build_reranked_retriever(vector_store_manager, bm25_retriever)

    get_logger(__name__).info("   Retriever: Vector (V1)")
    return VectorRetriever(vector_store_manager)
