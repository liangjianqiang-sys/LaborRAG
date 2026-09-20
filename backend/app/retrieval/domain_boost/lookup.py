"""知识图谱查找与注入逻辑。

数据表在 relations.py / concepts.py / source_map.py；本模块只放逻辑：
  - parent_store 精确查找（含懒加载缓存）
  - 概念映射查找（concept_lookup）
  - 伴生法条注入（inject_related_articles）
"""
import json
import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.utils.law_refs import article_to_cn, cn_to_int
from app.retrieval.domain_boost.concepts import CONCEPT_ARTICLE_MAP
from app.retrieval.domain_boost.relations import LAW_GRAPH
from app.retrieval.domain_boost.source_map import _LAW_SOURCE_MAP, _SOURCE_LAW_MAP


# ─── parent_store 缓存（懒加载） ───
_parent_store_cache: Optional[dict] = None


def _load_parent_store() -> dict:
    """从parent_store.json加载父块存储（带缓存）。"""
    global _parent_store_cache
    if _parent_store_cache is not None:
        return _parent_store_cache

    from app.core.config import settings
    ps_path = os.path.join(settings.VECTOR_STORE_PATH, "parent_store.json")
    if not os.path.exists(ps_path):
        _parent_store_cache = {}
        return _parent_store_cache

    try:
        with open(ps_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _parent_store_cache = {}
        for parent_id, item in data.items():
            _parent_store_cache[parent_id] = Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
        print(f"[LawGraph] parent_store已加载: {len(_parent_store_cache)}个父块")
    except Exception as e:
        print(f"[LawGraph] parent_store加载失败: {e}")
        _parent_store_cache = {}

    return _parent_store_cache


def _exact_lookup_from_parent_store(source: str, article: str, k: int = 2) -> List[Tuple[Document, float]]:
    """从parent_store精确查找法条文档。

    可命中向量库child chunk缺失但parent_store存在的法条（如工伤保险条例第15/35条）。
    注意：parent_store中的source可能与_LAW_SOURCE_MAP不同（.txt vs .pdf），
    因此同时匹配两种扩展名。
    """
    ps = _load_parent_store()
    if not ps:
        return []

    # 构造可能的source值（.txt和.pdf都试）
    possible_sources = {source}
    base = source.rsplit(".", 1)[0]
    if source.endswith(".pdf"):
        possible_sources.add(f"{base}.txt")
    elif source.endswith(".txt"):
        possible_sources.add(f"{base}.pdf")

    results = []
    for parent_id, doc in ps.items():
        if doc.metadata.get("source") in possible_sources and doc.metadata.get("article") == article:
            results.append((doc, 1.0))
            if len(results) >= k:
                break
    return results


def concept_lookup(question: str, k: int = 2) -> List[Tuple[Document, float]]:
    """根据问题关键词从概念映射表查找目标法条文档。

    遍历CONCEPT_ARTICLE_MAP，匹配问题中出现的关键词，
    从parent_store精确查找对应的法条文档。
    按概念长度降序排列（更长的概念更具体、更相关），只取前k个概念的文档。
    """
    # 收集匹配的概念，按长度降序（更具体的概念优先）
    matched_concepts = []
    for concept in CONCEPT_ARTICLE_MAP:
        if concept in question:
            matched_concepts.append(concept)
    matched_concepts.sort(key=len, reverse=True)

    ps_docs = []
    seen = set()  # (source, article) 去重
    for concept in matched_concepts[:k]:  # 只取前k个最具体的概念
        articles = CONCEPT_ARTICLE_MAP[concept]
        for law_name, article_num in articles:
            cn_article = article_to_cn(article_num)
            source = _LAW_SOURCE_MAP.get(law_name, "")
            key = (source, cn_article)
            if key in seen:
                continue
            seen.add(key)
            exact = _exact_lookup_from_parent_store(source, cn_article, k=1)
            if exact:
                ps_docs.extend(exact)
    return ps_docs


def _extract_law_article_from_doc(doc: Document) -> Optional[Tuple[str, str]]:
    """从文档metadata中提取(法律名, 法条号)。

    metadata格式：source='劳动合同法.txt', article='第四十七条'
    返回格式：('劳动合同法', '47')
    """
    source = doc.metadata.get("source", "")
    article = doc.metadata.get("article", "")

    if not source or not article:
        return None

    # source → 法律名
    law_name = _SOURCE_LAW_MAP.get(source)
    if not law_name:
        # 尝试从文件名提取（去掉扩展名）
        base = source.rsplit(".", 1)[0]
        law_name = base

    # article → 阿拉伯数字（"第四十七条" → "47"）
    if not (article.startswith("第") and article.endswith("条")):
        return None
    num = cn_to_int(article[1:-1])
    if num is None:
        return None

    return (law_name, str(num))


def get_related_articles(law_name: str, article_num: str) -> List[Tuple[str, str, str, str]]:
    """查询知识图谱，获取伴生法条列表。

    Args:
        law_name: 法律名（如"劳动合同法"）
        article_num: 法条号阿拉伯数字（如"47"）

    Returns:
        [(法律名, 法条号, 关系类型, 描述), ...]
    """
    key = (law_name, article_num)
    return LAW_GRAPH.get(key, [])


def inject_related_articles(
    docs: List[Tuple[Document, float]],
    retriever,
    max_inject: int = 3,
) -> List[Tuple[Document, float]]:
    """知识图谱注入：扫描已命中文档，自动补充伴生法条。

    在检索后调用，扫描已命中文档的metadata，
    查知识图谱获取伴生法条，用精确检索补充缺失的关联文档。

    Args:
        docs: 检索结果 [(Document, score), ...]
        retriever: 检索器实例（用于补充检索伴生法条）
        max_inject: 最多注入的伴生法条文档数

    Returns:
        扩展后的文档列表（去重）
    """
    if not docs:
        return docs

    # 1. 扫描已命中文档，收集(法律名, 法条号)
    #    只对分数较高的文档做图谱注入，避免低质量命中引入无关伴生法条
    #
    #    用「列表 + 去重集合」而非 set：set 的迭代顺序由字符串哈希的随机盐
    #    （PYTHONHASHSEED）决定，每次进程重启都不同；而下面按 max_inject 截断，
    #    于是同一个问题在不同进程里会注入不同的伴生法条，检索结果不可复现
    #    （曾让离线基线 MAP 在 0.7294 / 0.7301 之间抖动）。
    #    保持 docs 的分数降序，既确定又符合「优先处理高分命中」的原意。
    hit_articles: List[Tuple[str, str]] = []
    seen_hits: set = set()
    for doc, score in docs:
        if score < 0.5:
            continue
        result = _extract_law_article_from_doc(doc)
        if result and result not in seen_hits:
            seen_hits.add(result)
            hit_articles.append(result)

    if not hit_articles:
        return docs

    # 2. 查知识图谱，收集需要补充的伴生法条
    needed_articles = []  # [(法律名, 法条号, 关系类型, 描述)]
    seen = set(hit_articles)

    for law_name, article_num in hit_articles:
        related = get_related_articles(law_name, article_num)
        for rel_law, rel_article, rel_type, rel_desc in related:
            key = (rel_law, rel_article)
            if key not in seen:
                seen.add(key)
                needed_articles.append((rel_law, rel_article, rel_type, rel_desc))

    if not needed_articles:
        return docs

    # 3. 对每个伴生法条，构造精确检索查询
    injected = []
    seen_doc_ids = {
        doc.metadata.get("doc_id", doc.page_content[:80])
        for doc, score in docs
    }

    for rel_law, rel_article, rel_type, rel_desc in needed_articles[:max_inject]:
        # 构造精确查询：法律名+法条编号
        cn_article = article_to_cn(rel_article)
        source = _LAW_SOURCE_MAP.get(rel_law, "")

        try:
            # 优先从parent_store精确查找（可命中child chunk缺失的法条）
            exact = _exact_lookup_from_parent_store(source, cn_article, k=2)
            if exact:
                for doc, score in exact:
                    doc_id = doc.metadata.get("doc_id", doc.page_content[:80])
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        # 规则召回置顶：伴生法条分数不低于主检索，确保进入Top5
                        doc.metadata["injected_by"] = "rule"  # 标记为规则注入，压缩时跳过
                        injected.append((doc, 0.95))
                        print(f"[LawGraph] 注入伴生法条(精确): {rel_law}第{rel_article}条 ({rel_type}: {rel_desc})")
                continue  # 精确查找成功，跳过语义检索

            # 精确查找失败，回退到语义检索
            query = f"{rel_law}{cn_article}"
            from app.retrieval.reranked import RerankedRetriever
            if isinstance(retriever, RerankedRetriever):
                extra = retriever.retrieve(
                    query=query,
                    k=3,
                    score_threshold=0.0,
                    rerank_top_k=2,
                )
            else:
                extra = retriever.retrieve(
                    query=query,
                    k=3,
                    score_threshold=0.0,
                )

            # 语义检索结果去重合并
            for doc, score in extra:
                doc_id = doc.metadata.get("doc_id", doc.page_content[:80])
                result = _extract_law_article_from_doc(doc)
                if result and result == (rel_law, rel_article):
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        injected.append((doc, max(score, 0.90)))  # 规则召回置顶
                        print(f"[LawGraph] 注入伴生法条(语义): {rel_law}第{rel_article}条 ({rel_type}: {rel_desc})")
        except Exception as e:
            print(f"[LawGraph] 伴生法条检索失败 {rel_law}第{rel_article}条: {e}")
            continue

    # 4. 合并并按分数排序
    all_docs = list(docs) + injected
    all_docs.sort(key=lambda x: x[1], reverse=True)

    return all_docs
