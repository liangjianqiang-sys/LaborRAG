"""Retrieve — 检索节点（通用检索 + 对比检索）。"""
import re
from typing import List, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.core.generator.state import AgentState
from app.core.generator.helpers import extract_compare_concepts
from app.core.retriever.base import BaseRetriever


def _do_retrieve(retriever: BaseRetriever, query: str,
                 k: int, score_threshold: float, rerank_top_k: int) -> list:
    """执行单次检索，自动判断是否使用 RerankedRetriever。"""
    from app.core.retriever.reranked import RerankedRetriever
    if isinstance(retriever, RerankedRetriever):
        return retriever.retrieve(
            query=query, k=k, score_threshold=score_threshold,
            rerank_top_k=rerank_top_k,
        )
    return retriever.retrieve(query=query, k=k, score_threshold=score_threshold)


async def retrieve_docs(retriever: BaseRetriever, grader_llm, state: AgentState) -> dict:
    """Retriever Agent：检索法条文档（三种路径共用）。

    优先使用 rewrite 节点输出的 sub_queries（已含法言法语改写），
    仅对中等/复杂题且 rewrite 未提供子查询时，才调用 _expand_queries 补充。
    短查询自动HyDE：用LLM生成假设答案（含法言法语）替代原查询检索。
    """
    print(f"[Agent] 🔍 检索法条...")
    question = state.get("rewritten_question") or state["question"]
    complexity = state.get("complexity", "medium")

    # 自适应rerank_top_k
    rerank_top_k = 5 if complexity == "simple" else (8 if complexity == "medium" else 12)

    # ── HyDE：短查询(≤15字)自动生成假设答案用于检索 ──
    hyde_query = None
    if len(question) <= 15:
        hyde_query = await _generate_hyde(grader_llm, question)
        if hyde_query:
            print(f"[Agent] 🎭 HyDE: '{question}' → '{hyde_query[:50]}'")

    # ── 多查询：优先用 rewrite 节点输出的 sub_queries ──
    sub_queries = state.get("sub_queries")
    if sub_queries and len(sub_queries) > 0:
        # rewrite 已提供子查询，直接使用
        print(f"[Agent] 🔍 使用rewrite子查询: {sub_queries}")
    else:
        # rewrite 未提供（如跳过改写的专业问题），按复杂度决定是否扩展
        sub_queries = [question]
        if complexity in ("medium", "complex"):
            expanded = await _expand_queries(grader_llm, question)
            max_variants = 2 if complexity == "medium" else len(expanded)
            sub_queries = [question] + [q for q in expanded if q != question][:max_variants]

    if hyde_query and hyde_query not in sub_queries:
        sub_queries.append(hyde_query)

    all_docs: List[Tuple[Document, float]] = []
    seen_doc_ids = set()

    for q in sub_queries:
        docs = _do_retrieve(retriever, q, settings.TOP_K, settings.SCORE_THRESHOLD, rerank_top_k)
        for doc, score in docs:
            doc_id = doc.metadata.get("doc_id", doc.page_content[:80])
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_docs.append((doc, score))

    all_docs.sort(key=lambda x: x[1], reverse=True)

    # ── 知识图谱注入 ──
    from app.core.retriever.law_graph import inject_related_articles
    all_docs = inject_related_articles(all_docs, retriever)

    # ── 概念映射注入（同时用原始问题+改写问题匹配，确保口语词不丢失） ──
    # 关键：概念注入的doc强制排在最前面（它们是规则匹配的高置信度结果），
    # 不参与reranker分数排序竞争，避免被reranker低分淹没。
    # 即使法条已在all_docs中（但排序靠后），也要提升到顶部。
    from app.core.retriever.law_graph import concept_lookup, _extract_law_article_from_doc
    original_question = state["question"]
    concept_docs = concept_lookup(original_question + " " + question, k=4)
    injected_docs: List[Tuple[Document, float]] = []
    if concept_docs:
        # 收集概念匹配的法条ID
        concept_article_ids = set()
        for doc, score in concept_docs:
            result = _extract_law_article_from_doc(doc)
            if result:
                concept_article_ids.add(result)

        # 从all_docs中提取概念匹配的doc（提升到顶部），未匹配的保留
        remaining_docs: List[Tuple[Document, float]] = []
        for doc, score in all_docs:
            result = _extract_law_article_from_doc(doc)
            if result and result in concept_article_ids:
                doc.metadata["injected_by"] = "concept_promote"
                injected_docs.append((doc, 0.95))
                print(f"[Agent] 概念提升: {result[0]}第{result[1]}条")
            else:
                remaining_docs.append((doc, score))

        # 概念匹配但不在all_docs中的新法条，也注入
        existing_article_ids = set()
        for doc, score in injected_docs:
            result = _extract_law_article_from_doc(doc)
            if result:
                existing_article_ids.add(result)
        for doc, score in concept_docs:
            result = _extract_law_article_from_doc(doc)
            if result and result not in existing_article_ids:
                existing_article_ids.add(result)
                doc.metadata["injected_by"] = "rule"
                injected_docs.append((doc, 0.95))
                print(f"[Agent] 概念注入: {result[0]}第{result[1]}条")

        all_docs = remaining_docs

    # 截断普通文档（保留rerank_top_k个），概念注入doc不参与截断
    if len(all_docs) > rerank_top_k:
        all_docs = all_docs[:rerank_top_k]

    # 概念注入doc强制置顶（排在普通文档之前），确保P@1命中
    if injected_docs:
        all_docs = injected_docs + all_docs

    # ── 检索兜底 ──
    if not all_docs:
        all_docs = _fallback_retrieve(retriever, question)

    steps = state.get("steps", [])
    expand_info = f" (多查询扩展: {len(sub_queries)}个子查询)" if len(sub_queries) > 1 else ""
    steps.append(f"检索: 获取{len(all_docs)}个文档{expand_info}")
    return {"context_docs": all_docs, "steps": steps}


async def retrieve_for_compare(retriever: BaseRetriever, state: AgentState) -> dict:
    """Comparator检索：分别检索两组法条。

    规则提取两个对比概念（零LLM），分别检索+增强+知识图谱注入。
    """
    print(f"[Agent] 🔍 对比检索...")
    question = state.get("rewritten_question") or state["question"]

    from app.core.retriever.law_graph import (
        inject_related_articles, _extract_law_article_from_doc,
        get_related_articles, _article_to_cn,
    )

    # ── 规则提取两个对比概念（零LLM） ──
    concept_a, concept_b = extract_compare_concepts(question)
    if concept_a == question and concept_b == question:
        print(f"[Agent] 🔍 对比概念提取失败，用完整问题检索")
    else:
        print(f"[Agent] 🔍 对比概念: A='{concept_a}', B='{concept_b}'")

    # ── 分别检索两组 ──
    rerank_top_k = 8

    def _retrieve_concept(query: str) -> list:
        docs = _do_retrieve(retriever, query, settings.TOP_K, settings.SCORE_THRESHOLD, rerank_top_k)
        docs = inject_related_articles(docs, retriever)
        return docs

    docs_a = _retrieve_concept(concept_a)
    docs_b = _retrieve_concept(concept_b)

    # ── 知识图谱"区别对比"关系补充 ──
    for doc, score in docs_a[:5]:
        result = _extract_law_article_from_doc(doc)
        if result:
            law_name, article_num = result
            related = get_related_articles(law_name, article_num)
            for rel_law, rel_article, rel_type, rel_desc in related:
                if rel_type == "区别对比":
                    query = f"{rel_law}{_article_to_cn(rel_article)}"
                    try:
                        extra = _do_retrieve(retriever, query, 3, 0.0, 2)
                        docs_b.extend(extra)
                    except Exception:
                        pass
    if docs_b:
        docs_b = inject_related_articles(docs_b, retriever)

    steps = state.get("steps", [])
    steps.append(f"对比检索: A组{len(docs_a)}个, B组{len(docs_b)}个文档")
    return {"context_docs": docs_a, "context_docs_b": docs_b, "steps": steps}


# ─── 内部辅助 ──

async def _generate_hyde(grader_llm, question: str) -> str | None:
    """HyDE：生成假设性专业答案，用假设答案去检索。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from app.core.generator.prompts import HYDE_PROMPT
        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        chain = prompt | grader_llm
        result = await chain.ainvoke({"question": question, "conversation_context": "（无对话上下文）"})
        hyde = result.content.strip()
        return hyde if hyde and len(hyde) > 10 else None
    except Exception as e:
        print(f"[Agent] 🎭 HyDE生成失败: {e}")
        return None


async def _expand_queries(grader_llm, question: str) -> List[str]:
    """多查询扩展：将复杂问题拆分为多个子查询。"""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from app.core.generator.prompts import MULTI_QUERY_EXPANSION_PROMPT
        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_EXPANSION_PROMPT)
        chain = prompt | grader_llm
        result = await chain.ainvoke({"question": question})
        sub_queries = [line.strip() for line in result.content.strip().split("\n") if line.strip()]
        if not sub_queries:
            return [question]
        print(f"[Agent] 🔍 多查询扩展: '{question[:30]}...' → {sub_queries}")
        return sub_queries
    except Exception as e:
        print(f"[Agent] 🔍 多查询扩展失败: {e}，使用原始查询")
        return [question]


def _fallback_retrieve(retriever: BaseRetriever, question: str) -> list:
    """检索兜底：从问题中提取法律关键词做精确查找。"""
    from app.core.retriever.law_graph import (
        inject_related_articles, _exact_lookup_from_parent_store,
        _LAW_SOURCE_MAP, _CN_NUM,
    )
    fallback_docs = []
    for law_name, source in _LAW_SOURCE_MAP.items():
        if law_name in question:
            art_match = re.search(r'第[一二三四五六七八九十百零\d]+条', question)
            if art_match:
                cn = art_match.group()
                for cn_key, ar_val in _CN_NUM.items():
                    if cn == f"第{cn_key}条":
                        exact = _exact_lookup_from_parent_store(source, cn, k=3)
                        fallback_docs.extend(exact)
                        break
            if not fallback_docs:
                try:
                    extra = _do_retrieve(retriever, f"{law_name}相关规定", 3, 0.0, 3)
                    fallback_docs.extend(extra[:3])
                except Exception:
                    pass
            break

    if not fallback_docs:
        try:
            fallback_docs = _do_retrieve(retriever, question, 5, 0.0, 5)
        except Exception:
            pass

    if fallback_docs:
        fallback_docs = inject_related_articles(fallback_docs, retriever)
        print(f"[Agent] 检索兜底: 补充{len(fallback_docs)}个文档")
    return fallback_docs
