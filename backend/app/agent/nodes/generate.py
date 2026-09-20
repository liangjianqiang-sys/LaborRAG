"""Generate — 生成节点（快速生成 + 检索路径生成）。"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.agent.state import AgentState
from app.agent.prompts import ANSWER_STYLE_RULES, RAG_GENERATE_PROMPT
from app.utils.text import format_docs, normalize_article
from app.retrieval.base import BaseRetriever


async def simple_generate(retriever: BaseRetriever, gen_llm: ChatOpenAI, grader_llm: ChatOpenAI,
                    state: AgentState) -> dict:
    """快速路径：简单问题检索+充分性验证+生成，不经过Agent路由。"""
    question = state.get("rewritten_question") or state["question"]
    original_question = state["question"]
    conversation_context = state.get("conversation_context", "")

    # 检索
    rerank_top_k = 3 if state.get("complexity") == "simple" else 5
    from app.retrieval.reranked import RerankedRetriever
    if isinstance(retriever, RerankedRetriever):
        docs = retriever.retrieve(
            query=question, k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
            rerank_top_k=rerank_top_k,
        )
    else:
        docs = retriever.retrieve(
            query=question, k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
        )

    # ── 概念映射注入（与 retrieve_docs 相同逻辑，确保口语词不丢失） ──
    # 概念注入doc强制置顶，不参与reranker分数排序竞争。
    # 即使法条已在docs中（但排序靠后），也要提升到顶部。
    from app.retrieval.domain_boost import concept_lookup, _extract_law_article_from_doc
    concept_docs = concept_lookup(original_question + " " + question, k=4)
    injected_docs = []
    if concept_docs:
        # 收集概念匹配的法条ID
        concept_article_ids = set()
        for doc, score in concept_docs:
            result = _extract_law_article_from_doc(doc)
            if result:
                concept_article_ids.add(result)

        # 从docs中提取概念匹配的doc（提升到顶部），未匹配的保留
        remaining_docs = []
        for doc, score in docs:
            result = _extract_law_article_from_doc(doc)
            if result and result in concept_article_ids:
                doc.metadata["injected_by"] = "concept_promote"
                injected_docs.append((doc, 0.95))
                print(f"[Agent] 概念提升(快速路径): {result[0]}第{result[1]}条")
            else:
                remaining_docs.append((doc, score))

        # 概念匹配但不在docs中的新法条，也注入
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
                print(f"[Agent] 概念注入(快速路径): {result[0]}第{result[1]}条")

        docs = remaining_docs

    if injected_docs:
        docs = injected_docs + docs
    context = format_docs(docs, fallback="未找到相关参考资料。")

    steps = state.get("steps", [])

    # 两步生成：先验证检索充分性
    if settings.RETRIEVAL_VALIDATION:
        if docs:
            is_sufficient = await _check_retrieval_sufficiency(grader_llm, question, context)
            if not is_sufficient:
                steps.append("检索验证: 资料不足，拒绝回答")
                answer = "根据现有资料无法完整回答您的问题。"
                return {"answer": answer, "context_docs": docs, "steps": steps}
            steps.append("检索验证: 资料充分")
        else:
            steps.append("检索验证: 无检索结果")
            answer = "未找到与您问题相关的法律资料。"
            return {"answer": answer, "context_docs": docs, "steps": steps}

    # 生成
    prompt = ChatPromptTemplate.from_template(RAG_GENERATE_PROMPT)
    chain = prompt | gen_llm
    result = await chain.ainvoke({
        "context": context,
        "question": question,
        "conversation_context": conversation_context or "（无对话上下文）",
        "style_rules": ANSWER_STYLE_RULES,
    })

    answer = result.content
    steps.append(f"快速生成: 基于{len(docs)}个文档直接回答")

    # 轻量验证
    if answer and docs:
        from app.agent.nodes.validate import validate
        validated = validate({"answer": answer, "context_docs": docs, "steps": steps})
        answer = validated.get("answer", answer)
        steps = validated.get("steps", steps)

    return {"answer": answer, "context_docs": docs, "steps": steps}


async def generate_from_retrieval(gen_llm: ChatOpenAI, grader_llm: ChatOpenAI,
                            state: AgentState) -> dict:
    """Retriever路径生成：基于检索文档直接生成回答。"""
    print(f"[Agent] 🤖 检索路径生成回答...")
    question = state.get("rewritten_question") or state["question"]
    docs = state.get("context_docs", [])
    conversation_context = state.get("conversation_context", "")

    context = format_docs(docs, fallback="未找到相关参考资料。")

    steps = state.get("steps", [])

    # 两步生成：先验证检索充分性
    if settings.RETRIEVAL_VALIDATION and docs:
        is_sufficient = await _check_retrieval_sufficiency(grader_llm, question, context)
        if not is_sufficient:
            steps.append("检索验证: 资料不足，拒绝回答")
            answer = "根据现有资料无法完整回答您的问题。"
            return {"answer": answer, "steps": steps}
        steps.append("检索验证: 资料充分")

    prompt = ChatPromptTemplate.from_template(
        "{style_rules}\n\n"
        "对话上下文：{conversation_context}\n\n"
        "参考资料：\n{context}\n\n"
        "问题：{question}\n\n回答："
    )
    chain = prompt | gen_llm
    result = await chain.ainvoke({
        "context": context,
        "question": question,
        "conversation_context": conversation_context or "（无对话上下文）",
        "style_rules": ANSWER_STYLE_RULES,
    })

    steps = state.get("steps", [])
    steps.append(f"生成: 基于{len(docs)}个文档生成回答")
    return {"answer": result.content, "steps": steps}


async def _check_retrieval_sufficiency(grader_llm, question: str, context: str) -> bool:
    """用轻量LLM判断检索结果是否足以回答问题。"""
    if len(context) < 50:
        return False
    from app.agent.prompts import RETRIEVAL_EVALUATION_PROMPT
    prompt = ChatPromptTemplate.from_template(RETRIEVAL_EVALUATION_PROMPT)
    chain = prompt | grader_llm
    result = await chain.ainvoke({"question": question, "context": context[:2000]})
    verdict = result.content.strip()
    return "充分" in verdict and "不足" not in verdict
