"""Calculate — 计算节点。"""
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState
from app.agent.prompts import ANSWER_STYLE_RULES, CALCULATOR_PROMPT
from app.utils.text import format_docs
from app.tools.labor_calculator import auto_calculate
from app.retrieval.base import BaseRetriever


async def calculate(retriever: BaseRetriever, gen_llm: ChatOpenAI, state: AgentState) -> dict:
    """Calculator Agent：检索法条 + 精确计算。"""
    print(f"[Agent] 🧮 计算Agent执行...")
    question = state.get("rewritten_question") or state["question"]
    conversation_context = state.get("conversation_context", "")
    docs = state.get("context_docs", [])

    # 执行自动计算
    calc_result = auto_calculate(question, conversation_context)
    print(f"[Agent] 🧮 计算结果: {calc_result[:100]}...")

    # ── 计算类检索兜底 ──
    if not docs and "法条依据" in calc_result:
        law_refs = re.findall(r"《[^》]+》第[\d]+条", calc_result)
        if law_refs:
            from app.retrieval.reranked import RerankedRetriever
            seen_doc_ids = set()
            for ref in law_refs:
                try:
                    if isinstance(retriever, RerankedRetriever):
                        extra = retriever.retrieve(query=ref, k=3, score_threshold=0.0, rerank_top_k=2)
                    else:
                        extra = retriever.retrieve(query=ref, k=3, score_threshold=0.0)
                    for doc, score in extra:
                        doc_id = doc.metadata.get("doc_id", doc.page_content[:80])
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            docs.append((doc, score))
                except Exception as e:
                    print(f"[Agent] 🧮 法条补充检索失败 {ref}: {e}")

            if docs:
                from app.retrieval.domain_boost import inject_related_articles
                docs = inject_related_articles(docs, retriever)
                print(f"[Agent] 🧮 计算兜底检索: 补充{len(docs)}个文档")

    # 拼接法条上下文
    context = format_docs(docs)

    # LLM整合法条+计算结果
    prompt = ChatPromptTemplate.from_template(CALCULATOR_PROMPT)
    chain = prompt | gen_llm
    result = await chain.ainvoke({
        "context": context,
        "calculation_result": calc_result,
        "question": question,
        "conversation_context": conversation_context or "（无对话上下文）",
        "style_rules": ANSWER_STYLE_RULES,
    })

    steps = state.get("steps", [])
    steps.append(f"计算: {calc_result[:60]}...")
    return {"answer": result.content, "calculation_result": calc_result, "steps": steps}
