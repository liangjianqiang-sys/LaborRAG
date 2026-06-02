"""Compare — 对比分析节点。"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.generator.state import AgentState
from app.core.generator.prompts import ANSWER_STYLE_RULES, COMPARATOR_PROMPT
from app.core.generator.helpers import format_docs


async def compare(gen_llm: ChatOpenAI, state: AgentState) -> dict:
    """Comparator Agent：对比分析两组法条。"""
    print(f"[Agent] ⚖️ 对比Agent执行...")
    question = state.get("rewritten_question") or state["question"]
    docs_a = state.get("context_docs", [])
    docs_b = state.get("context_docs_b", [])

    context_a = format_docs(docs_a)
    context_b = format_docs(docs_b)

    prompt = ChatPromptTemplate.from_template(COMPARATOR_PROMPT)
    chain = prompt | gen_llm
    result = await chain.ainvoke({
        "context_a": context_a,
        "context_b": context_b,
        "question": question,
        "style_rules": ANSWER_STYLE_RULES,
    })

    steps = state.get("steps", [])
    steps.append("对比: 已生成结构化对比分析")
    return {"answer": result.content, "steps": steps}
