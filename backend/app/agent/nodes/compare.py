"""Compare — 对比分析节点。"""
from typing import TYPE_CHECKING
from langchain_core.prompts import ChatPromptTemplate

from app.agent.state import AgentState
from app.agent.prompts import ANSWER_STYLE_RULES, COMPARATOR_PROMPT
from app.utils.text import format_docs

if TYPE_CHECKING:
    # langchain_openai 顶层导入实测约 17s（连带 transformers + torch，见
    # rag_engine.py 的同类注释），而这里只在函数签名里用作注解 ——
    # 走 TYPE_CHECKING + 字符串注解，让 import app.agent.nodes.* 保持廉价。
    from langchain_openai import ChatOpenAI


async def compare(gen_llm: "ChatOpenAI", state: AgentState) -> dict:
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
