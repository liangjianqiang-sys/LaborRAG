"""Router — 意图路由节点 + 意图分发。"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState
from app.agent.prompts import INTENT_ROUTER_PROMPT


async def route_intent(grader_llm: ChatOpenAI, state: AgentState) -> dict:
    """Router Agent：意图识别，决定分派给哪个专业Agent。"""
    print(f"[Agent] 🧭 路由意图识别...")
    question = state.get("rewritten_question") or state["question"]
    conversation_context = state.get("conversation_context", "")

    prompt = ChatPromptTemplate.from_template(INTENT_ROUTER_PROMPT)
    chain = prompt | grader_llm
    result = await chain.ainvoke({
        "question": question,
        "conversation_context": conversation_context or "（无对话上下文）",
    })

    intent = result.content.strip().lower()
    print(f"[Agent] 🧭 意图识别原始输出: {intent}")
    if "calculate" in intent:
        intent = "calculate"
    elif "compare" in intent:
        intent = "compare"
    else:
        intent = "retrieve"

    steps = state.get("steps", [])
    steps.append(f"意图路由: {intent}")
    print(f"[Agent] 🧭 意图={intent}")
    return {"intent": intent, "steps": steps}


def decide_intent(state: AgentState) -> Literal[
    "retrieve_path", "calculate_path", "compare_path"
]:
    """意图路由后的分发。"""
    intent = state.get("intent", "retrieve")
    if intent == "calculate":
        return "calculate_path"
    elif intent == "compare":
        return "compare_path"
    return "retrieve_path"
