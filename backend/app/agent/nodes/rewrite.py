"""Rewrite — 上下文感知查询改写节点（法言法语翻译 + 多查询拆解，一次JSON输出）。"""
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState
from app.agent.prompts import CONTEXT_QUERY_REWRITE_PROMPT


async def rewrite_query(grader_llm: ChatOpenAI, state: AgentState) -> dict:
    """查询改写：法言法语翻译 + 多查询拆解，一次LLM调用输出JSON。

    输出写入 state：
    - rewritten_question: 由 legal_terms 拼接的法言法语查询
    - sub_queries: 拆解后的子查询列表（供 retrieve 节点使用）
    """
    question = state["question"]
    conversation_context = state.get("conversation_context", "")

    # 无上下文且问题已经足够专业/长，跳过改写
    if not conversation_context and len(question) > 30 and _looks_professional(question):
        steps = state.get("steps", [])
        steps.append("查询改写: 跳过（问题已专业）")
        return {"rewritten_question": question, "sub_queries": [question], "steps": steps}

    print(f"[Agent] ✏️ 查询改写+拆解...")
    prompt = ChatPromptTemplate.from_template(CONTEXT_QUERY_REWRITE_PROMPT)
    chain = prompt | grader_llm
    result = await chain.ainvoke({
        "question": question,
        "conversation_context": conversation_context or "（无对话上下文）",
    })

    rewritten = question
    sub_queries = [question]
    legal_terms = []

    # 解析JSON输出
    try:
        content = result.content.strip()
        # 去除可能的markdown代码块标记
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(content)

        # 提取 legal_terms
        legal_terms = parsed.get("legal_terms", [])
        if legal_terms:
            # 用法言法语术语拼接改写后的查询
            rewritten = " ".join(legal_terms)

        # 提取 sub_queries
        sub_queries = parsed.get("sub_queries", [])
        if not sub_queries:
            sub_queries = [rewritten]
    except (json.JSONDecodeError, KeyError, TypeError):
        # JSON解析失败，用原始内容作为改写结果
        content = result.content.strip()
        if content and content != question:
            rewritten = content
            sub_queries = [content]

    if rewritten == question and sub_queries == [question]:
        step_msg = "查询改写: 无需改写"
    else:
        step_msg = f"查询改写: '{question[:30]}...' → '{rewritten[:30]}...'"
        if legal_terms:
            step_msg += f" (术语: {', '.join(legal_terms[:3])})"
        if len(sub_queries) > 1:
            step_msg += f" (拆解{len(sub_queries)}个子查询)"
        print(f"[Agent] ✏️ 改写: {question} → {rewritten}")
        if legal_terms:
            print(f"[Agent] ✏️ 法言法语: {legal_terms}")
        print(f"[Agent] ✏️ 子查询: {sub_queries}")

    steps = state.get("steps", [])
    steps.append(step_msg)
    return {"rewritten_question": rewritten, "sub_queries": sub_queries, "steps": steps}


def _looks_professional(question: str) -> bool:
    """快速判断问题是否已包含法律术语（跳过改写的启发式）。"""
    legal_terms = ["法条", "合同", "补偿", "赔偿", "解除", "终止", "保险",
                   "仲裁", "条例", "规定", "法", "条款", "第", "《"]
    return any(t in question for t in legal_terms)
