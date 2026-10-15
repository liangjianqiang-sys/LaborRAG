"""Calculate — 计算节点。

参数抽取的两条路径
------------------
① **Function Calling（主）**：由 LLM 决定调哪个计算工具、传什么参数，
   代码只负责执行 —— 即「LLM 做决策、代码做执行」。
   相比正则，它能正确理解「月薪八千」「1万5」这类口语表达，
   也能自主判断该调一个工具还是多个。

② **正则兜底（备）**：模型未产生 tool_calls、或工具调用不可用时，
   回落到 `labor_calculator.auto_calculate` 的规则提取。

为什么必须保留兜底：工具调用依赖服务端的 function calling 能力，
`bind_tools` 只是把 schema 绑上去，**真正的失败发生在调用时** ——
一旦换模型或网关不支持，链路不能直接断掉。
"""
import re
from typing import TYPE_CHECKING, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from app.agent.state import AgentState
from app.agent.prompts import ANSWER_STYLE_RULES, CALCULATOR_PROMPT
from app.utils.text import format_docs
from app.tools.labor_calculator import auto_calculate
from app.tools.labor_tools import (
    LABOR_TOOLS,
    execute_tool_calls,
    format_tool_results,
    summarize_tool_calls,
)
from app.retrieval.base import BaseRetriever

if TYPE_CHECKING:
    # langchain_openai 顶层导入实测约 17s（连带 transformers + torch，见
    # rag_engine.py 的同类注释），而这里只在函数签名里用作注解 ——
    # 走 TYPE_CHECKING + 字符串注解，让 import app.agent.nodes.* 保持廉价。
    from langchain_openai import ChatOpenAI


# 工具调用的系统提示词。放 agent 层而不是 tools 层 —— 后者不许依赖本层。
_TOOL_CALL_SYSTEM_PROMPT = """你是劳动法计算助手。请根据用户问题调用合适的计算工具。

规则：
1. 只有当用户给出了**具体数字**（月薪 / 工作年限 / 加班小时数）时才调用工具；
   缺少关键参数时不要臆测，直接说明缺少哪些参数。
2. **例外（估算场景）**：如果用户给了月薪、但缺少工作年限，而问题明确在问金额
   （如"能拿多少补偿"），可以代入典型年限（1 年 / 3 年 / 5 年）**分别调用工具**，
   给用户一个参考区间 —— 但必须在回答中注明这些年限是假设值，并提示用户提供实际年限。
   注意：**月薪不能假设**（个体差异过大），缺月薪时只能向用户索要。
3. 必要时可以一次调用多个工具 —— 例如用户既问经济补偿金又问赔偿金。
4. 追问场景要结合对话上下文补全参数（例如上文已给月薪，本轮只问"那赔偿金呢"）。

对话上下文：
{conversation_context}"""


async def _calculate_via_tools(
    gen_llm: "ChatOpenAI", question: str, conversation_context: str
) -> Tuple[str, List[str]]:
    """让模型决定调用哪些计算工具，返回 (结果文本, 调用轨迹)。

    模型没产生任何工具调用时返回 ("", [])，由调用方决定是否回落正则路径。
    异常不在这里吞掉 —— 向上抛，由调用方统一处理「工具调用不可用」。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm_with_tools = gen_llm.bind_tools(LABOR_TOOLS)
    messages = [
        SystemMessage(content=_TOOL_CALL_SYSTEM_PROMPT.format(
            conversation_context=conversation_context or "（无对话上下文）")),
        HumanMessage(content=question),
    ]
    response = await llm_with_tools.ainvoke(messages)

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        return "", []

    executed = execute_tool_calls(tool_calls)
    return format_tool_results(executed), summarize_tool_calls(executed)


async def calculate(retriever: BaseRetriever, gen_llm: "ChatOpenAI", state: AgentState) -> dict:
    """Calculator Agent：检索法条 + 精确计算。"""
    print(f"[Agent] 🧮 计算Agent执行...")
    question = state.get("rewritten_question") or state["question"]
    conversation_context = state.get("conversation_context", "")
    docs = state.get("context_docs", [])

    # ── ① 主路径：Function Calling ──
    calc_result, tool_trace = "", []
    try:
        calc_result, tool_trace = await _calculate_via_tools(
            gen_llm, question, conversation_context
        )
    except Exception as e:
        print(f"[Agent] 🧮 工具调用不可用，回落正则提取: {e}")

    # ── ② 兜底：正则提取（模型未调工具 / 工具调用失败）──
    if calc_result:
        print(f"[Agent] 🧮 工具调用: {'、'.join(tool_trace)}")
        print(f"[Agent] 🧮 计算结果: {calc_result[:100]}...")
    else:
        calc_result = auto_calculate(question, conversation_context)
        print(f"[Agent] 🧮 正则提取结果: {calc_result[:100]}...")

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
    if tool_trace:
        steps.append(f"工具调用: {'、'.join(tool_trace)}")
    return {"answer": result.content, "calculation_result": calc_result, "steps": steps}
