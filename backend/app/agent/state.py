"""AgentState — Agentic RAG 工作流状态契约。

所有节点之间通信的唯一数据结构，后续所有拆分以此为准。
"""
from typing import List, Tuple

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Agentic RAG 工作流状态。"""

    question: str                                   # [入口写入] 原始问题
    rewritten_question: str                         # [rewrite_query 写入] → 所有人读
    sub_queries: List[str]                          # [rewrite_query 写入] → retrieve 读（多查询拆解结果）
    conversation_context: str                       # [入口写入] 对话上下文
    intent: str                                     # [route_intent 写入] → decide_intent 读
    complexity: str                                 # [classify_complexity 写入] → decide_after_classify 读
    context_docs: List[Tuple[Document, float]]      # [retrieve 写入] → generate/calculate/compare/validate 读
    context_docs_b: List[Tuple[Document, float]]    # [retrieve_for_compare 写入] → compare 读
    calculation_result: str                         # [calculate 写入] → run() 输出
    answer: str                                     # [generate/calculate/compare/simple_generate 写入] → validate 读/覆盖 → run() 输出
    steps: List[str]                                # [所有节点读写]
