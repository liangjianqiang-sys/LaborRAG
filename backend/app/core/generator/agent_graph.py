"""LangGraph Agentic RAG — 状态机编排器。

职责：实例化重型组件 → 注入给节点 → 组装图 → 运行。
不包含任何业务逻辑，所有节点逻辑在 nodes/ 中实现。
"""
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.config import settings
from app.core.generator.state import AgentState
from app.core.generator.helpers import strip_unsolicited_sections
from app.core.generator.guardrails import AnswerGuardrails
from app.core.retriever.base import BaseRetriever

# 节点函数
from app.core.generator.nodes.classify import classify_complexity, decide_after_classify
from app.core.generator.nodes.rewrite import rewrite_query
from app.core.generator.nodes.router import route_intent, decide_intent
from app.core.generator.nodes.retrieve import retrieve_docs, retrieve_for_compare
from app.core.generator.nodes.generate import simple_generate, generate_from_retrieval
from app.core.generator.nodes.calculate import calculate
from app.core.generator.nodes.compare import compare
from app.core.generator.nodes.validate import validate


# ─── Agentic RAG工作流 ──────────────────────────────────────

class AgenticRAGGraph:
    """LangGraph Agentic RAG 编排器。"""

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

        # 分类/验证用LLM（轻量模型）
        self.grader_llm = ChatOpenAI(
            model=settings.EVAL_LLM_MODEL_NAME,
            openai_api_key=settings.EVAL_LLM_API_KEY or settings.LLM_API_KEY,
            openai_api_base=settings.EVAL_LLM_API_BASE or settings.LLM_API_BASE,
            temperature=0, max_tokens=128, request_timeout=60,
            extra_body={"enable_thinking": False},
        )

        # 生成用LLM
        self.gen_llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            request_timeout=60,
            extra_body={"enable_thinking": False},
        )

        self.graph = self._build_graph()

    # ─── 节点适配器（闭包注入依赖） ──────────────────────────

    async def _node_rewrite(self, state: AgentState) -> dict:
        return await rewrite_query(self.grader_llm, state)

    def _node_classify(self, state: AgentState) -> dict:
        return classify_complexity(state)

    async def _node_simple_generate(self, state: AgentState) -> dict:
        return await simple_generate(self.retriever, self.gen_llm, self.grader_llm, state)

    async def _node_route_intent(self, state: AgentState) -> dict:
        return await route_intent(self.grader_llm, state)

    async def _node_retrieve_docs(self, state: AgentState) -> dict:
        return await retrieve_docs(self.retriever, self.grader_llm, state)

    async def _node_retrieve_for_compare(self, state: AgentState) -> dict:
        return await retrieve_for_compare(self.retriever, state)

    async def _node_generate_from_retrieval(self, state: AgentState) -> dict:
        return await generate_from_retrieval(self.gen_llm, self.grader_llm, state)

    async def _node_calculate(self, state: AgentState) -> dict:
        return await calculate(self.retriever, self.gen_llm, state)

    async def _node_compare(self, state: AgentState) -> dict:
        return await compare(self.gen_llm, state)

    def _node_validate(self, state: AgentState) -> dict:
        return validate(state)

    # ─── 构建Graph ─────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # 节点
        graph.add_node("rewrite_query", self._node_rewrite)
        graph.add_node("classify_complexity", self._node_classify)
        graph.add_node("simple_generate", self._node_simple_generate)
        graph.add_node("route_intent", self._node_route_intent)
        graph.add_node("retrieve_docs", self._node_retrieve_docs)
        graph.add_node("retrieve_for_calc", self._node_retrieve_docs)
        graph.add_node("retrieve_for_compare", self._node_retrieve_for_compare)
        graph.add_node("generate_from_retrieval", self._node_generate_from_retrieval)
        graph.add_node("calculate", self._node_calculate)
        graph.add_node("compare", self._node_compare)
        graph.add_node("validate", self._node_validate)

        # 入口
        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "classify_complexity")

        # 复杂度路由
        graph.add_conditional_edges(
            "classify_complexity", decide_after_classify,
            {"simple_path": "simple_generate", "agent_path": "route_intent"},
        )
        graph.add_edge("simple_generate", END)

        # 意图路由
        graph.add_conditional_edges(
            "route_intent", decide_intent,
            {"retrieve_path": "retrieve_docs", "calculate_path": "retrieve_for_calc", "compare_path": "retrieve_for_compare"},
        )

        # 各路径 → validate → END
        graph.add_edge("retrieve_docs", "generate_from_retrieval")
        graph.add_edge("generate_from_retrieval", "validate")
        graph.add_edge("retrieve_for_calc", "calculate")
        graph.add_edge("calculate", "validate")
        graph.add_edge("retrieve_for_compare", "compare")
        graph.add_edge("compare", "validate")
        graph.add_edge("validate", END)

        return graph.compile()

    # ─── 对外接口 ──────────────────────────────────────────

    async def run(self, question: str, conversation_context: str = "", skip_guardrails: bool = False) -> dict:
        """运行Agentic RAG工作流。"""
        initial_state: AgentState = {
            "question": question,
            "conversation_context": conversation_context,
            "steps": [],
        }
        result = await self.graph.ainvoke(initial_state)
        answer = result.get("answer", "")
        answer = strip_unsolicited_sections(answer)
        if not skip_guardrails:
            answer = AnswerGuardrails.check(answer)
        return {
            "answer": answer,
            "context_docs": result.get("context_docs", []),
            "steps": result.get("steps", []),
            "intent": result.get("intent", ""),
            "rewritten_question": result.get("rewritten_question", ""),
            "calculation_result": result.get("calculation_result", ""),
        }

