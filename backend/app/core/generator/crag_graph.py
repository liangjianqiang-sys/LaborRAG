"""V3: LangGraph CRAG (Corrective RAG) 自我纠错工作流。

工作流图:
    retrieve → grade_documents → [rewrite_query | generate]
                                         ↓              ↓
                                    retrieve(重试)   grade_answer → [output | regenerate]
                                                                          ↓
                                                                    generate(重试)
"""
from typing import List, Tuple, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.config import settings


# ─── 状态定义 ───────────────────────────────────────────────

class CragState(TypedDict, total=False):
    """CRAG工作流状态。"""
    question: str               # 原始问题
    rewritten_question: str     # 改写后的问题
    conversation_context: str   # 对话上下文（多轮对话历史）
    context_docs: List[Tuple[Document, float]]  # 检索到的文档
    context_grade: str          # 检索质量评估: "relevant" | "irrelevant"
    answer: str                 # 生成的回答
    answer_grade: str           # 回答质量评估: "faithful" | "unfaithful"
    rewrite_count: int          # 已改写次数
    regenerate_count: int       # 已重新生成次数
    steps: List[str]            # 工作流步骤记录


MAX_REWRITES = 1    # 最多改写1次
MAX_REGENERATES = 1  # 最多重新生成1次


# ─── Prompt模板 ─────────────────────────────────────────────

RETRIEVAL_GRADE_PROMPT = """你是RAG检索评估分类器，只做三分类，严格按规则判断：

定义：
1. precise：检索片段精准命中问题核心，只包含强相关内容，无大量无关冗余信息，能直接用来完整回答用户问题。
2. vague：检索片段和问题沾边，但范围过宽、混入大量次要/无关内容，不够聚焦，不能直接精准作答，需要改写问题重新检索。
3. irrelevant：检索片段和用户问题主题完全不相关，没有可用于回答的有效信息。

用户问题：{question}
检索参考片段：
{documents}

只输出一个单词：precise、vague、irrelevant，不要解释。"""

QUERY_REWRITE_PROMPT = """你是RAG问题改写专家，针对劳动法场景优化用户提问，用于再次检索知识库。

规则：
1. 把宽泛模糊的口语问题，改写成语义精准、范围收敛的标准专业问句；
2. 只聚焦用户核心意图，剔除无关歧义，缩小检索范围；
3. 保持原意不变，不扩题、不脑补、不新增额外需求；
4. 输出只给改写后的一句话，不要解释、不要多余说明。

对话上下文：{conversation_context}
用户原始问题：{question}
改写后的精准检索问句："""

ANSWER_GRADE_PROMPT = """你是一个回答质量评估专家。请判断AI的回答是否忠实于提供的参考资料。

参考资料：
{context}

AI回答：{answer}

判断标准（宽松原则，宁可放过不可误杀）：
- 回答的主要观点与参考资料主题一致 → faithful
- 回答对资料进行了概括、归纳或引用具体条文 → faithful
- 只有回答编造了与参考资料完全矛盾的核心事实 → unfaithful
- 参考资料是节选，回答引用的条文只要与资料主题相关就视为忠实

请只回答 "faithful"（忠实）或 "unfaithful"（不忠实），不要回答其他内容。"""

RAG_GENERATE_PROMPT = """你是劳动法律师助手。基于以下参考资料回答问题，规则：只引用资料中有的法条，标明出处；无相关内容则说明"现有资料无法回答"。

对话上下文：{conversation_context}

参考资料：
{context}

问题：{question}

回答："""


# ─── CRAG工作流 ─────────────────────────────────────────────

class CRAGGraph:
    """LangGraph CRAG 自我纠错工作流。"""

    def __init__(self, retriever):
        self.retriever = retriever

        # 评估用LLM（temperature=0，确保判断稳定）
        self.grader_llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            temperature=0,
            max_tokens=64,
            request_timeout=60,
        )

        # 生成用LLM
        self.gen_llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            request_timeout=60,
        )

        # 构建graph
        self.graph = self._build_graph()

    # ─── 节点函数 ──────────────────────────────────────────

    def _retrieve(self, state: CragState) -> dict:
        """检索节点：根据问题检索相关文档。"""
        query = state.get("rewritten_question") or state["question"]
        print(f"[CRAG] 🔍 检索节点: query='{query[:50]}'")
        docs = self.retriever.retrieve(
            query=query,
            k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        steps = state.get("steps", [])
        steps.append(f"检索: 使用查询'{query[:30]}...'检索到{len(docs)}个文档")
        return {"context_docs": docs, "steps": steps}

    def _grade_documents(self, state: CragState) -> dict:
        """检索质量评估节点：判断文档是否与问题相关。"""
        print(f"[CRAG] 📋 评估检索质量...")
        question = state["question"]
        docs = state.get("context_docs", [])

        if not docs:
            steps = state.get("steps", [])
            steps.append("评估检索: 未检索到任何文档，标记为irrelevant")
            return {"context_grade": "irrelevant", "steps": steps}

        # 拼接文档内容（截断避免过长）
        doc_texts = []
        for doc, score in docs[:3]:
            doc_texts.append(f"[相关度:{score:.2f}] {doc.page_content[:150]}")
        documents_str = "\n\n".join(doc_texts)

        prompt = ChatPromptTemplate.from_template(RETRIEVAL_GRADE_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({"question": question, "documents": documents_str})

        grade = result.content.strip().lower()
        print(f"[CRAG] 📋 检索评估原始输出: {grade}")
        if "irrelevant" in grade:
            grade = "irrelevant"
        elif "vague" in grade:
            grade = "vague"
        else:
            grade = "precise"

        steps = state.get("steps", [])
        steps.append(f"评估检索: 文档质量={grade}")
        return {"context_grade": grade, "steps": steps}

    def _rewrite_query(self, state: CragState) -> dict:
        """Query改写节点：结合对话上下文改写模糊问题为更精确的查询。"""
        print(f"[CRAG] ✏️ 改写查询...")
        question = state["question"]
        rewrite_count = state.get("rewrite_count", 0)
        conversation_context = state.get("conversation_context", "")

        prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
        })

        rewritten = result.content.strip()
        steps = state.get("steps", [])
        steps.append(f"改写查询: '{question[:30]}...' → '{rewritten[:30]}...'")
        return {
            "rewritten_question": rewritten,
            "rewrite_count": rewrite_count + 1,
            "steps": steps,
        }

    def _generate(self, state: CragState) -> dict:
        """生成节点：基于检索文档和对话上下文生成回答。"""
        regenerate_count = state.get("regenerate_count", 0)
        is_regenerate = state.get("answer") is not None
        if is_regenerate:
            regenerate_count += 1
            print(f"[CRAG] 🤖 重新生成回答(第{regenerate_count}次)...")
        else:
            print(f"[CRAG] 🤖 生成回答...")

        question = state["question"]
        docs = state.get("context_docs", [])
        conversation_context = state.get("conversation_context", "")

        context = "\n\n".join(
            f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc, score in docs
        ) if docs else "未找到相关参考资料。"

        prompt = ChatPromptTemplate.from_template(RAG_GENERATE_PROMPT)
        chain = prompt | self.gen_llm
        result = chain.invoke({
            "context": context,
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
        })

        steps = state.get("steps", [])
        steps.append(f"生成回答: 已基于{len(docs)}个文档生成回答")
        return {"answer": result.content, "steps": steps, "regenerate_count": regenerate_count}

    def _grade_answer(self, state: CragState) -> dict:
        """回答质量评估节点：判断回答是否忠实于原文。"""
        print(f"[CRAG] ✅ 评估回答忠实度...")
        docs = state.get("context_docs", [])
        answer = state.get("answer", "")[:600]

        context = "\n\n".join(
            doc.page_content[:300] for doc, score in docs[:2]
        ) if docs else ""

        prompt = ChatPromptTemplate.from_template(ANSWER_GRADE_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({"context": context, "answer": answer})

        grade = result.content.strip().lower()
        print(f"[CRAG] ✅ 忠实度评估原始输出: {grade}")
        if "unfaithful" in grade:
            grade = "unfaithful"
        else:
            grade = "faithful"

        steps = state.get("steps", [])
        steps.append(f"评估回答: 忠实度={grade}")
        return {"answer_grade": grade, "steps": steps}

    # ─── 条件边 ────────────────────────────────────────────

    def _decide_after_grade_docs(self, state: CragState) -> Literal["rewrite_query", "generate"]:
        """检索评估后的路由：精确→生成，泛泛/不相关→改写（如果还有次数）。"""
        if state.get("context_grade") == "precise":
            return "generate"

        rewrite_count = state.get("rewrite_count", 0)
        if rewrite_count < MAX_REWRITES:
            return "rewrite_query"

        # 已达改写上限，直接生成（用现有文档）
        return "generate"

    def _decide_after_grade_answer(self, state: CragState) -> Literal["regenerate", "output"]:
        """回答评估后的路由：忠实→输出，不忠实→重新生成（如果还有次数）。"""
        if state.get("answer_grade") == "faithful":
            return "output"

        regenerate_count = state.get("regenerate_count", 0)
        if regenerate_count < MAX_REGENERATES:
            return "regenerate"

        # 已达重试上限，直接输出
        return "output"

    # ─── 构建Graph ─────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """构建CRAG状态图。"""
        graph = StateGraph(CragState)

        # 添加节点
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate", self._generate)
        graph.add_node("grade_answer", self._grade_answer)

        # 设置入口
        graph.set_entry_point("retrieve")

        # 添加边
        graph.add_edge("retrieve", "grade_documents")

        # 检索评估 → 改写 or 生成
        graph.add_conditional_edges(
            "grade_documents",
            self._decide_after_grade_docs,
            {
                "rewrite_query": "rewrite_query",
                "generate": "generate",
            },
        )

        # 改写后重新检索
        graph.add_edge("rewrite_query", "retrieve")

        # 生成后评估回答
        graph.add_edge("generate", "grade_answer")

        # 回答评估 → 输出 or 重新生成
        graph.add_conditional_edges(
            "grade_answer",
            self._decide_after_grade_answer,
            {
                "regenerate": "generate",
                "output": END,
            },
        )

        return graph.compile()

    # ─── 对外接口 ──────────────────────────────────────────

    def run(self, question: str, conversation_context: str = "") -> dict:
        """运行CRAG工作流，返回结果。

        Args:
            question: 用户问题
            conversation_context: 对话上下文字符串
        """
        initial_state: CragState = {
            "question": question,
            "conversation_context": conversation_context,
            "rewrite_count": 0,
            "regenerate_count": 0,
            "steps": [],
        }
        result = self.graph.invoke(initial_state)
        return {
            "answer": result.get("answer", ""),
            "context_docs": result.get("context_docs", []),
            "steps": result.get("steps", []),
            "rewritten_question": result.get("rewritten_question", ""),
        }
