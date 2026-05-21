"""V3: LangGraph Adaptive RAG (自适应检索增强生成)。

工作流图:
    classify_complexity → [simple | medium | complex]
        │                    │         │          │
        │                    ↓         ↓          ↓
        │              direct_answer  retrieve   retrieve(短查询自动HyDE)
        │                  │         ↓              ↓
        │                  │      compress      compress
        │                  │         ↓              ↓
        │                  │      generate     grade_documents → [HyDE重试 | generate]
        │                  │         ↓                              ↓
        │                  │        END                         grade_answer → [output | regenerate]
        │                  │                                                        ↓
        │                  END                                               generate(重试)

Adaptive RAG：根据查询复杂度动态选择处理深度，简单查询省40%+token。
HyDE条件触发：短查询(≤15字)自动生成假设答案用于检索，长查询直接检索。
EmbeddingsFilter：检索后用Embedding相似度过滤低相关文档，减少喂给LLM的无关内容。
"""
from typing import List, Tuple, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.config import settings
from app.core.retriever.compressor import EmbeddingsCompressor


# ─── 状态定义 ───────────────────────────────────────────────

class CragState(TypedDict, total=False):
    """Adaptive RAG工作流状态。"""
    question: str               # 原始问题
    rewritten_question: str     # 改写后的问题
    conversation_context: str   # 对话上下文（多轮对话历史）
    complexity: str             # 查询复杂度: "simple" | "medium" | "complex"
    context_docs: List[Tuple[Document, float]]  # 检索到的文档
    context_grade: str          # 检索质量评估: "precise" | "vague" | "irrelevant"
    answer: str                 # 生成的回答
    answer_grade: str           # 回答质量评估: "faithful" | "unfaithful"
    rewrite_count: int          # 已改写次数
    regenerate_count: int       # 已重新生成次数
    steps: List[str]            # 工作流步骤记录


MAX_REWRITES = 1    # 最多改写1次
MAX_REGENERATES = 1  # 最多重新生成1次
HYDE_QUERY_LENGTH_THRESHOLD = 15  # HyDE触发阈值：查询≤15字时自动触发
GRADE_ANSWER_CONTEXT_CHARS = 300 # 评估回答时上下文截断字数
GRADE_ANSWER_ANSWER_CHARS = 600  # 评估回答时回答截断字数


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

HYDE_PROMPT = """你是劳动法律师，请根据用户问题写一段假设性的专业回答（即使你不确定具体法条，也要用法律专业术语给出合理的假设答案）。

规则：
1. 用法言法语回答，包含可能涉及的法律条文编号、专业术语
2. 回答要具体、详细，不要笼统概括
3. 不需要回答完全正确，但必须包含与问题相关的法律关键词和法条引用
4. 只输出假设答案，不要解释、不要多余说明

对话上下文：{conversation_context}
用户问题：{question}
假设性专业回答："""

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

COMPLEXITY_PROMPT = """你是查询复杂度分类器，判断用户问题的复杂度，只输出一个词。

定义：
1. simple：常识性问题，无需检索法律条文即可回答（如"劳动法有多少条"、"什么是劳动合同"）
2. medium：需要检索具体法条，但问题明确单一（如"加班工资怎么算"、"第四十四条怎么规定的"）
3. complex：需要多步推理、对比分析或综合多个法条（如"违法解除和合法解除的赔偿区别"、"经济补偿金和赔偿金有什么不同"）

对话上下文：{conversation_context}
用户问题：{question}

只输出一个词：simple、medium、complex，不要解释。"""

RAG_GENERATE_PROMPT = """你是劳动法律师助手。基于以下参考资料回答问题，规则：只引用资料中有的法条，标明出处；无相关内容则说明"现有资料无法回答"。

对话上下文：{conversation_context}

参考资料：
{context}

问题：{question}

回答："""


# ─── CRAG工作流 ─────────────────────────────────────────────

class AdaptiveRAGGraph:
    """LangGraph Adaptive RAG 自适应检索增强生成。"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.compressor = EmbeddingsCompressor()

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

    def _classify_complexity(self, state: CragState) -> dict:
        """复杂度分类节点：判断查询是simple/medium/complex，动态选择处理深度。"""
        print(f"[Adaptive] 🎯 分类查询复杂度...")
        question = state["question"]
        conversation_context = state.get("conversation_context", "")

        prompt = ChatPromptTemplate.from_template(COMPLEXITY_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
        })

        complexity = result.content.strip().lower()
        print(f"[Adaptive] 🎯 复杂度分类原始输出: {complexity}")
        if "complex" in complexity:
            complexity = "complex"
        elif "medium" in complexity:
            complexity = "medium"
        else:
            complexity = "simple"

        steps = state.get("steps", [])
        steps.append(f"复杂度分类: {complexity}")
        print(f"[Adaptive] 🎯 查询复杂度={complexity}")
        return {"complexity": complexity, "steps": steps}

    def _direct_answer(self, state: CragState) -> dict:
        """简单查询直接回答节点：无需检索，LLM直接生成。"""
        print(f"[Adaptive] 💬 简单查询直接回答...")
        question = state["question"]
        conversation_context = state.get("conversation_context", "")

        prompt = ChatPromptTemplate.from_template(
            "你是劳动法律师助手。请简洁回答以下问题，如果涉及具体法条请标明出处。\n\n"
            "对话上下文：{conversation_context}\n"
            "问题：{question}\n\n回答："
        )
        chain = prompt | self.gen_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
        })

        steps = state.get("steps", [])
        steps.append("直接回答: 跳过检索，LLM直接生成")
        return {"answer": result.content, "steps": steps}

    def _retrieve(self, state: CragState) -> dict:
        """检索节点：根据问题检索相关文档。

        HyDE条件触发：短查询（≤15字）自动生成假设答案用于检索，
        长查询/已有HyDE结果直接检索。
        """
        question = state["question"]
        hyde_query = state.get("rewritten_question")

        # HyDE条件触发：短查询且尚未生成假设答案时，先触发HyDE
        if not hyde_query and len(question) <= HYDE_QUERY_LENGTH_THRESHOLD:
            print(f"[CRAG] 🔍 短查询检测({len(question)}字)，触发HyDE...")
            hyde_state = self._rewrite_query(state)
            hyde_query = hyde_state.get("rewritten_question", question)
            state = {**state, **hyde_state}

        query = hyde_query or question
        print(f"[CRAG] 🔍 检索节点: query='{query[:50]}'")
        docs = self.retriever.retrieve(
            query=query,
            k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        steps = state.get("steps", [])
        steps.append(f"检索: 使用查询'{query[:30]}...'检索到{len(docs)}个文档")
        return {"context_docs": docs, "steps": steps,
                "rewritten_question": hyde_query or "",
                "rewrite_count": state.get("rewrite_count", 0)}

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
        """HyDE节点：生成假设性专业答案，用假设答案去检索（替代传统Query改写）。

        HyDE核心思想：短查询/口语化查询的embedding与法条向量距离远，
        但假设答案（含法言法语）的embedding与法条向量距离近，检索精度大幅提升。
        """
        print(f"[CRAG] 🎭 HyDE生成假设答案...")
        question = state["question"]
        rewrite_count = state.get("rewrite_count", 0)
        conversation_context = state.get("conversation_context", "")

        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
        })

        hyde_answer = result.content.strip()
        steps = state.get("steps", [])
        steps.append(f"HyDE: '{question[:30]}...' → 假设答案'{hyde_answer[:30]}...'")
        print(f"[CRAG] 🎭 HyDE假设答案: {hyde_answer[:80]}...")
        return {
            "rewritten_question": hyde_answer,
            "rewrite_count": rewrite_count + 1,
            "steps": steps,
        }

    def _compress(self, state: CragState) -> dict:
        """上下文压缩节点：用Embedding相似度过滤低相关文档，减少喂给LLM的无关内容。"""
        print(f"[CRAG] ✂️ 上下文压缩...")
        question = state["question"]
        docs = state.get("context_docs", [])

        if not docs:
            steps = state.get("steps", [])
            steps.append("压缩: 无文档可压缩")
            return {"context_docs": docs, "steps": steps}

        original_count = len(docs)
        compressed = self.compressor.compress(question, docs)

        # 如果压缩后为空，保留原始结果（宁可多喂也不无内容）
        if not compressed:
            print(f"[CRAG] ✂️ 压缩后为空，保留原始{original_count}个文档")
            compressed = docs

        steps = state.get("steps", [])
        steps.append(f"压缩: {original_count} → {len(compressed)}个文档")
        return {"context_docs": compressed, "steps": steps}

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
        answer = state.get("answer", "")[:GRADE_ANSWER_ANSWER_CHARS]

        context = "\n\n".join(
            doc.page_content[:GRADE_ANSWER_CONTEXT_CHARS] for doc, score in docs[:2]
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

    def _decide_after_classify(self, state: CragState) -> Literal["direct_answer", "medium_retrieve", "retrieve"]:
        """复杂度分类后的路由：simple→直接回答，medium→单次检索+生成，complex→完整CRAG。"""
        complexity = state.get("complexity", "medium")
        if complexity == "simple":
            return "direct_answer"
        elif complexity == "medium":
            return "medium_retrieve"
        else:
            return "retrieve"

    def _decide_after_generate(self, state: CragState) -> Literal["grade_answer", "output"]:
        """生成后的路由：medium→直接输出，complex→忠实度评估。"""
        complexity = state.get("complexity", "complex")
        if complexity == "medium":
            return "output"
        return "grade_answer"

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
        """构建Adaptive RAG状态图。"""
        graph = StateGraph(CragState)

        # ── 节点 ──
        graph.add_node("classify_complexity", self._classify_complexity)
        graph.add_node("direct_answer", self._direct_answer)
        # medium路径节点
        graph.add_node("medium_retrieve", self._retrieve)
        graph.add_node("medium_compress", self._compress)
        graph.add_node("medium_generate", self._generate)
        # complex路径节点（完整CRAG）
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("compress", self._compress)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate", self._generate)
        graph.add_node("grade_answer", self._grade_answer)

        # ── 入口 → 复杂度分类 ──
        graph.set_entry_point("classify_complexity")

        # ── 分类后三路路由 ──
        graph.add_conditional_edges(
            "classify_complexity",
            self._decide_after_classify,
            {
                "direct_answer": "direct_answer",
                "medium_retrieve": "medium_retrieve",
                "retrieve": "retrieve",
            },
        )

        # ── simple路径 ──
        graph.add_edge("direct_answer", END)

        # ── medium路径：检索→压缩→生成→END ──
        graph.add_edge("medium_retrieve", "medium_compress")
        graph.add_edge("medium_compress", "medium_generate")
        graph.add_edge("medium_generate", END)

        # ── complex路径：完整CRAG ──
        graph.add_edge("retrieve", "compress")
        graph.add_edge("compress", "grade_documents")

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

        # 生成后：medium→END, complex→grade_answer
        graph.add_conditional_edges(
            "generate",
            self._decide_after_generate,
            {
                "grade_answer": "grade_answer",
                "output": END,
            },
        )

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
        """运行Adaptive RAG工作流，返回结果。

        Args:
            question: 用户问题
            conversation_context: 对话上下文字符串
        """
        initial_state: CragState = {
            "question": question,
            "conversation_context": conversation_context,
            "complexity": "",
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
