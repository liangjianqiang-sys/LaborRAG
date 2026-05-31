"""V4: LangGraph Agentic RAG — 多Agent协作架构。

工作流图:
    route_intent → [retriever | calculator | comparator]
         ↓              ↓             ↓              ↓
    (意图路由)    (V3 CRAG检索)  (检索+计算)   (检索A+检索B+对比)
                       ↓             ↓              ↓
                  validator ← ← ← ← ← ← ← ← ← ← ←
                       ↓
                    output

Router Agent: LLM意图识别，分派到专业Agent
Retriever Agent: 复用V3 AdaptiveRAG检索能力
Calculator Agent: 检索法条公式 → Python精确计算
Comparator Agent: 分别检索两方法条 → 结构化对比
Validator Agent: 合法性+幻觉最终验证
"""
import re
import math
from typing import List, Tuple, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from app.config import settings
from app.core.retriever.compressor import EmbeddingsCompressor
from app.core.generator.prompts import ANSWER_STYLE_RULES, RETRIEVAL_EVALUATION_PROMPT, MULTI_QUERY_EXPANSION_PROMPT
from app.core.generator.guardrails import AnswerGuardrails


# ─── 状态定义 ───────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Agentic RAG工作流状态。"""
    question: str                           # 原始问题
    rewritten_question: str                 # 改写后的问题（追问补全）
    conversation_context: str               # 对话上下文
    intent: str                             # 意图分类: "retrieve" | "calculate" | "compare"
    complexity: str                         # 复杂度: "simple" | "medium" | "complex"
    context_docs: List[Tuple[Document, float]]  # 检索到的文档
    context_docs_b: List[Tuple[Document, float]]  # 对比用第二组文档
    calculation_result: str                 # 计算结果
    comparison_result: str                  # 对比结果
    answer: str                             # 最终回答
    validation_passed: bool                 # 验证是否通过
    steps: List[str]                        # 工作流步骤记录


# ─── Prompt模板 ─────────────────────────────────────────────

# 复杂度分类（复用CRAG的分类逻辑）
COMPLEXITY_PROMPT = """你是劳动法领域的查询复杂度分类器，判断用户问题的复杂度，只输出一个词。

定义：
1. simple：常识性/定义性问题，无需检索法律条文即可回答
   例："劳动法是什么"、"劳动合同有哪些类型"
   判断要点：不涉及具体条款或金额计算

2. medium：需要检索具体法条，但问题明确单一
   例："劳动合同法第47条怎么规定的"、"加班工资怎么算"、"试用期最长多久"
   判断要点：指向单一法条或单一法律概念
   ★ 包含"第X条"的问题 → 总是 medium

3. complex：需要多步推理、对比分析、综合多个法条或涉及金额计算
   例："经济补偿金和赔偿金有什么区别"、"被公司辞退了能拿多少钱"、
       "月薪8000加班10小时加班费多少"、"违法解除劳动合同有哪些法律后果"
   判断要点：涉及两个以上法条对比、需要计算具体金额、包含"区别""怎么维权"

只输出一个词：simple、medium、complex，不要解释。"""

QUERY_REWRITE_PROMPT = """你是对话上下文感知的查询改写器。根据对话历史，将用户的追问/补充问题改写为完整的独立问题。

规则：
1. 如果用户问题是追问（缺少主语/上下文），结合对话历史补全为完整问题
2. 保留用户补充的新信息（如具体金额、年限等参数）
3. 如果问题本身已经完整，直接原样输出
4. 只输出改写后的问题，不要解释

对话上下文：
{conversation_context}

用户问题：{question}

改写后的完整问题："""

INTENT_ROUTER_PROMPT = """你是劳动法问答系统的意图路由器，判断用户问题属于哪类，只输出一个词。

定义：
1. retrieve：法条查询或规则说明，只需检索法律条文（如"劳动合同法第44条"、"什么是经济补偿"、"加班工资怎么算"、"经济补偿金的计算标准"）
2. calculate：用户给出了具体数字，需要精确计算金额（如"月薪8000加班10小时加班费多少"、"工作5年赔偿金多少"）。注意：如果只是问计算方法/规则而没有具体数字，应归为retrieve
3. compare：对比分析，需要检索并对比不同法条/概念（如"经济补偿金和赔偿金有什么区别"、"固定期限和无固定期限合同区别"）

关键区分：问"怎么算/标准/规定"=retrieve，给了具体数字要"算多少"=calculate

对话上下文：{conversation_context}
用户问题：{question}

只输出一个词：retrieve、calculate、compare，不要解释。"""

CALCULATOR_PROMPT = """{style_rules}

额外规则（计算场景）：
- 必须代入具体数值计算，给出最终金额（保留2位小数）
- 先给出计算结果，再解释公式和法条依据
- 如果计算结果与法条不符，以法条为准

对话上下文：{conversation_context}

参考资料：
{context}

计算结果：{calculation_result}

问题：{question}

回答："""

COMPARATOR_PROMPT = """{style_rules}

额外规则（对比场景）：
- 用通俗语言对比两个概念的区别，不要堆砌法条原文
- 先总结核心区别，再展开细节

第一组（概念A）：
{context_a}

第二组（概念B）：
{context_b}

请从以下维度对比：
1. 法律依据（分别引用哪部法哪条）
2. 适用条件
3. 计算方式/标准
4. 关键区别

问题：{question}

对比分析："""

# 快速生成用Prompt（与crag_graph保持一致）
RAG_GENERATE_PROMPT = """{style_rules}

对话上下文：{conversation_context}

参考资料：
{context}

问题：{question}

回答："""

VALIDATOR_PROMPT = """你是劳动法回答质量验证专家，检查回答是否合法合规。

检查维度：
1. 回答是否与参考资料存在明显矛盾（如说"赔偿金是N倍"但资料写的是"2倍"）
2. 计算公式是否符合法律规定
3. 法条引用是否明显编造（如编造完全不存在的条款号，如"第999条"）
4. 回答中引用的每条法条是否有对应的参考资料支持
注意：回答中引用的法条必须有参考资料支持。如果引用了参考资料中不存在的法条，标记为不忠实，需要移除该引用后重新生成。

参考资料：
{context}

AI回答：{answer}

如果回答合法合规，输出"pass"；如果存在问题，输出"fail:具体原因"。
只输出pass或fail:原因，不要其他内容。"""


# ─── 劳动法计算工具 ──────────────────────────────────────────

# ─── 常量 ──────────────────────────────────────────────────

MONTHLY_WORK_DAYS = 21.75    # 月计薪天数
DAILY_WORK_HOURS = 8         # 日标准工时
VALIDATE_CONTEXT_CHARS = 2000  # 验证时上下文截断字数（增大以覆盖更多法条内容）
VALIDATE_ANSWER_CHARS = 2500   # 验证时回答截断字数

LABOR_CALCULATORS = {
    "加班费": {
        "formula": "加班费 = 月薪 ÷ 21.75 ÷ 8 × 加班小时数 × 倍率",
        "rates": {"工作日延时": 1.5, "休息日加班": 2.0, "法定节假日": 3.0},
        "law_ref": "《劳动法》第44条",
    },
    "经济补偿金": {
        "formula": "经济补偿金 = 工作年限 × 月工资",
        "law_ref": "《劳动合同法》第47条",
        "note": "每满1年支付1个月工资，6个月以上不满1年按1年算，不满6个月支付半个月",
    },
    "赔偿金": {
        "formula": "赔偿金 = 经济补偿金 × 2",
        "law_ref": "《劳动合同法》第87条",
        "note": "用人单位违反本法规定解除或终止劳动合同",
    },
    "未签合同双倍工资": {
        "formula": "双倍工资差额 = 月工资 × 应签未签月数(最多11个月)",
        "law_ref": "《劳动合同法》第82条",
        "note": "超过1个月不满1年未签书面合同，每月支付2倍工资",
    },
    "年休假工资": {
        "formula": "年休假工资 = 日工资 × 300% × 未休天数",
        "law_ref": "《职工带薪年休假条例》第5条",
        "note": "日工资 = 月薪 ÷ 21.75",
    },
}


def _extract_number(text: str, pattern: str) -> float:
    """从文本中提取数字。"""
    match = re.search(pattern, text)
    if match:
        return float(match.group(1).replace(",", ""))
    return 0.0


def calculate_overtime_pay(monthly_salary: float, hours: float,
                           rate_type: str = "工作日延时") -> dict:
    """计算加班费。"""
    rate = LABOR_CALCULATORS["加班费"]["rates"].get(rate_type, 1.5)
    hourly_wage = monthly_salary / MONTHLY_WORK_DAYS / DAILY_WORK_HOURS
    overtime_pay = round(hourly_wage * hours * rate, 2)
    return {
        "type": "加班费",
        "formula": LABOR_CALCULATORS["加班费"]["formula"],
        "law_ref": LABOR_CALCULATORS["加班费"]["law_ref"],
        "monthly_salary": monthly_salary,
        "hourly_wage": round(hourly_wage, 2),
        "overtime_hours": hours,
        "rate_type": rate_type,
        "rate": rate,
        "result": overtime_pay,
    }


def calculate_severance_pay(monthly_salary: float, years: float) -> dict:
    """计算经济补偿金。"""
    months = math.floor(years)
    if years - months >= 0.5:
        months += 1
    elif years - months > 0:
        months += 0.5
    severance = round(monthly_salary * months, 2)
    return {
        "type": "经济补偿金",
        "formula": LABOR_CALCULATORS["经济补偿金"]["formula"],
        "law_ref": LABOR_CALCULATORS["经济补偿金"]["law_ref"],
        "monthly_salary": monthly_salary,
        "working_years": years,
        "compensation_months": months,
        "result": severance,
    }


def calculate_damage_pay(monthly_salary: float, years: float) -> dict:
    """计算赔偿金（违法解除）。"""
    severance = calculate_severance_pay(monthly_salary, years)
    damage = round(severance["result"] * 2, 2)
    return {
        "type": "赔偿金",
        "formula": LABOR_CALCULATORS["赔偿金"]["formula"],
        "law_ref": LABOR_CALCULATORS["赔偿金"]["law_ref"],
        "monthly_salary": monthly_salary,
        "working_years": years,
        "severance_base": severance["result"],
        "result": damage,
    }


def _first_match(patterns: list, text: str) -> float:
    """按顺序尝试多个正则，返回第一个匹配的数值。"""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1).replace(",", ""))
    return 0.0


def _parse_cn_salary(text: str) -> float:
    """解析中文薪资表达，如"1万5"→15000。"""
    m = re.search(r"(\d+)万(\d*[千百]?)", text)
    if not m:
        return 0.0
    wan = float(m.group(1))
    rest_str = m.group(2)
    if not rest_str:
        return wan * 10000
    if "千" in rest_str:
        return wan * 10000 + float(rest_str.replace("千", "")) * 1000
    if "百" in rest_str:
        return wan * 10000 + float(rest_str.replace("百", "")) * 100
    return wan * 10000 + float(rest_str) * 1000


def auto_calculate(question: str, conversation_context: str = "") -> str:
    """根据问题自动识别计算类型并执行计算。

    尝试从问题中提取月薪、年限、小时数等参数。
    当追问缺少关键词时，从对话上下文继承计算类型和缺失参数。
    """
    combined = f"{conversation_context}\n{question}" if conversation_context else question

    # 提取月薪：优先当前问题，其次上下文，最后中文表达
    salary = _first_match(
        [r"月薪[为约]?(\d+[,.]?\d*)", r"工资[为约]?(\d+[,.]?\d*)", r"(\d+[,.]?\d*)元[/月]"],
        question,
    ) or _first_match([r"月薪[为约]?(\d+[,.]?\d*)"], combined) or _parse_cn_salary(question)

    # 提取年限：优先当前问题，其次上下文
    years = _first_match(
        [r"工作[了为约]?(\d+\.?\d*)年", r"(\d+\.?\d*)年"],
        question,
    ) or (conversation_context and _first_match([r"工作[了为约]?(\d+\.?\d*)年"], combined))

    # 提取加班小时：优先当前问题，其次上下文
    hours = _first_match(
        [r"加班[了为约]?(\d+\.?\d*)小时", r"(\d+\.?\d*)小时"],
        question,
    ) or (conversation_context and _first_match([r"加班[了为约]?(\d+\.?\d*)小时"], combined))

    # 判断加班类型
    rate_type = "工作日延时"
    if "休息日" in combined or "周末" in combined:
        rate_type = "休息日加班"
    elif "法定节假日" in combined or "节假日" in combined or "国庆" in combined or "春节" in combined:
        rate_type = "法定节假日"

    # 从上下文继承计算类型关键词
    has_overtime = "加班" in combined or "延时" in combined
    has_severance = "经济补偿" in combined or "补偿金" in combined
    has_damage = "赔偿金" in combined or "违法解除" in combined or "违法辞退" in combined

    results = []
    missing = []

    # 加班费计算
    if has_overtime:
        if salary > 0 and hours > 0:
            r = calculate_overtime_pay(salary, hours, rate_type)
            results.append(
                f"【{r['type']}】\n"
                f"  法条依据：{r['law_ref']}\n"
                f"  计算公式：{r['formula']}\n"
                f"  月薪：{r['monthly_salary']}元 → 时薪：{r['hourly_wage']}元\n"
                f"  加班{r['overtime_hours']}小时 × {r['rate']}倍({r['rate_type']})\n"
                f"  ➜ 加班费 = {r['result']}元"
            )
        else:
            if salary == 0: missing.append("月薪")
            if hours == 0: missing.append("加班小时数")
            results.append(
                f"【加班费】\n"
                f"  法条依据：{LABOR_CALCULATORS['加班费']['law_ref']}\n"
                f"  计算公式：{LABOR_CALCULATORS['加班费']['formula']}\n"
                f"  ⚠️ 缺少参数：{', '.join(missing)}，无法计算具体金额"
            )
            missing.clear()

    # 经济补偿金
    if has_severance:
        if salary > 0 and years > 0:
            r = calculate_severance_pay(salary, years)
            results.append(
                f"【{r['type']}】\n"
                f"  法条依据：{r['law_ref']}\n"
                f"  计算公式：{r['formula']}\n"
                f"  月薪：{r['monthly_salary']}元 × {r['compensation_months']}个月\n"
                f"  ➜ 经济补偿金 = {r['result']}元"
            )
        else:
            if salary == 0: missing.append("月薪")
            if years == 0: missing.append("工作年限")
            results.append(
                f"【经济补偿金】\n"
                f"  法条依据：{LABOR_CALCULATORS['经济补偿金']['law_ref']}\n"
                f"  计算公式：{LABOR_CALCULATORS['经济补偿金']['formula']}\n"
                f"  ⚠️ 缺少参数：{', '.join(missing)}，无法计算具体金额"
            )
            missing.clear()

    # 赔偿金
    if has_damage:
        if salary > 0 and years > 0:
            r = calculate_damage_pay(salary, years)
            results.append(
                f"【{r['type']}】\n"
                f"  法条依据：{r['law_ref']}\n"
                f"  计算公式：{r['formula']}\n"
                f"  经济补偿金基数：{r['severance_base']}元 × 2倍\n"
                f"  ➜ 赔偿金 = {r['result']}元"
            )
        else:
            if salary == 0: missing.append("月薪")
            if years == 0: missing.append("工作年限")
            results.append(
                f"【赔偿金】\n"
                f"  法条依据：{LABOR_CALCULATORS['赔偿金']['law_ref']}\n"
                f"  计算公式：{LABOR_CALCULATORS['赔偿金']['formula']}\n"
                f"  ⚠️ 缺少参数：{', '.join(missing)}，无法计算具体金额"
            )
            missing.clear()

    if not results:
        return "未能识别计算类型或缺少必要参数（月薪/年限/小时数）。"

    return "\n\n".join(results)


# ─── Agentic RAG工作流 ──────────────────────────────────────

class AgenticRAGGraph:
    """V4: LangGraph Agentic RAG 多Agent协作。"""

    def __init__(self, retriever):
        self.retriever = retriever
        self.compressor = EmbeddingsCompressor()

        # 分类/验证用LLM（使用轻量模型，速度更快）
        self.grader_llm = ChatOpenAI(
            model=settings.EVAL_LLM_MODEL_NAME,
            openai_api_key=settings.EVAL_LLM_API_KEY or settings.LLM_API_KEY,
            openai_api_base=settings.EVAL_LLM_API_BASE or settings.LLM_API_BASE,
            temperature=0,
            max_tokens=128,
            request_timeout=60,
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

    # ─── 节点函数 ──────────────────────────────────────────

    def _classify_complexity(self, state: AgentState) -> dict:
        """复杂度分类：simple→快速生成，medium/complex→Agent完整链路。"""
        question = state.get("rewritten_question") or state["question"]
        conversation_context = state.get("conversation_context", "")

        # 包含"第X条"的问题直接走快速生成，无需LLM分类
        import re
        if re.search(r"第[一二三四五六七八九十百千\d]+条", question):
            steps = state.get("steps", [])
            steps.append(f"复杂度分类: medium（含法条编号，快速检索）")
            return {"complexity": "medium", "steps": steps}

        # 其他问题用LLM判断
        print(f"[Agent] 🎯 分类查询复杂度...")
        prompt = ChatPromptTemplate.from_template(COMPLEXITY_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
        })
        complexity = result.content.strip().lower()
        if complexity not in ("simple", "medium", "complex"):
            complexity = "medium"  # 默认走中等路径
        print(f"[Agent] 🎯 复杂度: {complexity}")
        steps = state.get("steps", [])
        steps.append(f"复杂度分类: {complexity}")
        return {"complexity": complexity, "steps": steps}

    def _simple_generate(self, state: AgentState) -> dict:
        """快速路径：简单问题检索+充分性验证+生成，不经过Agent路由。"""
        question = state.get("rewritten_question") or state["question"]
        conversation_context = state.get("conversation_context", "")

        # 检索（自适应rerank_top_k：简单3条，复杂5条）
        rerank_top_k = 3 if state.get("complexity") == "simple" else 5
        from app.core.retriever.reranked import RerankedRetriever
        if isinstance(self.retriever, RerankedRetriever):
            docs = self.retriever.retrieve(
                query=question, k=settings.TOP_K,
                score_threshold=settings.SCORE_THRESHOLD,
                rerank_top_k=rerank_top_k,
            )
        else:
            docs = self.retriever.retrieve(
                query=question, k=settings.TOP_K,
                score_threshold=settings.SCORE_THRESHOLD,
            )
        context = "\n\n".join(
            f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc, score in docs
        ) if docs else "未找到相关参考资料。"

        steps = state.get("steps", [])

        # 两步生成：先验证检索充分性，不充分则拒绝回答（可通过RETRIEVAL_VALIDATION关闭）
        if settings.RETRIEVAL_VALIDATION:
            if docs:
                is_sufficient = self._check_retrieval_sufficiency(question, context)
                if not is_sufficient:
                    steps.append("检索验证: 资料不足，拒绝回答")
                    answer = "根据现有资料无法完整回答您的问题。"
                    return {"answer": answer, "context_docs": docs, "steps": steps}
                steps.append("检索验证: 资料充分")
            else:
                steps.append("检索验证: 无检索结果")
                answer = "未找到与您问题相关的法律资料。"
                return {"answer": answer, "context_docs": docs, "steps": steps}

        # 生成
        prompt = ChatPromptTemplate.from_template(RAG_GENERATE_PROMPT)
        chain = prompt | self.gen_llm
        result = chain.invoke({
            "context": context,
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
            "style_rules": ANSWER_STYLE_RULES,
        })

        answer = result.content
        steps.append(f"快速生成: 基于{len(docs)}个文档直接回答")

        # 轻量验证：用flash模型检查回答是否引用了不存在的法条
        if answer and docs:
            validate_state = {"answer": answer, "context_docs": docs, "steps": steps}
            validated = self._validate(validate_state)
            answer = validated.get("answer", answer)
            steps.extend(validated.get("steps", []))

        return {"answer": answer, "context_docs": docs, "steps": steps}

    def _decide_after_classify(self, state: AgentState) -> Literal["simple_path", "agent_path"]:
        """分类后路由：simple/medium→快速生成，complex→完整Agent链路。"""
        complexity = state.get("complexity", "medium")
        if complexity == "complex":
            return "agent_path"
        return "simple_path"

    def _rewrite_query(self, state: AgentState) -> dict:
        """上下文感知查询改写：将追问补全为完整独立问题。"""
        question = state["question"]
        conversation_context = state.get("conversation_context", "")

        # 无上下文或问题已经足够长，跳过改写
        if not conversation_context or len(question) > 30:
            steps = state.get("steps", [])
            steps.append(f"查询改写: 跳过（问题已完整）")
            return {"rewritten_question": question, "steps": steps}

        print(f"[Agent] ✏️ 上下文感知查询改写...")
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({
            "question": question,
            "conversation_context": conversation_context,
        })

        rewritten = result.content.strip()
        if not rewritten or rewritten == question:
            rewritten = question
            step_msg = "查询改写: 无需改写"
        else:
            step_msg = f"查询改写: '{question[:30]}...' → '{rewritten[:30]}...'"
            print(f"[Agent] ✏️ 改写结果: {question} → {rewritten}")

        steps = state.get("steps", [])
        steps.append(step_msg)
        return {"rewritten_question": rewritten, "steps": steps}

    def _route_intent(self, state: AgentState) -> dict:
        """Router Agent：意图识别，决定分派给哪个专业Agent。"""
        print(f"[Agent] 🧭 路由意图识别...")
        question = state.get("rewritten_question") or state["question"]
        conversation_context = state.get("conversation_context", "")

        prompt = ChatPromptTemplate.from_template(INTENT_ROUTER_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({
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

    def _retrieve_docs(self, state: AgentState) -> dict:
        """Retriever Agent：检索法条文档（三种路径共用）。

        复杂题自动多查询扩展：拆分为2-3个子查询分别检索，合并去重。
        """
        print(f"[Agent] 🔍 检索法条...")
        question = state.get("rewritten_question") or state["question"]
        complexity = state.get("complexity", "medium")

        # 自适应rerank_top_k：简单3条，中等5条，复杂8条
        rerank_top_k = 3 if complexity == "simple" else (5 if complexity == "medium" else 8)

        # ── 多查询扩展：复杂题拆分子查询分别检索 ──
        sub_queries = [question]
        if complexity == "complex":
            sub_queries = self._expand_queries(question)

        from app.core.retriever.reranked import RerankedRetriever
        all_docs = []
        seen_doc_ids = set()

        for q in sub_queries:
            if isinstance(self.retriever, RerankedRetriever):
                docs = self.retriever.retrieve(
                    query=q,
                    k=settings.TOP_K,
                    score_threshold=settings.SCORE_THRESHOLD,
                    rerank_top_k=rerank_top_k,
                )
            else:
                docs = self.retriever.retrieve(
                    query=q,
                    k=settings.TOP_K,
                    score_threshold=settings.SCORE_THRESHOLD,
                )
            # 去重合并
            for doc, score in docs:
                doc_id = doc.metadata.get("doc_id", doc.page_content[:80])
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    all_docs.append((doc, score))

        # 按分数降序排列
        all_docs.sort(key=lambda x: x[1], reverse=True)

        # 上下文压缩
        if all_docs:
            all_docs = self.compressor.compress(question, all_docs)
            if not all_docs:
                # 压缩失败，回退到原始检索
                if isinstance(self.retriever, RerankedRetriever):
                    all_docs = self.retriever.retrieve(
                        query=question, k=settings.TOP_K,
                        score_threshold=settings.SCORE_THRESHOLD,
                        rerank_top_k=rerank_top_k,
                    )
                else:
                    all_docs = self.retriever.retrieve(
                        query=question, k=settings.TOP_K,
                        score_threshold=settings.SCORE_THRESHOLD,
                    )

        steps = state.get("steps", [])
        expand_info = f" (多查询扩展: {len(sub_queries)}个子查询)" if len(sub_queries) > 1 else ""
        steps.append(f"检索: 获取{len(all_docs)}个文档{expand_info}")
        return {"context_docs": all_docs, "steps": steps}

    def _retrieve_for_compare(self, state: AgentState) -> dict:
        """Comparator检索：分别检索两组法条。

        从问题中提取两个对比概念，分别检索。
        """
        print(f"[Agent] 🔍 对比检索...")
        question = state.get("rewritten_question") or state["question"]

        # 先检索第一组（用完整问题）
        docs_a = self.retriever.retrieve(
            query=question,
            k=settings.TOP_K,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        if docs_a:
            docs_a = self.compressor.compress(question, docs_a) or docs_a

        # 用LLM提取第二个对比概念
        extract_prompt = ChatPromptTemplate.from_template(
            "从以下对比问题中提取两个对比概念，用'和'或'与'分隔输出。\n"
            "例如'经济补偿金和赔偿金有什么区别'→'经济补偿金和赔偿金'\n\n"
            "问题：{question}\n\n两个概念："
        )
        chain = extract_prompt | self.grader_llm
        concepts = chain.invoke({"question": question}).content.strip()

        # 分割概念
        parts = re.split(r"[和与及、]", concepts, maxsplit=1)
        if len(parts) >= 2:
            concept_b = parts[1].strip()
            docs_b = self.retriever.retrieve(
                query=concept_b,
                k=settings.TOP_K,
                score_threshold=settings.SCORE_THRESHOLD,
            )
            if docs_b:
                docs_b = self.compressor.compress(concept_b, docs_b) or docs_b
        else:
            docs_b = []

        steps = state.get("steps", [])
        steps.append(f"对比检索: A组{len(docs_a)}个, B组{len(docs_b)}个文档")
        return {"context_docs": docs_a, "context_docs_b": docs_b, "steps": steps}

    def _calculate(self, state: AgentState) -> dict:
        """Calculator Agent：检索法条 + 精确计算。"""
        print(f"[Agent] 🧮 计算Agent执行...")
        question = state.get("rewritten_question") or state["question"]
        conversation_context = state.get("conversation_context", "")
        docs = state.get("context_docs", [])

        # 执行自动计算（传入上下文以继承缺失参数）
        calc_result = auto_calculate(question, conversation_context)
        print(f"[Agent] 🧮 计算结果: {calc_result[:100]}...")

        # 拼接法条上下文
        context = self._format_docs(docs)

        # LLM整合法条+计算结果
        prompt = ChatPromptTemplate.from_template(CALCULATOR_PROMPT)
        chain = prompt | self.gen_llm
        result = chain.invoke({
            "context": context,
            "calculation_result": calc_result,
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
            "style_rules": ANSWER_STYLE_RULES,
        })

        steps = state.get("steps", [])
        steps.append(f"计算: {calc_result[:60]}...")
        return {"answer": result.content, "calculation_result": calc_result, "steps": steps}

    def _compare(self, state: AgentState) -> dict:
        """Comparator Agent：对比分析两组法条。"""
        print(f"[Agent] ⚖️ 对比Agent执行...")
        question = state.get("rewritten_question") or state["question"]
        docs_a = state.get("context_docs", [])
        docs_b = state.get("context_docs_b", [])

        context_a = self._format_docs(docs_a)
        context_b = self._format_docs(docs_b)

        prompt = ChatPromptTemplate.from_template(COMPARATOR_PROMPT)
        chain = prompt | self.gen_llm
        result = chain.invoke({
            "context_a": context_a,
            "context_b": context_b,
            "question": question,
            "style_rules": ANSWER_STYLE_RULES,
        })

        steps = state.get("steps", [])
        steps.append("对比: 已生成结构化对比分析")
        return {"answer": result.content, "steps": steps}

    def _generate_from_retrieval(self, state: AgentState) -> dict:
        """Retriever路径生成：基于检索文档直接生成回答。"""
        print(f"[Agent] 🤖 检索路径生成回答...")
        question = state.get("rewritten_question") or state["question"]
        docs = state.get("context_docs", [])
        conversation_context = state.get("conversation_context", "")

        context = self._format_docs(docs, fallback="未找到相关参考资料。")

        steps = state.get("steps", [])

        # 两步生成：先验证检索充分性（可通过RETRIEVAL_VALIDATION关闭）
        if settings.RETRIEVAL_VALIDATION and docs:
            is_sufficient = self._check_retrieval_sufficiency(question, context)
            if not is_sufficient:
                steps.append("检索验证: 资料不足，拒绝回答")
                answer = "根据现有资料无法完整回答您的问题。"
                return {"answer": answer, "steps": steps}
            steps.append("检索验证: 资料充分")

        prompt = ChatPromptTemplate.from_template(
            "{style_rules}\n\n"
            "对话上下文：{conversation_context}\n\n"
            "参考资料：\n{context}\n\n"
            "问题：{question}\n\n回答："
        )
        chain = prompt | self.gen_llm
        result = chain.invoke({
            "context": context,
            "question": question,
            "conversation_context": conversation_context or "（无对话上下文）",
            "style_rules": ANSWER_STYLE_RULES,
        })

        steps = state.get("steps", [])
        steps.append(f"生成: 基于{len(docs)}个文档生成回答")
        return {"answer": result.content, "steps": steps}

    def _validate(self, state: AgentState) -> dict:
        """Validator Agent：最终验证回答的合法性和准确性。"""
        print(f"[Agent] ✅ 验证Agent检查...")
        answer = state.get("answer", "")
        docs = state.get("context_docs", [])

        if not answer:
            steps = state.get("steps", [])
            steps.append("验证: 无回答可验证")
            return {"validation_passed": True, "steps": steps}

        context = "\n\n".join(
            doc.page_content[:VALIDATE_CONTEXT_CHARS] for doc, score in docs[:5]
        ) if docs else ""

        prompt = ChatPromptTemplate.from_template(VALIDATOR_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({"context": context, "answer": answer[:VALIDATE_ANSWER_CHARS]})

        verdict = result.content.strip().lower()
        print(f"[Agent] ✅ 验证原始输出: {verdict}")

        passed = not verdict.startswith("fail")
        steps = state.get("steps", [])
        if passed:
            steps.append("验证: 通过")
        else:
            reason = verdict.replace("fail:", "").strip()
            steps.append(f"验证: 未通过 - {reason}")

        return {"validation_passed": passed, "answer": answer, "steps": steps}

    def _self_correct(self, state: AgentState) -> dict:
        """自修正：验证未通过时，用更强约束的prompt重新生成。"""
        print(f"[Agent] 🔧 自修正重新生成...")
        question = state.get("rewritten_question") or state["question"]
        docs = state.get("context_docs", [])
        original_answer = state.get("answer", "")

        context = self._format_docs(docs, fallback="未找到相关参考资料。")

        correction_prompt = ChatPromptTemplate.from_template(
            "{style_rules}\n\n"
            "【重要】上一次生成的回答经验证存在不准确之处，请严格基于以下参考资料重新回答。\n"
            "规则：\n"
            "- 只使用参考资料中明确出现的法条和表述，禁止使用任何参考资料以外的知识\n"
            "- 每个论断必须标注来源法条，无来源的结论不得输出\n"
            "- 如果参考资料不足以回答，直接说明不足之处，不要编造\n\n"
            "参考资料：\n{context}\n\n"
            "问题：{question}\n\n回答："
        )
        chain = correction_prompt | self.gen_llm
        result = chain.invoke({
            "style_rules": ANSWER_STYLE_RULES,
            "context": context,
            "question": question,
        })

        steps = state.get("steps", [])
        steps.append("自修正: 基于更强约束重新生成")
        return {"answer": result.content, "validation_passed": True, "steps": steps}

    # ─── 辅助方法 ──────────────────────────────────────────

    def _expand_queries(self, question: str) -> List[str]:
        """多查询扩展：将复杂问题拆分为多个子查询。

        使用轻量LLM拆分，返回子查询列表（含原始问题）。
        """
        try:
            prompt = ChatPromptTemplate.from_template(MULTI_QUERY_EXPANSION_PROMPT)
            chain = prompt | self.grader_llm
            result = chain.invoke({"question": question})
            sub_queries = [line.strip() for line in result.content.strip().split("\n") if line.strip()]
            if not sub_queries:
                return [question]
            print(f"[Agent] 🔍 多查询扩展: '{question[:30]}...' → {sub_queries}")
            return sub_queries
        except Exception as e:
            print(f"[Agent] 🔍 多查询扩展失败: {e}，使用原始查询")
            return [question]

    def _check_retrieval_sufficiency(self, question: str, context: str) -> bool:
        """用轻量LLM判断检索结果是否足以回答问题。"""
        # 上下文太短，大概率不充分
        if len(context) < 50:
            return False
        prompt = ChatPromptTemplate.from_template(RETRIEVAL_EVALUATION_PROMPT)
        chain = prompt | self.grader_llm
        result = chain.invoke({"question": question, "context": context[:2000]})
        verdict = result.content.strip()
        return "充分" in verdict and "不足" not in verdict

    @staticmethod
    def _format_docs(docs: list, fallback: str = "未找到相关法条。") -> str:
        """将检索文档列表格式化为上下文字符串。"""
        if not docs:
            return fallback
        return "\n\n".join(
            f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc, score in docs
        )

    # ─── 条件边 ────────────────────────────────────────────

    def _decide_intent(self, state: AgentState) -> Literal[
        "retrieve_path", "calculate_path", "compare_path"
    ]:
        """意图路由后的分发。"""
        intent = state.get("intent", "retrieve")
        if intent == "calculate":
            return "calculate_path"
        elif intent == "compare":
            return "compare_path"
        return "retrieve_path"

    # ─── 构建Graph ─────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """构建Agentic RAG状态图。"""
        graph = StateGraph(AgentState)

        # ── 节点 ──
        graph.add_node("route_intent", self._route_intent)

        # Retriever路径
        graph.add_node("retrieve_docs", self._retrieve_docs)
        graph.add_node("generate_from_retrieval", self._generate_from_retrieval)

        # Calculator路径
        graph.add_node("retrieve_for_calc", self._retrieve_docs)
        graph.add_node("calculate", self._calculate)

        # Comparator路径
        graph.add_node("retrieve_for_compare", self._retrieve_for_compare)
        graph.add_node("compare", self._compare)

        # Validator
        graph.add_node("validate", self._validate)
        graph.add_node("self_correct", self._self_correct)

        # ── 复杂度分类（新增） ──
        graph.add_node("classify_complexity", self._classify_complexity)
        graph.add_node("simple_generate", self._simple_generate)

        # ── 入口：先改写追问，再判断复杂度 ──
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "classify_complexity")

        # ── 复杂度路由：simple/medium→快速生成，complex→Agent链路 ──
        graph.add_conditional_edges(
            "classify_complexity",
            self._decide_after_classify,
            {
                "simple_path": "simple_generate",
                "agent_path": "route_intent",
            },
        )
        graph.add_edge("simple_generate", END)

        # ── 意图路由（仅complex问题走到这里） ──
        graph.add_conditional_edges(
            "route_intent",
            self._decide_intent,
            {
                "retrieve_path": "retrieve_docs",
                "calculate_path": "retrieve_for_calc",
                "compare_path": "retrieve_for_compare",
            },
        )

        # ── Retriever路径 ──
        graph.add_edge("retrieve_docs", "generate_from_retrieval")
        graph.add_edge("generate_from_retrieval", "validate")

        # ── Calculator路径 ──
        graph.add_edge("retrieve_for_calc", "calculate")
        graph.add_edge("calculate", "validate")

        # ── Comparator路径 ──
        graph.add_edge("retrieve_for_compare", "compare")
        graph.add_edge("compare", "validate")

        # ── Validator → 通过则END，未通过则自修正 ──
        graph.add_conditional_edges(
            "validate",
            lambda state: "output" if state.get("validation_passed", True) else "self_correct",
            {
                "output": END,
                "self_correct": "self_correct",
            },
        )
        graph.add_edge("self_correct", END)

        return graph.compile()

    # ─── 对外接口 ──────────────────────────────────────────

    def run(self, question: str, conversation_context: str = "", skip_guardrails: bool = False) -> dict:
        """运行Agentic RAG工作流。"""
        initial_state: AgentState = {
            "question": question,
            "conversation_context": conversation_context,
            "steps": [],
        }
        result = self.graph.invoke(initial_state)
        answer = result.get("answer", "")
        # 评估模式下跳过护栏追加，避免RAGAS将追加内容判为幻觉
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
