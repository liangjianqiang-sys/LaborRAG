"""Classify — 复杂度分类节点 + 分类后路由。"""
from typing import Literal

from app.agent.state import AgentState
from app.agent.constants import COMPLEX_KEYWORDS, CALCULATE_KEYWORDS, SIMPLE_KEYWORDS
from app.utils.law_refs import ARTICLE_REF_RE


def classify_complexity(state: AgentState) -> dict:
    """规则化复杂度分类：simple→快速生成，complex→Agent完整链路。

    纯规则判断，零LLM调用：
    - 含对比关键词 → complex
    - 含计算关键词 → complex（走calculate路径）
    - 含法条编号 → medium（快速检索）
    - 含简单关键词或短问题 → simple
    - 默认 → medium
    """
    # 原始问题与改写结果**都要看**，取信号并集。
    #
    # 只看改写结果会丢信号：改写节点会把「有什么区别」这类**关系性措辞**滤掉、
    # 只保留法言法语术语。实测「经济补偿金和赔偿金有什么区别？」被改写成
    # 「经济补偿金 赔偿金」——「区别」没了，于是本应走对比路径（带禁止合成/
    # 禁止泛化约束）的问题被判成 simple，退化成普通检索问答。
    original = state["question"]
    rewritten = state.get("rewritten_question") or original
    question = rewritten if original == rewritten else f"{original} {rewritten}"

    if any(kw in question for kw in COMPLEX_KEYWORDS):
        steps = state.get("steps", [])
        steps.append("复杂度分类: complex（含对比关键词）")
        return {"complexity": "complex", "steps": steps}

    if any(kw in question for kw in CALCULATE_KEYWORDS):
        steps = state.get("steps", [])
        steps.append("复杂度分类: complex（含计算关键词）")
        return {"complexity": "complex", "steps": steps}

    if ARTICLE_REF_RE.search(question):
        steps = state.get("steps", [])
        steps.append("复杂度分类: medium（含法条编号）")
        return {"complexity": "medium", "steps": steps}

    if any(kw in question for kw in SIMPLE_KEYWORDS) or len(question) <= 10:
        steps = state.get("steps", [])
        steps.append("复杂度分类: simple")
        return {"complexity": "simple", "steps": steps}

    steps = state.get("steps", [])
    steps.append("复杂度分类: medium（默认）")
    return {"complexity": "medium", "steps": steps}


def decide_after_classify(state: AgentState) -> Literal["simple_path", "agent_path"]:
    """分类后路由：simple/medium→快速生成，complex→完整Agent链路。"""
    complexity = state.get("complexity", "medium")
    if complexity == "complex":
        return "agent_path"
    return "simple_path"
