"""Classify — 复杂度分类节点 + 分类后路由。"""
from typing import Literal
import re

from app.core.generator.state import AgentState
from app.core.generator.constants import COMPLEX_KEYWORDS, CALCULATE_KEYWORDS, SIMPLE_KEYWORDS


def classify_complexity(state: AgentState) -> dict:
    """规则化复杂度分类：simple→快速生成，complex→Agent完整链路。

    纯规则判断，零LLM调用：
    - 含对比关键词 → complex
    - 含计算关键词 → complex（走calculate路径）
    - 含法条编号 → medium（快速检索）
    - 含简单关键词或短问题 → simple
    - 默认 → medium
    """
    question = state.get("rewritten_question") or state["question"]

    if any(kw in question for kw in COMPLEX_KEYWORDS):
        steps = state.get("steps", [])
        steps.append("复杂度分类: complex（含对比关键词）")
        return {"complexity": "complex", "steps": steps}

    if any(kw in question for kw in CALCULATE_KEYWORDS):
        steps = state.get("steps", [])
        steps.append("复杂度分类: complex（含计算关键词）")
        return {"complexity": "complex", "steps": steps}

    if re.search(r"第[一二三四五六七八九十百千\d]+条", question):
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
