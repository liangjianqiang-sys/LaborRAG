"""Validate — 代码校验节点。"""
import re

from app.core.generator.state import AgentState
from app.core.generator.helpers import normalize_article


def validate(state: AgentState) -> dict:
    """代码校验：检查回答中引用的法条是否在检索文档中存在。

    零LLM调用，纯正则+集合比对（中文/阿拉伯数字归一化后比对）：
    - 提取回答中所有"第X条"引用
    - 提取contexts中所有"第X条"
    - 归一化后比对，回答中引用了contexts中不存在的法条 → 移除
    """
    answer = state.get("answer", "")
    docs = state.get("context_docs", [])

    if not answer or not docs:
        steps = state.get("steps", [])
        steps.append("验证: 跳过（无回答或无文档）")
        return {"validation_passed": True, "steps": steps}

    # 提取回答中的法条引用
    answer_articles_raw = set(re.findall(r"第[一二三四五六七八九十百千零〇\d]+条", answer))
    answer_articles = {normalize_article(a) for a in answer_articles_raw}

    # 提取contexts中的法条并归一化
    context_articles = set()
    for doc, score in docs:
        raw = re.findall(r"第[一二三四五六七八九十百千零〇\d]+条", doc.page_content)
        context_articles.update(normalize_article(a) for a in raw)

    # 检查回答中是否有contexts中不存在的法条引用
    hallucinated_articles = answer_articles - context_articles

    steps = state.get("steps", [])
    if hallucinated_articles:
        hallucinated_raw = set()
        for a_raw in answer_articles_raw:
            if normalize_article(a_raw) in hallucinated_articles:
                hallucinated_raw.add(a_raw)
        # 移除幻觉法条引用
        clean_answer = answer
        for art in hallucinated_raw:
            clean_answer = re.sub(
                r"[，。；]?\s*(?:根据|依照|依据|按照)?\s*《?[^》]*》?\s*"
                + re.escape(art) + r"[^。；]*[。；]?",
                "", clean_answer,
            )
        steps.append(f"验证: 移除未支撑法条 {', '.join(hallucinated_articles)}")
        return {"validation_passed": True, "answer": clean_answer.strip(), "steps": steps}

    steps.append("验证: 通过（法条引用均有文档支撑）")
    return {"validation_passed": True, "answer": answer, "steps": steps}
