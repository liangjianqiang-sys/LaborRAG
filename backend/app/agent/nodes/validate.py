"""Validate — 代码校验节点（幻觉护栏）。

零 LLM 调用：纯正则 + 集合比对，检查回答里引用的法条是否真的出现在检索文档中。
"""
import re

from app.agent.state import AgentState
from app.utils.law_refs import ARTICLE_REF_RE
from app.utils.text import normalize_article


def _split_sentences(text: str) -> list[str]:
    """按句末标点切分，标点保留在句尾。

    只丢弃**空串**，保留**纯空白片段** —— 这一点必须小心：
    切分符含 `\\n`，所以 markdown 的段落空行（`\\n\\n`）会变成独立的 `"\\n"` 片段。
    若按 `s.strip()` 过滤，这些片段会被吃掉，`\\n\\n` 塌成 `\\n`，
    而 markdown 里**单换行会被渲染成空格** —— 于是 `**小标题**` 被并进上一段正文。

    实测（2026-09-17，真实链路）：护栏触发后答案里出现
    `两者不能同时主张。**对比分析：**` —— 小标题被并进上一句。
    纯空白片段本身不含法条引用，保留它们即可让结构原样穿过。
    """
    return [s for s in re.split(r"(?<=[。；\n])", text) if s]


def _drop_sentences_citing(answer: str, unsupported: set[str]) -> str:
    """删除引用了**未支撑法条**的整句。

    为什么按句删除，而不是用「从某处删到某处」的正则
    -------------------------------------------------
    原实现用的是：

        [，。；]?\\s*(?:根据|依照|依据|按照)?\\s*《?[^》]*》?\\s*第X条[^。；]*[。；]?

    其中 `《?` / `[^》]*` / `》?` 三段都过于宽松，贪婪的 `[^》]*` 会**从更早的位置
    开始吞**。实测（2026-09-17）：

        输入：根据《劳动合同法》第四十七条，经济补偿按年限算。
              根据《劳动合同法》第九十九条，应当加倍处罚。
        上下文只支撑第四十七条（第九十九条是幻觉）
        期望：只删第二句
        实际：'根据《劳动合同法》'      ← 第一句也被吞掉了

    即**幻觉护栏会误删大段正确内容** —— 这比它要防的幻觉更糟：用户在不知情的情况下
    丢掉正确信息，而且不会有任何报错。

    按句切分后逐句判断，行为可预测，且绝不会误删相邻句子。

    取舍：一句话同时引用了支撑与未支撑法条时，**整句丢弃**（保守）。
    理由是护栏的职责是不让无依据的法律主张流出去；保留半句反而会留下
    断章取义的断言。
    """
    kept = []
    for sentence in _split_sentences(answer):
        refs = {normalize_article(a) for a in ARTICLE_REF_RE.findall(sentence)}
        if refs & unsupported:
            continue  # 该句引用了未支撑法条 → 整句丢弃
        kept.append(sentence)
    return "".join(kept).strip()


def validate(state: AgentState) -> dict:
    """代码校验：检查回答中引用的法条是否在检索文档中存在。

    零LLM调用，纯正则+集合比对（中文/阿拉伯数字归一化后比对）：
    - 提取回答中所有"第X条"引用
    - 提取contexts中所有"第X条"
    - 归一化后比对，回答中引用了contexts中不存在的法条 → 删除**引用它的整句**
    """
    answer = state.get("answer", "")
    docs = state.get("context_docs", [])

    if not answer or not docs:
        steps = state.get("steps", [])
        steps.append("验证: 跳过（无回答或无文档）")
        # 注：validation_passed 目前只被赋值、全项目无人读取（图里 validate→END 无条件边）。
        # 保留字段以免破坏 state 契约，但**不要**据此认为"答案已通过校验"。
        return {"validation_passed": True, "steps": steps}

    # 提取回答中的法条引用（归一化为阿拉伯格式）
    answer_articles = {normalize_article(a) for a in ARTICLE_REF_RE.findall(answer)}

    # 提取 contexts 中的法条并归一化
    context_articles: set[str] = set()
    for doc, score in docs:
        context_articles.update(
            normalize_article(a) for a in ARTICLE_REF_RE.findall(doc.page_content)
        )

    unsupported = answer_articles - context_articles

    steps = state.get("steps", [])
    if not unsupported:
        steps.append("验证: 通过（法条引用均有文档支撑）")
        return {"validation_passed": True, "answer": answer, "steps": steps}

    clean_answer = _drop_sentences_citing(answer, unsupported)
    steps.append(f"验证: 移除未支撑法条 {', '.join(sorted(unsupported))}")
    return {"validation_passed": True, "answer": clean_answer, "steps": steps}
