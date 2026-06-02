"""Helpers — 纯函数工具集，不感知 AgentState，不依赖 LLM/Retriever。"""
import os
import re

from app.core.generator.constants import CN_DIGIT_MAP


def normalize_article(article: str) -> str:
    """将法条编号归一化为阿拉伯数字：第四十七条 → 第47条，第九十一条 → 第91条。"""
    core = article[1:-1]  # 去掉"第"和"条"
    if core.isdigit():
        return f"第{int(core)}条"
    # 中文数字转阿拉伯
    result = 0
    current = 0
    for ch in core:
        val = CN_DIGIT_MAP.get(ch)
        if val is None:
            return article  # 无法解析，返回原值
        if val >= 10:
            if current == 0:
                current = 1
            result += current * val
            current = 0
        else:
            current = val
    result += current
    return f"第{result}条" if result > 0 else article


def format_docs(docs: list, fallback: str = "未找到相关法条。") -> str:
    """将检索文档列表格式化为结构化上下文字符串。

    父子分块结构化格式：
    [背景法条]：《法律名》第X条（完整内容含N款）
    [核心命中]：精准检索到与本问题最相关的是该条的【第(一)款】
    [法条原文]：
    第四十七条 经济补偿按...（省略）
    （一）高薪人员封顶...（此处为命中核心）
    （二）月工资定义...（省略）
    """
    if not docs:
        return fallback

    parts = []
    for doc, score in docs:
        source = doc.metadata.get("source", "未知")
        article = doc.metadata.get("article", "")
        chunk_type = doc.metadata.get("chunk_type", "")
        parent_id = doc.metadata.get("parent_id", "")
        injected_by = doc.metadata.get("injected_by", "")

        # 非父子分块或无结构信息，走简单格式
        if not chunk_type or not article:
            parts.append(f"[来源：{source}]\n{doc.page_content}")
            continue

        # 父块：完整法条，标注来源和条款数
        if chunk_type == "parent":
            lines = [l.strip() for l in doc.page_content.split("\n") if l.strip()]
            clause_count = sum(1 for l in lines if _is_clause_line(l))
            header = f"[背景法条]：《{_law_display(source)}》{article}"
            if clause_count > 0:
                header += f"（完整内容含{clause_count}款）"
            if injected_by == "rule":
                header += " [知识图谱注入]"
            parts.append(f"{header}\n[法条原文]：\n{doc.page_content}")
            continue

        # 子块：标注核心命中段落
        if chunk_type == "child":
            # 尝试识别是第几款
            clause_label = _extract_clause_label(doc.page_content)
            header = f"[背景法条]：《{_law_display(source)}》{article}"
            hit_info = f"[核心命中]：精准检索到与本问题最相关的是该条的【{clause_label}】" if clause_label else "[核心命中]：精准检索到与本问题最相关的是该条内容"
            if injected_by == "rule":
                hit_info += " [知识图谱注入]"
            parts.append(f"{header}\n{hit_info}\n[法条原文]：\n{doc.page_content}")
            continue

        # 兜底
        parts.append(f"[来源：{source}]\n{doc.page_content}")

    return "\n\n".join(parts)


def _law_display(source: str) -> str:
    """将 source 文件名转为法律显示名。"""
    from app.core.retriever.law_graph import _SOURCE_LAW_MAP
    return _SOURCE_LAW_MAP.get(source, os.path.splitext(os.path.basename(source))[0])


def _is_clause_line(line: str) -> bool:
    """判断是否为款项行（如（一）、1. 等）。"""
    return bool(re.match(r"^[（(][一二三四五六七八九十\d]+[）)]|^\d+[.、]", line))


def _extract_clause_label(text: str) -> str:
    """从子块文本中提取款项标签，如'第(一)款'。"""
    first_line = text.split("\n")[0].strip()
    # 匹配（一）、（二）等
    m = re.match(r"^[（(]([一二三四五六七八九十\d]+)[）)]", first_line)
    if m:
        return f"第({m.group(1)})款"
    # 匹配 1.、2. 等
    m = re.match(r"^(\d+)[.、]", first_line)
    if m:
        return f"第{m.group(1)}项"
    return ""


def strip_unsolicited_sections(answer: str) -> str:
    """移除LLM擅自追加的"注意事项"/"说明"等段落（Prompt约束不住时兜底）。"""
    answer = re.sub(r"\n*---\s*\n\*\*注意事项[：:]\*\*.*", "", answer, flags=re.DOTALL)
    answer = re.sub(r"\n*注意事项[：:][^\n]*\n(?:- .*\n)*", "", answer)
    answer = re.sub(r"\n*说明[：:][^\n]*\n(?:- .*\n)*", "", answer)
    return answer.strip()


def extract_compare_concepts(question: str) -> tuple:
    """从对比问题中规则提取两个对比概念（零LLM）。

    返回 (concept_a, concept_b)，提取失败返回 (question, question)。
    """
    # 去掉尾部疑问句式
    core = re.sub(
        r"(有什么区别|有什么不同|的区别|的不同|的对比|对比分析|"
        r"区别是什么|不同是什么|区别在哪|不同在哪|有什么差别|的差别)[？?]?$",
        "", question,
    ).strip()
    # 按"和/与/及/对比"分割
    parts = re.split(r"[和与及]|[对比]{2}", core, maxsplit=1)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2:
        # 尝试更宽松的分割：按"、"分割
        parts = re.split(r"[、]", core, maxsplit=1)
        parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return parts[0], parts[1]
    return question, question
