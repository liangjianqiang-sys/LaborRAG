"""检索评估指标模块：计算检索质量的量化指标。

指标：
  - Precision@k: 前k个结果中相关文档比例
  - Recall@k:    相关文档被检索到的比例
  - F1@k:        P和R的调和平均
  - MRR:         第一个相关文档排名倒数
  - MAP:         所有位置的平均精度均值

匹配逻辑：
  从检索返回的 sources 中提取法条标识（source文件名 + content中"第X条"），
  与 eval_dataset 的 relevant_articles 标注匹配。
"""

import os
from collections import defaultdict
from typing import Dict, List

from app.utils.law_refs import ARTICLE_REF_RE as _ARTICLE_PATTERN
from app.utils.law_refs import cn_to_int


# ── 法条编号提取与标准化 ──

# 正则与中文数字解析的唯一真相源都在 app.utils.law_refs。
# 曾经本模块自带一份解析表，且不含「〇」—— 于是「第一百〇一条」被静默算成
# 第 100 条，检索命中也被判未命中。这类失真不抛异常，只让指标悄悄变差。


def _normalize_article(article: str) -> str:
    """法条编号标准化为阿拉伯数字格式，保留法律名前缀。

    "劳动法第四十四条" → "劳动法第44条"，"第47条" → "第47条"。
    """
    # 提取法律名前缀（长名优先匹配，避免"中华人民共和国劳动合同法"被"劳动法"截断）
    law_prefix = ""
    for law in (
        # 全称（优先匹配）
        "中华人民共和国劳动合同法实施条例",
        "中华人民共和国劳动争议调解仲裁法",
        "中华人民共和国社会保险法",
        "工伤保险条例", "职工带薪年休假条例",
        "女职工劳动保护特别规定", "最低工资规定", "工资支付暂行规定", "失业保险条例",
        # 简称
        "劳动合同法实施条例", "劳动争议调解仲裁法", "社会保险法",
        "劳动合同法", "劳动法",
    ):
        if article.startswith(law):
            law_prefix = law
            break

    match = _ARTICLE_PATTERN.search(article)
    if not match:
        return article
    raw = match.group()
    num = cn_to_int(raw[1:-1])  # 去掉"第"和"条"
    if num is None:
        # 解析不了就原样保留，不猜测条号
        return f"{law_prefix}{raw}"
    return f"{law_prefix}第{num}条"


_LAW_NAME_SHORT = {
    "中华人民共和国劳动合同法": "劳动合同法",
    "中华人民共和国劳动合同法实施条例": "劳动合同法实施条例",
    "中华人民共和国劳动法": "劳动法",
    "中华人民共和国劳动争议调解仲裁法": "劳动争议调解仲裁法",
    "中华人民共和国社会保险法": "社会保险法",
}


def _extract_law_name(source: str) -> str:
    """从source字段提取法律名称（简称）。"中华人民共和国劳动合同法.pdf" → "劳动合同法" """
    raw = os.path.splitext(os.path.basename(source))[0]
    return _LAW_NAME_SHORT.get(raw, raw)


def extract_article_id(source: str, content: str) -> str:
    """从检索结果提取法条标识（取第一个法条编号作为主标识）。

    法条切分模式下，每个chunk的第一行就是该法条编号，
    后续出现的"第X条"是引用其他法条，不作为主标识。

    Returns: "劳动法第44条" or ""
    """
    law_name = _extract_law_name(source)
    match = _ARTICLE_PATTERN.search(content)
    if not match:
        return ""
    article = _normalize_article(match.group())
    return f"{law_name}{article}"


# ── 单指标计算 ──

def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """Precision@k: 前k个结果中相关文档比例。"""
    if k <= 0 or not retrieved:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for aid in retrieved[:k] if aid in rel_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """Recall@k: 相关文档被检索到的比例。"""
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    hits = sum(1 for aid in retrieved[:k] if aid in rel_set)
    return hits / len(rel_set)


def f1_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """F1@k: P@k和R@k的调和平均。"""
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """MRR: 第一个相关文档排名倒数。"""
    if not retrieved or not relevant:
        return 0.0
    rel_set = set(relevant)
    for i, aid in enumerate(retrieved, 1):
        if aid in rel_set:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """AP: 单个query的平均精度，用于计算MAP。"""
    if not retrieved or not relevant:
        return 0.0
    rel_set = set(relevant)
    hits = 0
    sum_precision = 0.0
    for i, aid in enumerate(retrieved, 1):
        if aid in rel_set:
            hits += 1
            sum_precision += hits / i
    return sum_precision / len(rel_set) if hits > 0 else 0.0


# ── 主入口 ──

def _METRIC_KEYS(k: int = 5) -> list:
    """聚合结果的键集合 —— 非空与空输入两条路径共用，保证契约一致。"""
    return ["precision@1", "precision@3", "precision@5", f"recall@{k}", "f1@5", "mrr", "map"]


def compute_retrieval_metrics(
    sources_list: List[List[Dict]],
    relevant_articles_list: List[List[str]],
    k: int = 5,
) -> Dict:
    """计算检索评估指标。

    Args:
        sources_list: 每个query的检索结果，每个source是
                      {"source": "劳动法.txt", "content": "第四十四条..."}
        relevant_articles_list: 每个query的相关法条，如 ["劳动法第44条"]
        k: 截断位置，默认5

    Returns:
        {
            "precision@1": 0.90,
            "precision@3": 0.63,
            "precision@5": 0.40,
            "recall@5": 0.92,
            "f1@5": 0.56,
            "mrr": 0.94,
            "map": 0.80,
            "per_query": [...],
        }

    注：P@3 / P@5 / F1 / MRR / MAP 是后来补齐的 —— 此前 docstring 宣称返回它们，
    实现却只产出 P@1 与 R@k，导致 README 宣称的「P@1/P@3/P@5/R@5/MRR」评估覆盖
    在官方链路上并不存在（答辩时若被要求看 MRR 会拿不出来）。
    消费方（runner / persistent 把 per_query 的键拷进每题明细、_summary 遍历数值键）
    都是加性的，故补齐不影响既有输出。
    """
    n = len(sources_list)
    if n == 0:
        # 契约对齐：空输入必须返回**与非空路径完全相同**的键集合，
        # 否则下游读 mrr / map 会 KeyError（原实现只返回 P@1 与 R@k，
        # 补齐指标后这里也必须同步补齐）。
        return {key: 0.0 for key in _METRIC_KEYS(k)} | {"per_query": []}

    per_query = []
    for sources, relevant in zip(sources_list, relevant_articles_list):
        # 从每个source提取法条ID（按检索顺序）
        retrieved_ids = []
        for src in sources:
            aid = extract_article_id(src.get("source", ""), src.get("content", ""))
            if aid:
                retrieved_ids.append(aid)

        # 标准化relevant_articles（确保阿拉伯数字格式）
        normalized_relevant = [_normalize_article(a) for a in relevant]

        pq = {
            "precision@1": round(precision_at_k(retrieved_ids, normalized_relevant, 1), 4),
            "precision@3": round(precision_at_k(retrieved_ids, normalized_relevant, 3), 4),
            "precision@5": round(precision_at_k(retrieved_ids, normalized_relevant, 5), 4),
            f"recall@{k}": round(recall_at_k(retrieved_ids, normalized_relevant, k), 4),
            "f1@5": round(f1_at_k(retrieved_ids, normalized_relevant, k), 4),
            "mrr": round(mrr(retrieved_ids, normalized_relevant), 4),
            "map": round(average_precision(retrieved_ids, normalized_relevant), 4),
            "retrieved_ids": retrieved_ids[:k],
            "relevant_ids": normalized_relevant,
        }
        per_query.append(pq)

    # 计算均值
    metrics = {
        key: round(sum(pq[key] for pq in per_query) / len(per_query), 4)
        for key in _METRIC_KEYS(k)
    }
    metrics["per_query"] = per_query
    return metrics


def compute_retrieval_metrics_by_type(
    sources_list: List[List[Dict]],
    relevant_articles_list: List[List[str]],
    question_types: List[str],
    k: int = 5,
) -> Dict[str, Dict]:
    """按问题类型分组计算检索指标。"""
    groups: Dict[str, Dict] = defaultdict(lambda: {"sources": [], "relevant": []})
    for i, qtype in enumerate(question_types):
        groups[qtype]["sources"].append(sources_list[i])
        groups[qtype]["relevant"].append(relevant_articles_list[i])

    return {
        qtype: compute_retrieval_metrics(g["sources"], g["relevant"], k=k)
        for qtype, g in groups.items()
    }
