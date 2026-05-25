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
import re
from collections import defaultdict
from typing import Dict, List


# ── 法条编号提取与标准化 ──

_ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+条")

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000,
}


def _cn_to_int(cn: str) -> int:
    """中文数字 → 整数（1-999）。"""
    if cn.isdigit():
        return int(cn)
    result = 0
    current = 0
    for ch in cn:
        if ch not in _CN_DIGITS:
            break
        val = _CN_DIGITS[ch]
        if val >= 10:
            if current == 0:
                current = 1
            result += current * val
            current = 0
        else:
            current = val
    result += current
    return result


def _normalize_article(article: str) -> str:
    """法条编号标准化为阿拉伯数字格式。

    "第四十四条" → "第44条"，"第47条" → "第47条"。
    """
    match = _ARTICLE_PATTERN.search(article)
    if not match:
        return article
    raw = match.group()
    num_part = raw[1:-1]  # 去掉"第"和"条"
    try:
        num = _cn_to_int(num_part)
        return f"第{num}条"
    except (ValueError, IndexError):
        return raw


def _extract_law_name(source: str) -> str:
    """从source字段提取法律名称。"劳动法.txt" → "劳动法" """
    return os.path.splitext(os.path.basename(source))[0]


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
            "precision@5": 0.60,
            "recall@5": 0.75,
            "f1@5": 0.67,
            "mrr": 0.85,
            "map": 0.78,
            "per_query": [...],
        }
    """
    n = len(sources_list)
    if n == 0:
        return {
            f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"f1@{k}": 0.0,
            "mrr": 0.0, "map": 0.0, "per_query": [],
        }

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
            f"precision@{k}": round(precision_at_k(retrieved_ids, normalized_relevant, k), 4),
            f"recall@{k}": round(recall_at_k(retrieved_ids, normalized_relevant, k), 4),
            f"f1@{k}": round(f1_at_k(retrieved_ids, normalized_relevant, k), 4),
            "mrr": round(mrr(retrieved_ids, normalized_relevant), 4),
            "ap": round(average_precision(retrieved_ids, normalized_relevant), 4),
            "retrieved_ids": retrieved_ids[:k],
            "relevant_ids": normalized_relevant,
        }
        per_query.append(pq)

    # 计算均值（ap在输出时重命名为map）
    metrics = {}
    for key in [f"precision@{k}", f"recall@{k}", f"f1@{k}", "mrr"]:
        metrics[key] = round(sum(pq[key] for pq in per_query) / len(per_query), 4)
    metrics["map"] = round(sum(pq["ap"] for pq in per_query) / len(per_query), 4)
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
