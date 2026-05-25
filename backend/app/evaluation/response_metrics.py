"""响应评估指标模块：计算生成回答质量的量化指标。

指标：
  - ROUGE-L:        最长公共子序列，衡量答案与ground_truth的词汇重叠
  - BLEU:           n-gram精度，衡量生成质量
  - 幻觉率:         回答中无法被context支撑的claim比例（复用RAGAS faithfulness）
  - 完整性:         LLM判断ground_truth的关键信息点是否被回答覆盖

实现方式：
  - ROUGE-L / BLEU：纯Python实现（rouge-score库），无需LLM调用，零成本
  - 幻觉率：1 - faithfulness（复用RAGAS结果）
  - 完整性：LLM评估，需调用LLM
"""

import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional


# ── 中文分词辅助 ──

def _char_tokenize(text: str) -> List[str]:
    """中文按字符分词（去除标点空白），用于ROUGE/BLEU计算。"""
    # 去除标点和空白
    text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
    return list(text)


# ── ROUGE-L ──

def _lcs_length(x: List[str], y: List[str]) -> int:
    """计算最长公共子序列长度。"""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # 空间优化：只保留两行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    """计算ROUGE-L F1分数。

    基于最长公共子序列(LCS)，衡量生成答案与参考答案的词汇重叠。
    对中文按字符粒度计算。
    """
    pred_tokens = _char_tokenize(prediction)
    ref_tokens = _char_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── BLEU ──

def _get_ngrams(tokens: List[str], n: int) -> Dict[str, int]:
    """获取n-gram计数。"""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def _clip_count(pred_ngrams: Dict[str, int], ref_ngrams: Dict[str, int]) -> int:
    """计算裁剪后的n-gram匹配数。"""
    count = 0
    for gram, cnt in pred_ngrams.items():
        count += min(cnt, ref_ngrams.get(gram, 0))
    return count


def _brevity_penalty(pred_len: int, ref_len: int) -> float:
    """BLEU的简短惩罚因子。"""
    if pred_len > ref_len:
        return 1.0
    if pred_len == 0:
        return 0.0
    return min(1.0, (pred_len / ref_len) ** 0.5) if ref_len > 0 else 0.0


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """计算BLEU分数（几何平均 + 简短惩罚）。

    对中文按字符粒度计算1-gram到4-gram的精度。
    """
    pred_tokens = _char_tokenize(prediction)
    ref_tokens = _char_tokenize(reference)
    if not pred_tokens:
        return 0.0

    # 计算各阶n-gram精度
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        clipped = _clip_count(pred_ngrams, ref_ngrams)
        total = sum(pred_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)

    # 任何一阶精度为0则BLEU=0
    if any(p == 0 for p in precisions):
        return 0.0

    # 几何平均（对数求和避免下溢）
    geo_avg = math.exp(sum(math.log(p) for p in precisions) / max_n)

    bp = _brevity_penalty(len(pred_tokens), len(ref_tokens))
    return bp * geo_avg


# ── 幻觉率 ──

def hallucination_rate(faithfulness_score: float) -> float:
    """幻觉率 = 1 - faithfulness。

    复用RAGAS的faithfulness结果，faithfulness=1表示无幻觉。
    """
    if faithfulness_score is None:
        return None
    return round(1.0 - faithfulness_score, 4)


# ── 完整性（LLM评估）──

_COMPLETENESS_PROMPT = """你是一个评估专家。请判断生成的回答是否覆盖了参考答案中的关键信息点。

参考答案（ground_truth）:
{ground_truth}

生成的回答:
{answer}

请按以下步骤评估：
1. 从参考答案中提取关键信息点（法条编号、具体数字、核心规则等）
2. 检查每个关键信息点是否在生成的回答中被覆盖
3. 计算覆盖率 = 被覆盖的信息点数 / 总信息点数

请以JSON格式输出：
{{"key_points": ["信息点1", "信息点2", ...], "covered": ["被覆盖的信息点", ...], "completeness": 0.75}}

只输出JSON，不要其他内容。"""


def completeness_llm(answer: str, ground_truth: str, llm) -> float:
    """使用LLM评估回答的完整性。

    Args:
        answer: 生成的回答
        ground_truth: 参考答案
        llm: LangChain ChatOpenAI实例

    Returns:
        完整性分数 (0.0-1.0)
    """
    prompt = _COMPLETENESS_PROMPT.format(
        ground_truth=ground_truth, answer=answer
    )
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # 处理可能的markdown代码块包裹
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        result = json.loads(content)
        return float(result.get("completeness", 0.0))
    except Exception as e:
        print(f"[completeness_llm] Error: {e}")
        return 0.0


# ── 主入口 ──

def compute_response_metrics(
    answers: List[str],
    ground_truths: List[str],
    faithfulness_scores: Optional[List[float]] = None,
    llm=None,
) -> Dict:
    """计算响应评估指标。

    Args:
        answers: 生成的回答列表
        ground_truths: 参考答案列表
        faithfulness_scores: RAGAS faithfulness分数列表（可选，用于计算幻觉率）
        llm: LangChain LLM实例（可选，用于计算完整性）

    Returns:
        {
            "rouge_l": 0.45,
            "bleu": 0.32,
            "hallucination_rate": 0.28,  # 仅当faithfulness_scores提供时
            "completeness": 0.75,        # 仅当llm提供时
            "per_query": [...],
        }
    """
    n = len(answers)
    if n == 0:
        return {"rouge_l": 0.0, "bleu": 0.0, "per_query": []}

    per_query = []
    for i in range(n):
        pq = {
            "rouge_l": round(rouge_l(answers[i], ground_truths[i]), 4),
            "bleu": round(bleu(answers[i], ground_truths[i]), 4),
        }

        # 幻觉率（复用faithfulness）
        if faithfulness_scores and i < len(faithfulness_scores):
            pq["hallucination_rate"] = hallucination_rate(faithfulness_scores[i])

        # 完整性（LLM评估，较慢，按需启用）
        if llm is not None:
            pq["completeness"] = round(
                completeness_llm(answers[i], ground_truths[i], llm), 4
            )

        per_query.append(pq)

    # 计算均值
    metrics = {"per_query": per_query}
    for key in ["rouge_l", "bleu", "hallucination_rate", "completeness"]:
        values = [pq[key] for pq in per_query if key in pq]
        if values:
            metrics[key] = round(sum(values) / len(values), 4)

    return metrics


def compute_response_metrics_by_type(
    answers: List[str],
    ground_truths: List[str],
    question_types: List[str],
    faithfulness_scores: Optional[List[float]] = None,
    llm=None,
) -> Dict[str, Dict]:
    """按问题类型分组计算响应指标。"""
    groups: Dict[str, Dict] = defaultdict(lambda: {"answers": [], "ground_truths": [], "faithfulness": []})
    for i, qtype in enumerate(question_types):
        groups[qtype]["answers"].append(answers[i])
        groups[qtype]["ground_truths"].append(ground_truths[i])
        if faithfulness_scores and i < len(faithfulness_scores):
            groups[qtype]["faithfulness"].append(faithfulness_scores[i])

    return {
        qtype: compute_response_metrics(
            g["answers"], g["ground_truths"],
            faithfulness_scores=g["faithfulness"] or None, llm=llm,
        )
        for qtype, g in groups.items()
    }
