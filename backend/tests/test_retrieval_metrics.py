"""retrieval_metrics 检索指标单测。

纯函数，仅依赖标准库。覆盖：单指标数值正确性、空输入边界、
中文数字转换、法条标准化（长名优先）、端到端 compute。
"""
from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    average_precision,
    f1_at_k,
    _cn_to_int,
    _normalize_article,
    extract_article_id,
    compute_retrieval_metrics,
    compute_retrieval_metrics_by_type,
)


def test_single_query_metrics():
    """构造 retrieved/relevant，断言各指标数值。"""
    retrieved = ["劳动法第44条", "劳动合同法第47条", "某法第1条"]
    relevant = ["劳动合同法第47条"]

    assert precision_at_k(retrieved, relevant, 1) == 0.0   # 首位不相关
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 5) == 1.0      # 相关的都在前5
    assert mrr(retrieved, relevant) == 0.5                  # 第2个命中 → 1/2
    assert average_precision(retrieved, relevant) == 0.5


def test_empty_inputs():
    """空 retrieved / 空 relevant 应返回 0.0，不抛异常。"""
    assert precision_at_k([], ["a"], 5) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0
    assert mrr([], ["a"]) == 0.0
    assert mrr(["a"], []) == 0.0
    assert average_precision([], ["a"]) == 0.0
    assert average_precision(["a"], []) == 0.0
    assert f1_at_k([], [], 5) == 0.0


def test_cn_to_int():
    """中文数字 → 整数。"""
    assert _cn_to_int("四十四") == 44
    assert _cn_to_int("一百零八") == 108
    assert _cn_to_int("十") == 10
    assert _cn_to_int("10") == 10
    assert _cn_to_int("5") == 5


def test_normalize_article_priority():
    """长名优先匹配：'劳动合同法' 不应被 '劳动法' 截断。"""
    assert _normalize_article("劳动合同法第四十七条") == "劳动合同法第47条"
    assert _normalize_article("劳动法第四十四条") == "劳动法第44条"
    assert _normalize_article("第47条") == "第47条"  # 无法律前缀
    # 阿拉伯数字直通
    assert _normalize_article("劳动法第47条") == "劳动法第47条"


def test_extract_article_id_and_compute():
    """extract_article_id 从 chunk 抽 ID + compute 端到端结构。"""
    # 从 source 文件名 + content 抽 ID
    aid = extract_article_id("中华人民共和国劳动法.txt", "第四十四条 加班工资...")
    assert aid == "劳动法第44条"
    # content 无第X条 → ""
    assert extract_article_id("x.txt", "无法条编号内容") == ""

    # 端到端 compute：2 个 query，都完美命中
    sources_list = [
        [{"source": "中华人民共和国劳动法.txt", "content": "第四十四条 加班工资..."}],
        [{"source": "中华人民共和国劳动合同法.txt", "content": "第四十七条 经济补偿..."}],
    ]
    relevant_articles_list = [["劳动法第44条"], ["劳动合同法第47条"]]
    result = compute_retrieval_metrics(sources_list, relevant_articles_list, k=5)
    assert result["precision@1"] == 1.0
    assert result["recall@5"] == 1.0
    assert len(result["per_query"]) == 2
    assert result["per_query"][0]["retrieved_ids"][0] == "劳动法第44条"

    # 空输入早返回
    empty = compute_retrieval_metrics([], [], k=5)
    assert empty["precision@1"] == 0.0
    assert empty["per_query"] == []

    # 按类型分组
    by_type = compute_retrieval_metrics_by_type(
        sources_list, relevant_articles_list, ["retrieve", "retrieve"], k=5
    )
    assert "retrieve" in by_type
    assert by_type["retrieve"]["precision@1"] == 1.0
