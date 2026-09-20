"""retrieval_metrics 检索指标单测。

纯函数，仅依赖标准库。覆盖：单指标数值正确性、空输入边界、
中文数字转换、法条标准化（长名优先）、端到端 compute。

注：中文数字转换的实现已收敛到 app.utils.law_refs（唯一真相源），
故此处从该模块导入 cn_to_int 而非本模块的私有副本。
"""
from app.utils.law_refs import cn_to_int
from app.evaluation.metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    mrr,
    average_precision,
    f1_at_k,
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
    assert cn_to_int("四十四") == 44
    assert cn_to_int("一百零八") == 108
    assert cn_to_int("十") == 10
    assert cn_to_int("10") == 10
    assert cn_to_int("5") == 5


def test_cn_to_int_含〇不得算错():
    """回归：本模块曾自带一份不含「〇」的字符表，把「一百〇一」算成 100。

    失真后果：评估侧把「第一百〇一条」当成第 100 条，检索命中也被判未命中。
    这类错误不抛异常，只让指标悄悄变差。
    """
    assert cn_to_int("一百〇一") == 101
    assert cn_to_int("一百零一") == 101
    assert cn_to_int("一百〇五") == 105


def test_cn_to_int_解析失败返回_None_而非半截值():
    """遇到无法解析的字符必须返回 None。

    原实现是 `break` 后返回部分结果 —— 例如「一百X」会静默变成 100，
    调用方拿错条号却毫不知情。
    """
    assert cn_to_int("一百X") is None
    assert cn_to_int("四十七abc") is None
    assert cn_to_int("") is None


def test_normalize_article_含〇归一正确():
    """含〇的条号必须与含零的写法归一到同一个值。"""
    assert _normalize_article("劳动法第一百〇一条") == "劳动法第101条"
    assert _normalize_article("劳动法第一百零一条") == "劳动法第101条"


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


def test_聚合指标键集合完整且空输入一致():
    """聚合结果必须包含 docstring 宣称的全部指标，且空/非空两条路径键集合相同。

    回归：实现此前只返回 precision@1 与 recall@k，而 docstring 与 README 都宣称
    覆盖 P@3/P@5/MRR/MAP —— 答辩时若被要求看 MRR 会拿不出来。
    且空输入分支只返回 2 个键，下游读 mrr / map 会 KeyError。
    """
    src = [[{"source": "劳动法.txt", "content": "第四十四条 加班工资"}]]
    rel = [["劳动法第44条"]]

    m = compute_retrieval_metrics(src, rel, k=5)
    expected = {
        "precision@1", "precision@3", "precision@5",
        "recall@5", "f1@5", "mrr", "map",
    }
    assert expected <= set(m), f"聚合结果缺少指标：{sorted(expected - set(m))}"

    empty = compute_retrieval_metrics([], [], k=5)
    assert set(empty) - {"per_query"} == expected, (
        "空输入与非空路径的键集合必须一致，否则下游读 mrr/map 会 KeyError"
    )
