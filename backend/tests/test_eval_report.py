"""evaluation.report 报告生成/对比的单测。

纯文件 I/O + 分组/排序逻辑，此前零测试覆盖。用 report_dir 注入临时目录隔离，
覆盖：save/load 往返、NaN 清洗、按 retriever_type 分组取最新、bad case 排序、
可观测性三视图。
"""
import json

from app.evaluation.report import EvalReport


def _report(tmp_path) -> EvalReport:
    return EvalReport(report_dir=str(tmp_path))


def test_save_load_往返一致(tmp_path):
    r = _report(tmp_path)
    result = {"scores": {"faithfulness": 0.9}, "retrieval": {"precision@1": 1.0}}
    fp = r.save_result("hybrid", result)
    loaded = r.load_result(fp)
    assert loaded["retriever_type"] == "hybrid"
    assert loaded["scores"]["faithfulness"] == 0.9


def test_save_result_清洗_NaN(tmp_path):
    r = _report(tmp_path)
    fp = r.save_result("hybrid", {"scores": {"x": float("nan")}})
    loaded = r.load_result(fp)
    assert loaded["scores"]["x"] is None


def test_compare_按类型分组取最新(tmp_path):
    r = _report(tmp_path)
    # 手写文件以获得可控的 timestamp（save_result 用 datetime.now，同一秒内无法区分先后）
    (tmp_path / "eval_hybrid_20260101_000000.json").write_text(json.dumps({
        "retriever_type": "hybrid", "timestamp": "20260101_000000", "scores": {"f": 0.1},
    }), encoding="utf-8")
    (tmp_path / "eval_hybrid_20260102_000000.json").write_text(json.dumps({
        "retriever_type": "hybrid", "timestamp": "20260102_000000", "scores": {"f": 0.9},
    }), encoding="utf-8")
    (tmp_path / "eval_reranked_20260101_000000.json").write_text(json.dumps({
        "retriever_type": "reranked", "timestamp": "20260101_000000", "scores": {"f": 0.5},
    }), encoding="utf-8")

    c = r.compare()
    assert c["results"]["hybrid"]["f"] == 0.9      # 同一类型保留最新
    assert c["results"]["reranked"]["f"] == 0.5    # 不同类型各自保留
    assert c["metrics"] == ["f"]                   # 从 scores 收集 key


def test_get_bad_cases_排序取最低分(tmp_path):
    r = _report(tmp_path)
    (tmp_path / "eval_x_20260101_000000.json").write_text(json.dumps({
        "retriever_type": "x", "timestamp": "20260101_000000",
        "details": [
            {"question": "好", "response": {"rouge_l": 0.5, "completeness": 0.8, "hallucination_rate": 0.1}},
            {"question": "坏", "response": {"rouge_l": 0.1, "completeness": 0.2, "hallucination_rate": 0.9}},
        ],
    }), encoding="utf-8")

    bad = r.get_bad_cases(top_n=1)
    assert bad[0]["question"] == "坏"          # 综合分更低
    assert "_score" not in bad[0]              # 内部字段不应泄漏


def test_get_observability_三视图(tmp_path):
    r = _report(tmp_path)
    (tmp_path / "eval_x_20260101_000000.json").write_text(json.dumps({
        "retriever_type": "x", "timestamp": "20260101_000000",
        "retrieval": {"precision@1": 0.9, "recall@5": 0.8},
        "triad": {"faithfulness": 0.9},
        "response": {"hallucination_rate": 0.1, "completeness": 0.8},
        "details": [{"answer": "有效回答"}, {"answer": "无法回答"}],
    }), encoding="utf-8")

    obs = r.get_observability()
    assert obs["retrieval_quality"]["precision@1"] == 0.9
    assert obs["generation_quality"]["faithfulness"] == 0.9
    assert obs["business"]["resolution_rate"] == 1.0            # 2/2 非空答案
    assert obs["business"]["first_answer_usability"] == 0.5     # 1/2 无「无法」


def test_空目录_返回占位信息(tmp_path):
    r = _report(tmp_path)
    assert r.compare() == {"message": "暂无评估结果"}
    assert r.get_observability() == {"message": "暂无评估结果"}
    assert r.get_bad_cases() == []
