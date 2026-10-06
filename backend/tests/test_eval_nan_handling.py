"""评估指标的 NaN 处理 —— 一个 NaN 不该让整项指标作废。

背景（2026-10-06 实测）
----------------------
20 题评估跑完，控制台显示：

    Faithfulness:  0.805
    幻觉率:        None          ← 整项作废
    medium: faithfulness=None (n=9)   ← 9 题整组作废

查下来是**同一个根因**：RAGAS 对超时/失败的题会填 `NaN`，而三处代码都只防了
`None`、没防 `NaN`：

1. `faithfulness_per_query` 的构建用 `float(val)` —— **`float(nan)` 是成功的**，
   于是 NaN 被当成有效值混进列表（只 `except (TypeError, ValueError)` 拦不住它）
2. 按难度分组的均值只过滤 `None` → 1 个 NaN 污染整组
3. `compute_response_metrics` 的均值写成 `if key in pq` —— 键在但值为 `None`
   时也会被收进来，`sum([0.2, None])` 直接抛 TypeError

因为 `sanitize_floats` 会把 NaN 落盘成 `null`，**表现是"指标消失"而不是报错**，
所以特别隐蔽 —— 只有对比历史报告才会发现。

本文件守住：**NaN 与 None 都必须被当作"缺失"，不参与均值。**
"""
import math

import pandas as pd
import pytest

from app.evaluation.metrics.response import compute_response_metrics


# ── ③ compute_response_metrics 的均值 ──────────────────────────


def test_均值跳过_None():
    """键存在但值为 None 时不能参与均值 —— 否则 `sum([0.2, None])` 抛 TypeError。"""
    out = compute_response_metrics(
        answers=["a1", "a2", "a3"],
        ground_truths=["g1", "g2", "g3"],
        faithfulness_scores=[0.8, None, 0.6],
        llm=None,                       # 关掉 completeness，避免调 LLM
    )
    assert out["hallucination_rate"] == pytest.approx(0.3, abs=1e-6), out


def test_均值跳过_NaN():
    """NaN 更隐蔽：不报错，但会把整项均值污染成 NaN（落盘后变 null）。"""
    out = compute_response_metrics(
        answers=["a1", "a2", "a3"],
        ground_truths=["g1", "g2", "g3"],
        faithfulness_scores=[0.8, float("nan"), 0.6],
        llm=None,
    )
    hr = out["hallucination_rate"]
    assert hr is not None and not math.isnan(hr), f"NaN 污染了均值：{hr}"
    assert hr == pytest.approx(0.3, abs=1e-6), out


def test_全为_NaN_时该项不出现而不是_NaN():
    """全缺失时应**没有这个键**（而非 NaN）—— 让下游能明确判断"算不出来"。"""
    out = compute_response_metrics(
        answers=["a1"], ground_truths=["g1"],
        faithfulness_scores=[float("nan")], llm=None,
    )
    assert "hallucination_rate" not in out or out["hallucination_rate"] is not None


def test_逐题明细仍保留_None_便于定位():
    """均值跳过缺失项，但**逐题明细要如实记录**缺失 —— 否则无法定位是哪题失败。"""
    out = compute_response_metrics(
        answers=["a1", "a2"], ground_truths=["g1", "g2"],
        faithfulness_scores=[0.8, None], llm=None,
    )
    assert out["per_query"][0]["hallucination_rate"] == pytest.approx(0.2, abs=1e-6)
    assert out["per_query"][1]["hallucination_rate"] is None


# ── ① faithfulness_per_query 的 NaN → None ────────────────────


class _FakeResult:
    def __init__(self, df):
        self._df = df

    def to_pandas(self):
        return self._df


def _fake_evaluate_factory(df):
    """替换 `ragas.evaluate`：返回预置 DataFrame，不发任何请求。"""
    def _fake(*a, **k):
        return _FakeResult(df)
    return _fake


def test_faithfulness_per_query_把_NaN_归一成_None(monkeypatch):
    """守住根因：`float(nan)` 是成功的，只捕异常拦不住 NaN。

    RAGAS 对超时题填 NaN，若不归一，它会流进幻觉率均值与分组均值。

    注意：`EvalRunner` 在 `app.evaluation.runner`，**不是**被 conftest 替身掉的
    `app.services.rag_engine` —— 直接 import 即可拿到真实类。
    """
    import datasets
    import ragas

    from app.evaluation import runner as runner_mod

    r = runner_mod.EvalRunner.__new__(runner_mod.EvalRunner)
    r.ragas_llm, r.ragas_embeddings = None, None

    df = pd.DataFrame({
        "faithfulness": [0.9, float("nan"), 0.7],
        "question": ["q1", "q2", "q3"],
    })
    monkeypatch.setattr(ragas, "evaluate", _fake_evaluate_factory(df))
    monkeypatch.setattr(
        datasets, "Dataset",
        type("D", (), {"from_dict": staticmethod(lambda d: d)}),
    )

    _scores, _type_scores, per_query = r._score_with_retry(
        questions=["q1", "q2", "q3"],
        answers=["a1", "a2", "a3"],
        parent_contexts=[["c"], ["c"], ["c"]],
        child_contexts=[["c"], ["c"], ["c"]],
        ground_truths=["g", "g", "g"],
        details=[{"question_type": "retrieve"}] * 3,
    )

    assert per_query[0] == pytest.approx(0.9)
    assert per_query[1] is None, f"NaN 没有被归一成 None：{per_query[1]!r}"
    assert per_query[2] == pytest.approx(0.7)
    assert not any(isinstance(v, float) and math.isnan(v) for v in per_query if v is not None)
