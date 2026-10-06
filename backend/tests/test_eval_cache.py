"""评估缓存（`_save_cache` / `_load_cache`）的测试 —— Bug 21 的回归网。

为什么需要
----------
`runner.py` 原本只打印一句「Answers cached. **Safe to retry if scoring fails.**」，
但代码里**只有 `_save_cache`、没有 `_load_cache`** —— 缓存写出去之后
**再没有任何代码读过它**，那句注释描述的能力实际不存在。

后果在内存紧张的环境下很致命：Phase 3（RAGAS）崩溃时，Phase 1 的 20 次生成
（约 230k token）**必须全部重跑**，等于白烧一次。

本文件守住两条不变量：

1. **存进去的能原样读回来** —— 含 2026-10-06 补的 `sources_raw`（供 Phase 2 算
   检索指标）与 `child_contexts`（供 RAGAS 的 Context Precision/Recall）。
   缺这两个就只是"读回一半数据"，Phase 1 还是得重跑。
2. **题目集不一致时绝不能复用** —— 缓存文件名只带 label 与时间戳、**不带筛选条件**。
   上一次跑 `--difficulty easy`（6 题）、这次跑 `--subset full`（20 题），
   直接复用会把 6 题的答案当成 20 题的结果 —— 而且**不会报错**，
   只会产出一份看起来完全正常的错报告。这是本文件最重要的一条。
"""
import json

import pytest

from app.evaluation import runner as R


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """把缓存目录指到临时目录，绝不碰真实 evaluation_reports/cache/。"""
    d = tmp_path / "cache"
    monkeypatch.setattr(R, "_CACHE_DIR", str(d))
    return d


def _runner():
    """绕过 `__init__`（它会调 `get_embeddings()` 加载 BGE-M3，约 60s）。

    这两个方法只读写文件，不依赖实例状态；但 `_execute` 的尾部会读
    `self.ragas_llm` 当参数传给 `compute_response_metrics` —— 所以补一个占位属性，
    否则 `__new__` 出来的实例会 AttributeError。
    """
    r = R.EvalRunner.__new__(R.EvalRunner)
    r.ragas_llm = None
    return r


def _save(runner, label, questions, **over):
    n = len(questions)
    kw = dict(
        questions=questions,
        answers=[f"答案{i}" for i in range(n)],
        contexts=[[f"父块{i}"] for i in range(n)],
        child_contexts=[[f"子块{i}"] for i in range(n)],
        ground_truths=[f"参考答案{i}" for i in range(n)],
        sources_raw=[[{"source": "劳动合同法.txt", "content": f"片段{i}"}] for i in range(n)],
        details=[{"question": q, "question_type": "retrieve", "relevant_articles": ["劳动合同法第82条"]}
                 for q in questions],
    )
    kw.update(over)
    runner._save_cache(label, **kw)


def _fake_gen_result(item):
    """`_gen_answer` 的返回值形状（字段与 runner 里收集用的一致）。"""
    return {
        "question": item["question"], "answer": "a", "contexts": ["c"],
        "child_contexts": ["cc"], "ground_truth": item["ground_truth"],
        "relevant_articles": [], "question_type": "retrieve",
        "difficulty": "easy", "test_dimension": "",
        "eval_subset": ["full"], "law": "", "sources": [],
        "source_count": 0, "rag_mode": "agent",
    }


def _dataset(questions):
    """构造 `_execute` 需要的最小 dataset（字段与 golden_set 一致）。"""
    return [{"question": q, "ground_truth": f"g{i}", "relevant_articles": [],
             "question_type": "retrieve", "difficulty": "easy",
             "test_dimension": "", "eval_subset": ["full"], "law": ""}
            for i, q in enumerate(questions)]


# ── 存读往返 ────────────────────────────────────────────────────


def test_存读往返_含新增字段(cache_dir):
    r = _runner()
    _save(r, "agent", ["问题一", "问题二"])
    got = r._load_cache("agent", ["问题一", "问题二"])

    assert got is not None
    assert got["questions"] == ["问题一", "问题二"]
    assert got["answers"] == ["答案0", "答案1"]
    assert got["contexts"] == [["父块0"], ["父块1"]]
    assert got["child_contexts"] == [["子块0"], ["子块1"]]
    assert got["ground_truths"] == ["参考答案0", "参考答案1"]
    assert got["sources_raw"][0][0]["source"] == "劳动合同法.txt"
    assert got["details"][0]["relevant_articles"] == ["劳动合同法第82条"]


def test_无缓存目录时返回_None(cache_dir):
    assert _runner()._load_cache("agent", ["问题一"]) is None


def test_没有对应_label_时返回_None(cache_dir):
    r = _runner()
    _save(r, "agent", ["问题一"])
    assert r._load_cache("另一个label", ["问题一"]) is None


# ── 核心不变量：题目集必须逐项一致 ──────────────────────────────


def test_题目集不一致时不复用(cache_dir):
    """**本文件最重要的一条。**

    缓存 6 题、请求 20 题 —— 若直接复用，6 题的答案会被当成 20 题的结果，
    且**不会报错**，只产出一份看起来正常的错报告。
    """
    r = _runner()
    _save(r, "agent", [f"问题{i}" for i in range(6)])
    assert r._load_cache("agent", [f"问题{i}" for i in range(20)]) is None


def test_题目内容不同时不复用(cache_dir):
    r = _runner()
    _save(r, "agent", ["问题一", "问题二"])
    assert r._load_cache("agent", ["问题一", "完全不同的问题"]) is None


def test_题目顺序不同时不复用(cache_dir):
    """顺序也必须一致 —— 否则 answers[i] 会与 questions[i] 错位。"""
    r = _runner()
    _save(r, "agent", ["问题一", "问题二"])
    assert r._load_cache("agent", ["问题二", "问题一"]) is None


# ── 旧格式与损坏文件 ────────────────────────────────────────────


def test_旧格式缓存不复用(cache_dir):
    """2026-10-06 之前的缓存没有 sources_raw / child_contexts。

    这种缓存**无法真正跳过 Phase 1**（Phase 2 与 RAGAS 都缺输入），
    宁可重跑，也不要"复用一半"导致后半程报错。
    """
    r = _runner()
    _save(r, "agent", ["问题一"], sources_raw=[], child_contexts=[])
    assert r._load_cache("agent", ["问题一"]) is None


def test_损坏缓存文件被跳过且不抛(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "answers_agent_20260101_000000.json").write_text("{坏 JSON", encoding="utf-8")
    assert _runner()._load_cache("agent", ["问题一"]) is None


def test_多份缓存时取最新(cache_dir):
    """文件名带时间戳，倒序取第一份命中 —— 保证「重跑后用新结果」。"""
    r = _runner()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ts, ans in (("20260101_000000", "旧"), ("20260601_000000", "新")):
        (cache_dir / f"answers_agent_{ts}.json").write_text(json.dumps({
            "rag_mode": "agent", "timestamp": ts, "questions": ["问题一"],
            "answers": [ans], "contexts": [[]], "child_contexts": [[]],
            "ground_truths": ["g"], "sources_raw": [[{"source": "x"}]], "details": [{}],
        }, ensure_ascii=False), encoding="utf-8")

    assert r._load_cache("agent", ["问题一"])["answers"] == ["新"]


def test_跳过损坏文件后仍能命中更旧的有效缓存(cache_dir):
    """损坏的最新文件不该让整个缓存机制失效。"""
    r = _runner()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "answers_agent_20260601_000000.json").write_text("{坏", encoding="utf-8")
    (cache_dir / "answers_agent_20260101_000000.json").write_text(json.dumps({
        "rag_mode": "agent", "timestamp": "20260101_000000", "questions": ["问题一"],
        "answers": ["好的"], "contexts": [[]], "child_contexts": [[]],
        "ground_truths": ["g"], "sources_raw": [[{"source": "x"}]], "details": [{}],
    }, ensure_ascii=False), encoding="utf-8")

    assert r._load_cache("agent", ["问题一"])["answers"] == ["好的"]


# ── 与 _execute 的集成：命中缓存必须真的跳过生成 ────────────────


def test_命中缓存时_不调用_gen_answer(cache_dir, monkeypatch):
    """守住「跳过 Phase 1」这个**行为**，而不只是「_load_cache 返回了东西」。

    否则哪天 _execute 忘了用返回值，_load_cache 的测试依然全绿。

    ⚠️ 用**计数器**而不是「抛异常」来判定：`parallel_map` 会把单题的异常
    吞掉存进结果列表（单题失败不阻塞整体），所以 `_gen_answer` 里抛
    AssertionError 根本传不出来 —— 那样写出来的用例是空跑的。
    （这个坑是阴性对照发现的：把缓存整个去掉，用例依然全绿。）
    """
    r = _runner()
    questions = ["问题一", "问题二"]
    _save(r, "agent", questions)

    calls = {"n": 0}

    def _counting_gen(item):
        calls["n"] += 1
        return _fake_gen_result(item)

    monkeypatch.setattr(r, "_gen_answer", _counting_gen)
    # Phase 3 与响应指标不是本用例的观察对象，替换掉以免真的调 LLM
    monkeypatch.setattr(r, "_score_with_retry", lambda *a, **k: ({}, {}, []))
    monkeypatch.setattr(R, "compute_response_metrics", lambda *a, **k: {"per_query": []})

    r._execute(_dataset(questions), "agent")

    assert calls["n"] == 0, f"命中缓存时不该再生成，实际调用了 {calls['n']} 次（那是 230k token）"


def test_未命中缓存时会调用_gen_answer(cache_dir, monkeypatch):
    """非空性对照：没有缓存时必须走生成路径。

    否则上面那条用例可能因为「_execute 根本没用缓存」而假绿 ——
    两条一起看，才能证明「有缓存跳过、没缓存不跳过」。
    """
    r = _runner()
    calls = {"n": 0}

    def _counting_gen(item):
        calls["n"] += 1
        return _fake_gen_result(item)

    monkeypatch.setattr(r, "_gen_answer", _counting_gen)
    monkeypatch.setattr(r, "_score_with_retry", lambda *a, **k: ({}, {}, []))
    monkeypatch.setattr(R, "compute_response_metrics", lambda *a, **k: {"per_query": []})

    r._execute(_dataset(["全新问题"]), "agent")
    assert calls["n"] == 1, "无缓存时必须真的生成"
