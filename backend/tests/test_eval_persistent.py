"""`persistent.py`（断点续评）的存储层与纯计算测试。

为什么先写测试、再考虑拆分
--------------------------
这个文件 850 行（其中 `PersistentEvalManager` 占 680 行），且**此前零测试覆盖**。
它的核心价值是「崩溃后能续评」—— 而这条路径要真跑评估（消耗 LLM 额度）
才能端到端验证。所以**直接拆分的回归风险无法用现有手段发现**。

本文件覆盖**可离线验证的部分**（文件 I/O + 纯计算），作为拆分前的安全网：
拆完之后这些用例必须依然全绿。

不覆盖（需要真实引擎/LLM）：`_execute_task`、`_run_ragas_scoring`、
`_evaluate_single`、`_run_in_thread`。

关键约定：三个存储目录是**模块级常量**，测试用 monkeypatch 指到临时目录，
绝不碰真实的 `runtime_data/evaluation/persistent/`。
"""
import json
import os

import pytest

from app.evaluation import persistent as P


@pytest.fixture
def store(tmp_path, monkeypatch):
    """把三个存储目录重定向到临时目录。"""
    monkeypatch.setattr(P, "_TASKS_DIR", str(tmp_path / "tasks"))
    monkeypatch.setattr(P, "_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(P, "_REPORTS_DIR", str(tmp_path / "reports"))
    return tmp_path


def _task(task_id="eval_20260920_120000", **kw):
    base = {
        "task_id": task_id,
        "status": "created",
        "total_questions": 20,
        "completed_indices": [],
        "failed_indices": [],
        "rag_mode": "rerank",
        "question_type": "all",
    }
    base.update(kw)
    return base


# ── 存储层：存读往返 ────────────────────────────────────────────


def test_任务元数据_存读往返(store):
    t = _task(completed_indices=[0, 1])
    P.save_task(t)
    assert P._load_task(t["task_id"]) == t


def test_加载不存在的任务返回_None(store):
    assert P._load_task("不存在") is None


def test_报告存读往返(store):
    P._save_report("t1", {"summary": {"success": 3}, "task_id": "t1"})
    assert P._load_report("t1")["summary"]["success"] == 3


def test_加载不存在的报告返回_None(store):
    assert P._load_report("不存在") is None


def test_保存时非有限浮点被清洗成合法_JSON(store):
    """`NaN` / `Infinity` 不是合法 JSON —— 直接 dump 会产出**别人读不了**的文件。

    评估结果里出现 NaN 是常态（某题某指标算不出来）。若不清洗，落盘的文件
    在 `json.load` 时抛错，而 `_load_results` 只捕获 `JSONDecodeError` —— 
    续评时整条结果链会断在这里。
    """
    P.save_task(_task(score=float("nan"), other=float("inf")))
    raw = open(P._task_path("eval_20260920_120000"), encoding="utf-8").read()
    json.loads(raw)  # 不抛 → 合法 JSON
    assert "NaN" not in raw and "Infinity" not in raw


# ── 存储层：逐题结果实时落地（断点续评的基石） ──────────────────


def test_结果按行追加_可累加(store):
    for i in range(3):
        P._append_result("t1", {"index": i, "answer": f"a{i}"})
    got = P._load_results("t1")
    assert [r["index"] for r in got] == [0, 1, 2]


def test_结果文件不存在时返回空列表(store):
    assert P._load_results("从未写过") == []


def test_损坏行被跳过_其余仍可读(store):
    """jsonl 的价值就在于此：**一行坏了不该毁掉整个文件**。

    进程被 kill 时最后一行很容易写一半 —— 续评必须能读回前面的部分，
    否则「断点续评」在真实崩溃场景下就是失效的。
    """
    P._append_result("t1", {"index": 0})
    with open(P._results_path("t1"), "a", encoding="utf-8") as f:
        f.write('{"index": 1, "broken": ')  # 半行
    P._append_result("t1", {"index": 2})
    got = P._load_results("t1")
    assert [r["index"] for r in got] == [0, 2], f"损坏行未跳过或误伤相邻行：{got}"


def test_清理任务文件_三个文件都删(store):
    P.save_task(_task("t1"))
    P._append_result("t1", {"index": 0})
    P._save_report("t1", {"task_id": "t1"})
    P._clear_task_files("t1")
    assert P._load_task("t1") is None
    assert P._load_results("t1") == []
    assert P._load_report("t1") is None


def test_清理不存在的任务不报错(store):
    P._clear_task_files("从未存在")


def test_load_all_tasks_跳过非_json与损坏文件(store):
    P.save_task(_task("good1"))
    P.save_task(_task("good2"))
    os.makedirs(P._TASKS_DIR, exist_ok=True)
    open(os.path.join(P._TASKS_DIR, "readme.txt"), "w").write("x")       # 非 json
    open(os.path.join(P._TASKS_DIR, "broken.json"), "w").write("{坏")     # 损坏
    ids = {t["task_id"] for t in P.load_all_tasks()}
    assert ids == {"good1", "good2"}, ids


# ── 进度摘要 ────────────────────────────────────────────────────


def test_进度摘要的计数与百分比(store):
    P.save_task(_task("t1", total_questions=20,
                      completed_indices=[0, 1, 2], failed_indices=[3]))
    p = P.PersistentEvalManager(None).get_task_progress("t1")
    assert (p["total"], p["completed"], p["failed"], p["remaining"]) == (20, 3, 1, 16)
    assert p["progress_pct"] == 15.0


def test_进度摘要_总数为零时不除零(store):
    P.save_task(_task("t1", total_questions=0))
    assert P.PersistentEvalManager(None).get_task_progress("t1")["progress_pct"] == 0


def test_进度摘要_任务不存在时返回_error(store):
    assert "error" in P.PersistentEvalManager(None).get_task_progress("不存在")


# ── 任务状态机（_check_task）────────────────────────────────────


def _mgr():
    return P.PersistentEvalManager(None)


def test_校验_任务不存在抛错(store):
    with pytest.raises(ValueError, match="不存在"):
        _mgr()._check_task("不存在")


def test_校验_运行中抛错(store):
    P.save_task(_task("t1", status="running"))
    with pytest.raises(ValueError, match="正在运行"):
        _mgr()._check_task("t1")


def test_校验_created_默认不允许但可放行(store):
    P.save_task(_task("t1", status="created"))
    with pytest.raises(ValueError, match="尚未启动"):
        _mgr()._check_task("t1")
    assert _mgr()._check_task("t1", allow_created=True)["status"] == "created"


def test_校验_completed_默认不允许但可放行(store):
    P.save_task(_task("t1", status="completed"))
    with pytest.raises(ValueError, match="已完成"):
        _mgr()._check_task("t1")
    assert _mgr()._check_task("t1", allow_completed=True)["status"] == "completed"


def test_校验_need_failed_且无失败项时抛错(store):
    P.save_task(_task("t1", status="stopped", failed_indices=[]))
    with pytest.raises(ValueError, match="无失败条目"):
        _mgr()._check_task("t1", need_failed=True)


# ── 报告聚合（纯计算）──────────────────────────────────────────


def test_嵌套均值_忽略缺失项并四舍五入():
    results = [
        {"retrieval": {"mrr": 1.0, "map": 0.5}},
        {"retrieval": {"mrr": 0.0}},                 # map 缺失 → 不计入
        {"retrieval": {}},                            # 整项缺失
    ]
    got = P.PersistentEvalManager._avg_nested_key(results, "retrieval", ["mrr", "map"])
    assert got == {"mrr": 0.5, "map": 0.5}, got


def test_嵌套均值_全缺失时返回空字典():
    assert P.PersistentEvalManager._avg_nested_key([{"x": {}}], "x", ["k"]) == {}


def test_报告汇总_计数与成功率(store):
    task = _task("t1", status="completed", total_questions=4)
    report = _mgr()._compute_report(task, [{"retrieval": {}}] * 3, [{"index": 3, "error": "boom"}])
    s = report["summary"]
    assert (s["total"], s["success"], s["failed"], s["remaining"]) == (4, 3, 1, 0)
    assert s["success_rate"] == 75.0


def test_报告汇总_失败项只截取题干前_50_字(store):
    task = _task("t1")
    long_q = "问" * 120
    report = _mgr()._compute_report(task, [], [{"index": 7, "question": long_q, "error": "超时"}])
    item = report["failed_items"][0]
    assert item["index"] == 7 and item["error"] == "超时"
    assert len(item["question"]) == 50


def test_报告汇总_无成功结果时不带均值字段(store):
    """`retrieval_avg` 只在有数据时出现 —— 否则报告里会多出空字典，
    下游（`report.py` / 前端）会把它当成「算过了但都是 0」。"""
    report = _mgr()._compute_report(_task("t1"), [], [])
    assert "retrieval_avg" not in report and "response_avg" not in report
