"""报告层与指标层的**键名契约**。

为什么需要它
------------
`scripts/run_eval.py` 从 `/evaluation/status` 的 `retrieval` 字段取数并打印。
它曾写成：

    retrieval.get("P@1")     # 实际键是 precision@1
    retrieval.get("R@5")     # 实际键是 recall@5
    retrieval.get("MRR")     # 实际键是 mrr

于是**检索指标永远显示 N/A** —— 而它们其实已经算出来、就在响应体里。
键名写错不会报错、不会抛异常，只会静默丢数据；一个只看输出的人会以为
「检索指标没算」，进而去查错方向。

本文件把「消费方读取的键」与「生产方产出的键」绑成契约：
扫描 `scripts/run_eval.py` 里所有 `retrieval.get("...")`，
断言每个键都存在于 `compute_retrieval_metrics` 的实际产出中。
"""
import ast
from pathlib import Path

import pytest

from app.evaluation.metrics.retrieval import compute_retrieval_metrics

BACKEND_DIR = Path(__file__).resolve().parent.parent
RUN_EVAL = BACKEND_DIR / "scripts" / "run_eval.py"


def _keys_read_from(run_eval_path: Path, var_name: str) -> list[str]:
    """扫描脚本里 `<var_name>.get("KEY", ...)` 形式读取的键。"""
    tree = ast.parse(run_eval_path.read_text(encoding="utf-8"), filename=str(run_eval_path))
    keys = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == var_name):
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            keys.append(node.args[0].value)
    return keys


def _produced_keys() -> set[str]:
    """compute_retrieval_metrics 实际产出的键（用一条最小输入探测）。"""
    src = [[{"source": "劳动法.txt", "content": "第四十四条 x"}]]
    rel = [["劳动法第44条"]]
    return set(compute_retrieval_metrics(src, rel, k=5))


def test_run_eval_不硬编码检索指标名():
    """报告脚本必须**遍历**响应里的键，而不是硬编码指标名。

    硬编码是当初出问题的形态：脚本写 `retrieval.get("P@1")`，指标层产出的是
    `precision@1` —— 键名对不上，检索指标永远显示 N/A，而数据就在响应体里。
    遍历之后「指标层改名 / 新增指标」都不会再造成静默丢数据，故本用例要求
    `retrieval` 上**不存在任何字面量键读取**。
    """
    literal_keys = _keys_read_from(RUN_EVAL, "retrieval")
    assert literal_keys == [], (
        f"run_eval.py 又硬编码了检索指标名 {literal_keys}；"
        f"请改为遍历 retrieval.items()，实际可用键：{sorted(_produced_keys())}"
    )
    src = RUN_EVAL.read_text(encoding="utf-8")
    assert "retrieval.items()" in src, "未看到遍历写法，可能又改回硬编码了"


def test_run_eval_读取的响应键都真实存在():
    """同样校验响应指标（幻觉率 / 完整性）。"""
    read_keys = _keys_read_from(RUN_EVAL, "response")
    produced = {"hallucination_rate", "completeness"}
    missing = [k for k in read_keys if k not in produced]
    assert not missing, f"run_eval.py 读取了不存在的响应指标键 {missing}"
