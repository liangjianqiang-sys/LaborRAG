"""断点续评的**存储层**：任务 / 逐题结果 / 报告的落盘与读取。

从 `persistent.py` 拆出（2026-09-20）。拆分的依据是**可验证性**：
这一层是纯文件 I/O + 纯计算，可以完全离线测试（见 `tests/test_eval_persistent.py`
的 24 个用例）；而 `persistent.py` 里剩下的执行路径（`_execute_task` /
`_run_ragas_scoring`）需要真实引擎与 LLM 才能验证，**故意保持不动** ——
在没有验证手段前拆分它，等于制造无法发现的回归。

⚠️ 三个目录常量（`_TASKS_DIR` / `_RESULTS_DIR` / `_REPORTS_DIR`）是**模块级**的，
测试通过 monkeypatch 本模块的属性来重定向到临时目录。
因此**调用方必须用模块属性访问**（`store._TASKS_DIR`），
不能 `from ... import _TASKS_DIR` —— 后者是 import 时的快照，patch 不生效。
（与 `app/api/deps.py` 的 `deps.engine` 是同一条约定。）
"""
import json
import os
import re
from typing import Dict, Any, List, Optional

from app.utils.files import file_lock
from app.evaluation.utils import sanitize_floats


# ─── 存储根目录 ──────────────────────────────────────────────

_BACKEND_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _get_eval_data_dir() -> str:
    """获取评估数据存储根目录，优先使用环境变量，回退到backend目录下。"""
    from app.core.config import settings
    return os.getenv("EVAL_DATA_DIR", os.path.join(str(settings.RUNTIME_DATA_DIR), "evaluation"))


_PERSISTENT_ROOT = os.path.join(_get_eval_data_dir(), "persistent")
_TASKS_DIR = os.path.join(_PERSISTENT_ROOT, "tasks")
_RESULTS_DIR = os.path.join(_PERSISTENT_ROOT, "results")
_REPORTS_DIR = os.path.join(_PERSISTENT_ROOT, "reports")


def _ensure_dirs():
    """确保存储目录存在。"""
    for d in (_TASKS_DIR, _RESULTS_DIR, _REPORTS_DIR):
        os.makedirs(d, exist_ok=True)


# ─── 任务元数据操作 ──────────────────────────────────────────

def _task_path(task_id: str) -> str:
    return os.path.join(_TASKS_DIR, f"{task_id}.json")


def _results_path(task_id: str) -> str:
    return os.path.join(_RESULTS_DIR, f"{task_id}.jsonl")


def _report_path(task_id: str) -> str:
    return os.path.join(_REPORTS_DIR, f"{task_id}_report.json")


def _load_task(task_id: str) -> Optional[Dict[str, Any]]:
    """加载任务元数据，不存在返回None。"""
    path = _task_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_task(task: Dict[str, Any]):
    """保存任务元数据。"""
    _ensure_dirs()
    with open(_task_path(task["task_id"]), "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(task), f, ensure_ascii=False, indent=2)


def _append_result(task_id: str, record: Dict[str, Any]):
    """追加一条结果到jsonl文件（线程安全）。

    ⚠️ 追加前必须确保**上一行已换行结束**（2026-09-20 修复）。
    进程被 kill 时最后一行可能只写了一半（没有换行符），此时直接 append 会把
    新记录接到那半行后面 —— 两条一起变成非法 JSON，于是**新记录静默丢失**：
    `_load_results` 只会跳过损坏行，无从知道丢了什么。

    实测：文件内容为 `{"index":0}\\n{"index":1, "broken": ` 时追加 index=2，
    读回只剩 index=0（index=2 被并进损坏行）。

    这个缺陷只在「崩溃后继续追加」这一条路径上出现 —— 而「崩溃后能续评」
    正是本模块存在的理由，所以必须补上。
    """
    with file_lock:
        _ensure_dirs()
        path = _results_path(task_id)
        # 检查文件末尾是否有换行（空文件 / 不存在则无需补）
        need_newline = False
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                need_newline = f.read(1) != b"\n"
        with open(path, "a", encoding="utf-8") as f:
            if need_newline:
                f.write("\n")
            f.write(json.dumps(sanitize_floats(record), ensure_ascii=False) + "\n")


def _load_results(task_id: str) -> List[Dict[str, Any]]:
    """读取已有结果列表。"""
    path = _results_path(task_id)
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _save_report(task_id: str, report: Dict[str, Any]):
    """保存最终报告。"""
    _ensure_dirs()
    with open(_report_path(task_id), "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(report), f, ensure_ascii=False, indent=2)


def _load_report(task_id: str) -> Optional[Dict[str, Any]]:
    """读取已保存的报告文件，不存在返回None。"""
    path = _report_path(task_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _clear_task_files(task_id: str):
    """删除某个任务的所有文件（用于重新评估）。"""
    for path_func in (_task_path, _results_path, _report_path):
        p = path_func(task_id)
        if os.path.exists(p):
            os.remove(p)


def _clear_legacy_report(task_id: str):
    """删除evaluation_reports/下与该任务对应的旧格式报告文件。"""
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                              "evaluation_reports")
    if not os.path.exists(report_dir):
        return
    # 旧格式报告文件名以task_id中的日期时间开头，如 eval_unknown_20260530_184606.json
    # 匹配规则：文件名包含task_id中的时间戳部分（YYYYMMDD_HHMMSS）
    import re
    ts_match = re.search(r'(\d{8}_\d{6})', task_id)
    if not ts_match:
        return
    ts_part = ts_match.group(1)
    for f in os.listdir(report_dir):
        if f.endswith(".json") and ts_part in f:
            os.remove(os.path.join(report_dir, f))


def load_all_tasks() -> List[Dict[str, Any]]:
    """加载所有任务元数据（模块级函数，供启动时使用）。"""
    _ensure_dirs()
    tasks = []
    for filename in os.listdir(_TASKS_DIR):
        if filename.endswith(".json"):
            try:
                task = _load_task(filename[:-5])
                if task:
                    tasks.append(task)
            except Exception:
                continue
    return tasks
