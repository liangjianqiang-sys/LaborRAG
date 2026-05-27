"""断点续评管理器：支持任务持久化、实时落地、断点续评、异常恢复。

与原有 EvalRunner 完全独立，不修改原有逻辑。
通过组合方式调用 EvalRunner，在其基础上增加：
  - 任务持久化（task JSON）
  - 逐题结果实时落地（jsonl 追加）
  - 断点续评（崩溃后从断点继续）
  - 重新评估（清空缓存从头开始）
  - 单题异常不阻塞整体流程
  - 最终报告自动统计

文件存储结构：
  evaluation_reports/
  └── persistent/
      ├── tasks/                        # 任务元数据
      │   └── {task_id}.json            # 任务进度 + 配置
      ├── results/                      # 逐题结果（jsonl追加）
      │   └── {task_id}.jsonl           # 每行一题完整结果
      └── reports/                      # 最终报告
          └── {task_id}_report.json      # 汇总统计报告
"""
import json
import os
import uuid
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.evaluation.utils import sanitize_floats, parallel_map, file_lock


# ─── 存储根目录 ──────────────────────────────────────────────

_BACKEND_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PERSISTENT_ROOT = os.path.join(_BACKEND_ROOT, "evaluation_reports", "persistent")
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


def _save_task(task: Dict[str, Any]):
    """保存任务元数据。"""
    _ensure_dirs()
    with open(_task_path(task["task_id"]), "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(task), f, ensure_ascii=False, indent=2)


def _append_result(task_id: str, record: Dict[str, Any]):
    """追加一条结果到jsonl文件（线程安全）。"""
    with file_lock:
        _ensure_dirs()
        with open(_results_path(task_id), "a", encoding="utf-8") as f:
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


def _load_all_tasks() -> List[Dict[str, Any]]:
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


# ─── 断点续评管理器 ──────────────────────────────────────────

class PersistentEvalManager:
    """断点续评管理器：独立于原有EvalRunner，组合调用实现持久化评估。

    用法：
        manager = PersistentEvalManager(rag_engine)
        task_id = manager.create_task(rag_mode="agent", sample_count=10)
        manager.start(task_id)              # 后台启动
        manager.resume(task_id)             # 断点续评
        manager.restart(task_id)            # 重新评估
        report = manager.get_report(task_id)
    """

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self._running_tasks: Dict[str, threading.Thread] = {}

    # ─── 任务创建 ──────────────────────────────────────────

    def create_task(
        self,
        rag_mode: str = None,
        sample_count: int = None,
        question_type: str = None,
    ) -> str:
        """创建新的评估任务，返回task_id。

        不立即执行，仅创建任务元数据。
        """
        task_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        task = {
            "task_id": task_id,
            "status": "created",          # created / running / paused / completed / failed
            "rag_mode": rag_mode,
            "sample_count": sample_count,
            "question_type": question_type,
            "total_questions": 0,
            "completed_indices": [],       # 已完成的题目索引列表
            "failed_indices": [],          # 失败的题目索引列表
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        _save_task(task)
        return task_id

    # ─── 任务列表查询 ──────────────────────────────────────

    def list_tasks(self) -> List[Dict[str, Any]]:
        """列出所有任务（按创建时间倒序）。"""
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
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取单个任务详情。"""
        return _load_task(task_id)

    def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """获取任务进度摘要。"""
        task = _load_task(task_id)
        if not task:
            return {"error": f"任务 {task_id} 不存在"}

        total = task.get("total_questions", 0)
        completed = len(task.get("completed_indices", []))
        failed = len(task.get("failed_indices", []))

        return {
            "task_id": task_id,
            "status": task.get("status"),
            "total": total,
            "completed": completed,
            "failed": failed,
            "remaining": total - completed - failed,
            "progress_pct": round(completed / total * 100, 1) if total else 0,
            "rag_mode": task.get("rag_mode"),
            "question_type": task.get("question_type"),
        }

    # ─── 前置校验（公共方法） ──────────────────────────────

    def _check_task(self, task_id: str, *, allow_created: bool = False,
                    allow_completed: bool = False, need_failed: bool = False) -> Dict[str, Any]:
        """校验任务状态，返回task dict。不符合条件抛ValueError。"""
        task = _load_task(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        if task["status"] in ("running", "scoring"):
            raise ValueError(f"任务 {task_id} 正在运行中")
        if not allow_created and task["status"] == "created":
            raise ValueError(f"任务 {task_id} 尚未启动，请使用 start")
        if not allow_completed and task["status"] == "completed":
            raise ValueError(f"任务 {task_id} 已完成，请使用 restart 重新评估")
        if need_failed and not task.get("failed_indices"):
            raise ValueError(f"任务 {task_id} 无失败条目可重试")
        return task

    # ─── 启动评估（首次） ──────────────────────────────────

    def start(self, task_id: str):
        """首次启动评估任务（后台线程）。"""
        self._check_task(task_id, allow_created=True)
        self._run_in_thread(task_id, resume=False)

    # ─── 断点续评 ──────────────────────────────────────────

    def resume(self, task_id: str):
        """从断点继续评估（后台线程）。仅允许paused/failed状态。"""
        task = _load_task(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        if task["status"] in ("running", "scoring"):
            raise ValueError(f"任务 {task_id} 正在运行中")
        if task["status"] == "completed":
            raise ValueError(f"任务 {task_id} 已完成，请使用 restart 重新评估")
        if task["status"] == "created":
            raise ValueError(f"任务 {task_id} 尚未启动，请使用 start")
        self._run_in_thread(task_id, resume=True)

    # ─── 重新评估 ──────────────────────────────────────────

    def restart(self, task_id: str):
        """重新评估：清空该任务所有缓存文件，从头开始。"""
        task = self._check_task(task_id, allow_completed=True, allow_created=True)
        # 先保存原始配置
        saved_config = {
            "rag_mode": task.get("rag_mode"),
            "sample_count": task.get("sample_count"),
            "question_type": task.get("question_type"),
        }
        _clear_task_files(task_id)
        # 重建任务元数据（保留配置）
        task = {"task_id": task_id}
        task.update(saved_config)
        task.update({
            "status": "created",
            "total_questions": 0,
            "completed_indices": [],
            "failed_indices": [],
            "started_at": None,
            "finished_at": None,
            "error": None,
            "updated_at": datetime.now().isoformat(),
        })
        _save_task(task)
        self._run_in_thread(task_id, resume=False)

    # ─── 重试失败条目 ──────────────────────────────────────

    def retry_failed(self, task_id: str):
        """仅重试之前失败的题目（后台线程）。"""
        self._check_task(task_id, allow_completed=True, need_failed=True)
        self._run_in_thread(task_id, resume=True, retry_failed_only=True)

    # ─── 强制停止 ──────────────────────────────────────────

    def force_stop(self, task_id: str):
        """强制停止运行中的任务，将状态改为paused。"""
        task = _load_task(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        if task["status"] not in ("running", "scoring"):
            raise ValueError(f"任务 {task_id} 当前状态为 {task['status']}，无需停止")
        with file_lock:
            task["status"] = "paused"
            task["error"] = "手动停止"
            task["updated_at"] = datetime.now().isoformat()
            _save_task(task)
        self._running_tasks.pop(task_id, None)

    # ─── 删除任务 ──────────────────────────────────────────

    def delete(self, task_id: str):
        """删除任务及其所有文件。运行中的任务需先停止。"""
        task = _load_task(task_id)
        if not task:
            raise ValueError(f"任务 {task_id} 不存在")
        if task["status"] in ("running", "scoring"):
            raise ValueError(f"任务 {task_id} 正在运行中，请先停止")
        _clear_task_files(task_id)
        self._running_tasks.pop(task_id, None)

    # ─── 获取报告 ──────────────────────────────────────────

    def get_report(self, task_id: str) -> Dict[str, Any]:
        """获取最终评估报告。

        优先读取已保存的报告文件（包含ragas_full等完整数据），
        仅在无报告文件时基于已有结果实时计算。
        """
        task = _load_task(task_id)
        if not task:
            return {"error": f"任务 {task_id} 不存在"}

        # 优先读取已保存的完整报告
        saved = _load_report(task_id)
        if saved:
            return saved

        results = _load_results(task_id)
        if not results:
            return {
                "task_id": task_id,
                "status": task.get("status"),
                "message": "暂无评估结果",
            }

        # 分离成功和失败
        success_results = [r for r in results if r.get("status") == "success"]
        failed_results = [r for r in results if r.get("status") == "failed"]

        # 计算各项平均分
        report = self._compute_report(task, success_results, failed_results)

        # 保存报告
        _save_report(task_id, report)
        return report

    # ─── 内部辅助方法 ──────────────────────────────────────

    @staticmethod
    def _mark_completed(task_id: str):
        """标记任务完成并保存。"""
        task = _load_task(task_id)
        if task:
            task.update({"status": "completed",
                         "finished_at": datetime.now().isoformat(),
                         "updated_at": datetime.now().isoformat()})
            _save_task(task)
            print(f"[PersistentEval] 任务 {task_id} 完成")

    @staticmethod
    def _update_task_progress(task_id: str, idx: int, success: bool):
        """更新任务元数据中的已完成/失败索引（线程安全）。"""
        with file_lock:
            task = _load_task(task_id)
            if not task:
                return
            if success:
                if idx not in task.get("completed_indices", []):
                    task.setdefault("completed_indices", []).append(idx)
                task["failed_indices"] = [i for i in task.get("failed_indices", []) if i != idx]
            else:
                if idx not in task.get("failed_indices", []):
                    task.setdefault("failed_indices", []).append(idx)
            task["updated_at"] = datetime.now().isoformat()
            _save_task(task)

    # ─── 内部核心执行逻辑 ──────────────────────────────────

    def _run_in_thread(self, task_id: str, resume: bool = False, retry_failed_only: bool = False):
        """在后台线程中运行评估。"""
        thread = threading.Thread(
            target=self._execute_task,
            args=(task_id, resume, retry_failed_only),
            daemon=True,
        )
        self._running_tasks[task_id] = thread
        thread.start()

    # 任务整体超时（秒），超时后标记为failed
    _TASK_TIMEOUT = int(os.getenv("EVAL_TASK_TIMEOUT", "1800"))  # 默认30分钟

    def _execute_task(self, task_id: str, resume: bool, retry_failed_only: bool):
        """评估任务执行主逻辑（在后台线程中运行，带整体超时保护）。"""
        from app.evaluation.eval_runner import EvalRunner
        from app.evaluation.eval_report import EvalReport
        from app.evaluation.eval_dataset import EVAL_DATASET
        from app.config import settings

        def _timeout_watcher():
            """超时监控：超过_TASK_TIMEOUT后强制标记任务失败。"""
            time.sleep(self._TASK_TIMEOUT)
            task = _load_task(task_id)
            if task and task["status"] in ("running", "scoring"):
                task.update({"status": "failed", "error": f"任务超时({self._TASK_TIMEOUT}s)",
                             "updated_at": datetime.now().isoformat()})
                with file_lock:
                    _save_task(task)
                print(f"[PersistentEval] 任务 {task_id} 超时({self._TASK_TIMEOUT}s)，已标记失败")

        watcher = threading.Thread(target=_timeout_watcher, daemon=True)
        watcher.start()

        task = _load_task(task_id)
        if not task:
            return

        # ── 准备数据集 ──
        dataset = EVAL_DATASET
        question_type = task.get("question_type")
        if question_type:
            dataset = [d for d in dataset if d.get("question_type") == question_type]
        sample_count = task.get("sample_count")
        if sample_count:
            dataset = dataset[:sample_count]

        total = len(dataset)
        now = datetime.now().isoformat()
        task["total_questions"] = total
        task["status"] = "running"
        task["started_at"] = task.get("started_at") or now
        task["updated_at"] = now
        _save_task(task)

        # ── 确定需要评估的题目索引 ──
        if retry_failed_only:
            indices_to_eval = set(task.get("failed_indices", []))
        elif resume:
            done = set(task.get("completed_indices", []))
            indices_to_eval = set(range(total)) - done
        else:
            indices_to_eval = set(range(total))

        if not indices_to_eval:
            # 所有题目已评估完，但仍需运行RAGAS评分（如果还没跑过）
            all_results = _load_results(task_id)
            success_results = [r for r in all_results if r.get("status") == "success"]
            if success_results and not any("retrieval" in r or "response" in r for r in success_results):
                print(f"[PersistentEval] 所有题目已完成，但RAGAS评分未运行，开始评分...")
                eval_runner = EvalRunner(self.rag_engine)
                try:
                    full_report = self._run_ragas_scoring(task_id, success_results, eval_runner)
                    report = self._compute_report(task, success_results, [r for r in all_results if r.get("status") == "failed"])
                    if full_report:
                        report["ragas_full"] = full_report
                        try:
                            EvalReport().save_result(task.get("rag_mode") or "unknown", full_report)
                        except Exception:
                            pass
                    _save_report(task_id, report)
                except Exception as e:
                    print(f"[PersistentEval] RAGAS评分失败: {e}")
            self._mark_completed(task_id)
            self._running_tasks.pop(task_id, None)
            return

        # ── 初始化RAG引擎 ──
        original_mode = settings.RAG_MODE
        rag_mode = task.get("rag_mode")
        if rag_mode:
            settings.RAG_MODE = rag_mode
            from app.core.rag_engine import RAGEngine
            self.rag_engine = RAGEngine()
            self.rag_engine.initialize()

        # ── 并发评估 ──
        eval_runner = EvalRunner(self.rag_engine)
        max_workers = settings.EVAL_MAX_CONCURRENT
        sorted_indices = sorted(indices_to_eval)
        eval_items = [(idx, dataset[idx], eval_runner) for idx in sorted_indices]

        print(f"[PersistentEval] Evaluating {len(eval_items)} items (concurrent={max_workers})...")

        try:
            def _eval_one(args):
                idx, item, er = args
                return idx, self._evaluate_single(idx, item, er)

            results = parallel_map(_eval_one, eval_items, max_workers=max_workers, desc="Eval")

            # 按序保存结果
            for result in results:
                if result is None or isinstance(result, Exception):
                    continue
                idx, record = result
                _append_result(task_id, record)
                self._update_task_progress(task_id, idx, record["status"] == "success")

        except Exception as e:
            task = _load_task(task_id)
            task.update({"status": "failed", "error": str(e),
                         "updated_at": datetime.now().isoformat()})
            _save_task(task)
            print(f"[PersistentEval] 任务 {task_id} 整体失败: {e}")
            return

        finally:
            if rag_mode:
                settings.RAG_MODE = original_mode

        # ── 全部完成 → 运行RAGAS批量评分 + 生成最终报告 ──
        all_results = _load_results(task_id)
        success_results = [r for r in all_results if r.get("status") == "success"]
        failed_results = [r for r in all_results if r.get("status") == "failed"]
        full_report = None

        if success_results:
            # 更新任务状态为scoring，让前端知道正在评分
            with file_lock:
                task = _load_task(task_id)
                if task:
                    task["status"] = "scoring"
                    task["updated_at"] = datetime.now().isoformat()
                    _save_task(task)
            print(f"[PersistentEval] 答案生成完成({len(success_results)}/{total}成功)，运行RAGAS批量评分...")
            try:
                full_report = self._run_ragas_scoring(task_id, success_results, eval_runner)
            except Exception as e:
                print(f"[PersistentEval] RAGAS评分失败: {e}")

        # 生成汇总报告
        task = _load_task(task_id)
        report = self._compute_report(task, success_results, failed_results)
        if full_report:
            report["ragas_full"] = full_report
            # 兼容原有EvalReport格式
            try:
                EvalReport().save_result(task.get("rag_mode") or "unknown", full_report)
            except Exception:
                pass
        _save_report(task_id, report)

        self._mark_completed(task_id)
        self._running_tasks.pop(task_id, None)

    def _evaluate_single(
        self, idx: int, item: dict, eval_runner
    ) -> Dict[str, Any]:
        """评估单题，返回结果记录。异常不抛出，标记为失败。"""
        from app.models.schemas import ChatRequest

        record = {
            "index": idx,
            "question": item["question"],
            "question_type": item.get("question_type", "retrieve"),
            "law": item.get("law", ""),
            "ground_truth": item["ground_truth"],
            "relevant_articles": item.get("relevant_articles", []),
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "answer": None, "contexts": None, "sources": None, "rag_mode": None,
            "error": None,
        }

        try:
            response = eval_runner.rag_engine.chat(
                ChatRequest(question=item["question"])
            )
            record["answer"] = response.answer
            record["contexts"] = [src.content for src in response.sources]
            record["sources"] = [{"source": s.source, "content": s.content} for s in response.sources]
            record["rag_mode"] = response.rag_mode
        except Exception as e:
            record["status"] = "failed"
            record["error"] = f"答案生成失败: {str(e)[:200]}"
            print(f"[PersistentEval] 题目 {idx} 答案生成失败: {e}")

        return record

    def _run_ragas_scoring(
        self,
        task_id: str,
        success_results: List[Dict[str, Any]],
        eval_runner,
    ) -> Dict[str, Any]:
        """对已完成的全部题目运行RAGAS评分 + 检索指标 + 响应指标。"""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from app.evaluation.retrieval_metrics import (
            compute_retrieval_metrics, compute_retrieval_metrics_by_type,
        )
        from app.evaluation.response_metrics import (
            compute_response_metrics, compute_response_metrics_by_type,
        )
        from app.evaluation.utils import strip_per_query
        from app.evaluation.eval_runner import _mean_scores

        questions = [r["question"] for r in success_results]
        answers = [r["answer"] for r in success_results]
        contexts = [r["contexts"] for r in success_results]
        ground_truths = [r["ground_truth"] for r in success_results]
        sources_raw = [r["sources"] for r in success_results]
        relevant_articles_list = [r["relevant_articles"] for r in success_results]
        question_types = [r["question_type"] for r in success_results]

        # ── 阶段2：检索指标 ──
        print("[PersistentEval] Phase 2: 检索指标...")
        retrieval = compute_retrieval_metrics(sources_raw, relevant_articles_list, k=5)
        type_retrieval = compute_retrieval_metrics_by_type(
            sources_raw, relevant_articles_list, question_types, k=5
        )

        # ── 阶段3：RAGAS评分 ──
        print("[PersistentEval] Phase 3: RAGAS评分...")
        ds = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

        scores, type_scores, faithfulness_per_query = {}, {}, []
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        for attempt in range(1, eval_runner.MAX_RETRIES + 1):
            try:
                result = evaluate(ds, metrics=metrics,
                                  llm=eval_runner.ragas_llm, embeddings=eval_runner.ragas_embeddings,
                                  batch_size=3)
                result_df = result.to_pandas()
                scores = _mean_scores(result_df)

                # 分类型统计
                if any(qt != "retrieve" for qt in question_types):
                    result_df["question_type"] = question_types
                    for qtype in set(question_types):
                        type_scores[qtype] = _mean_scores(result_df[result_df["question_type"] == qtype])

                # 提取每题faithfulness
                if "faithfulness" in result_df.columns:
                    for val in result_df["faithfulness"]:
                        try:
                            faithfulness_per_query.append(float(val))
                        except (TypeError, ValueError):
                            faithfulness_per_query.append(None)
                break
            except Exception as e:
                print(f"[PersistentEval] RAGAS attempt {attempt} failed: {e}")
                if attempt < eval_runner.MAX_RETRIES:
                    time.sleep(attempt * 30)
                else:
                    print("[PersistentEval] RAGAS全部重试失败")

        # ── 响应指标 ──
        print("[PersistentEval] Phase 3b: 响应指标...")
        response_metrics = compute_response_metrics(
            answers, ground_truths, faithfulness_scores=faithfulness_per_query, llm=eval_runner.ragas_llm,
        )
        type_response = compute_response_metrics_by_type(
            answers, ground_truths, question_types,
            faithfulness_scores=faithfulness_per_query, llm=eval_runner.ragas_llm,
        )

        # ── 回写每题指标到结果jsonl ──
        all_results = _load_results(task_id)
        per_query_retrieval = retrieval.get("per_query", [])
        per_query_response = response_metrics.get("per_query", [])
        for i, r in enumerate(all_results):
            if r.get("status") != "success":
                continue
            if i < len(per_query_retrieval):
                pq = per_query_retrieval[i]
                r["retrieval"] = {k: v for k, v in pq.items() if k not in ("retrieved_ids", "relevant_ids")}
            if i < len(per_query_response):
                r["response"] = per_query_response[i]

        with open(_results_path(task_id), "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(sanitize_floats(r), ensure_ascii=False) + "\n")

        # ── 构建输出 ──
        if not scores:
            print("[PersistentEval] ⚠️ RAGAS评分全部失败，报告中RAGAS指标不可用")
        triad = {
            "context_relevancy": scores.get("context_recall") or 0.0,
            "faithfulness": scores.get("faithfulness") or 0.0,
            "answer_relevancy": scores.get("answer_relevancy") or 0.0,
        }
        details = [
            {"question": r.get("question", ""), "question_type": r.get("question_type", "retrieve"),
             "law": r.get("law", ""), "answer": r.get("answer", ""),
             "ground_truth": r.get("ground_truth", ""),
             "relevant_articles": r.get("relevant_articles", []),
             "source_count": len(r.get("sources", [])), "rag_mode": r.get("rag_mode", ""),
             **({"retrieval": r["retrieval"]} if "retrieval" in r else {}),
             **({"response": r["response"]} if "response" in r else {})}
            for r in all_results
        ]

        return {
            "triad": triad,
            "retrieval": strip_per_query(retrieval),
            "response": strip_per_query(response_metrics),
            "scores": scores, "type_scores": type_scores,
            "type_retrieval": {k: strip_per_query(v) for k, v in type_retrieval.items()},
            "type_response": {k: strip_per_query(v) for k, v in type_response.items()},
            "rag_mode": _load_task(task_id).get("rag_mode") or "unknown",
            "sample_count": len(success_results),
            "details": details,
        }

    @staticmethod
    def _avg_nested_key(results: List[Dict], outer: str, keys: List[str]) -> Dict[str, float]:
        """从结果列表中提取嵌套key的均值，如 r['retrieval']['precision@5']。"""
        avg = {}
        for key in keys:
            values = [r.get(outer, {}).get(key) for r in results if r.get(outer, {}).get(key) is not None]
            if values:
                avg[key] = round(sum(values) / len(values), 4)
        return avg

    def _compute_report(
        self,
        task: Dict[str, Any],
        success_results: List[Dict[str, Any]],
        failed_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """基于已有结果计算汇总报告。"""
        total = task.get("total_questions", len(success_results) + len(failed_results))
        success_count = len(success_results)
        failed_count = len(failed_results)

        report = {
            "task_id": task["task_id"],
            "status": task.get("status"),
            "rag_mode": task.get("rag_mode"),
            "question_type": task.get("question_type"),
            "summary": {
                "total": total, "success": success_count, "failed": failed_count,
                "remaining": total - success_count - failed_count,
                "success_rate": round(success_count / total * 100, 1) if total else 0,
            },
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "finished_at": task.get("finished_at"),
        }

        if success_results:
            retrieval_avg = self._avg_nested_key(success_results, "retrieval",
                                                   ["precision@5", "recall@5", "f1@5", "mrr", "map"])
            response_avg = self._avg_nested_key(success_results, "response",
                                                   ["rouge_l", "bleu", "hallucination_rate", "completeness"])
            if retrieval_avg:
                report["retrieval_avg"] = retrieval_avg
            if response_avg:
                report["response_avg"] = response_avg

        if failed_results:
            report["failed_items"] = [
                {"index": r.get("index"), "question": r.get("question", "")[:50], "error": r.get("error", "")}
                for r in failed_results
            ]

        return report
