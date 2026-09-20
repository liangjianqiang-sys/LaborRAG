"""评估与断点续评路由。"""
import traceback
import threading

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import verify_bearer
from app.api import deps
from app.evaluation.utils import sanitize_floats

router = APIRouter(dependencies=[Depends(verify_bearer)])

# ── 普通评估任务状态（run/status/report 等）──
_eval_status = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}


def _run_eval_task(sample_count, rag_mode, question_type, eval_subset, difficulty):
    """后台线程执行评估。"""
    from app.evaluation.runner import EvalRunner
    from app.evaluation.report import EvalReport

    _eval_status["running"] = True
    _eval_status["progress"] = "评估进行中..."
    _eval_status["result"] = None
    _eval_status["error"] = None

    try:
        result = EvalRunner(deps.engine).run(sample_count=sample_count, rag_mode=rag_mode,
                                             question_type=question_type, eval_subset=eval_subset,
                                             difficulty=difficulty)
        filepath = EvalReport().save_result(result["rag_mode"], result)
        _eval_status["result"] = {"message": "评估完成", "report_path": filepath, **result}
        _eval_status["progress"] = "评估完成"
    except Exception as e:
        traceback.print_exc()
        _eval_status["error"] = str(e)
        _eval_status["progress"] = f"评估失败: {str(e)}"
    finally:
        _eval_status["running"] = False


@router.post("/evaluation/run")
async def run_evaluation(sample_count: int = None, rag_mode: str = None, question_type: str = None,
                         eval_subset: str = None, difficulty: str = None):
    """启动 RAGAS 评估（后台异步执行）。

    Args:
        eval_subset: 子集过滤(smoke=5题快速/full=20题全量)
        difficulty: 难度过滤(easy/medium/hard)
    """
    deps.require_engine()
    if _eval_status["running"]:
        raise HTTPException(status_code=409, detail="评估正在进行中")

    thread = threading.Thread(
        target=_run_eval_task,
        args=(sample_count, rag_mode, question_type, eval_subset, difficulty),
        daemon=True,
    )
    thread.start()
    return {"message": "评估已启动", "status": "running"}


@router.get("/evaluation/status")
async def get_evaluation_status():
    """查询评估任务状态。"""
    return sanitize_floats(_eval_status)


@router.get("/evaluation/report")
async def get_evaluation_report():
    """获取评估对比报告（三维度：三元组 + 检索 + 响应）。"""
    from app.evaluation.report import EvalReport
    return sanitize_floats(EvalReport().compare())


@router.get("/evaluation/observability")
async def get_observability():
    """可观测性三维度概览（检索质量 / 生成质量 / 业务指标）。"""
    from app.evaluation.report import EvalReport
    return sanitize_floats(EvalReport().get_observability())


@router.get("/evaluation/bad-cases")
async def get_bad_cases(top_n: int = 5):
    """获取得分最低的 Bad Case 列表。"""
    from app.evaluation.report import EvalReport
    return sanitize_floats(EvalReport().get_bad_cases(top_n=top_n))


# ── 断点续评任务管理 ──
_persistent_manager = None


def _get_persistent_manager():
    """懒初始化持久化评估管理器。"""
    global _persistent_manager
    if _persistent_manager is None:
        from app.evaluation.persistent import PersistentEvalManager
        _persistent_manager = PersistentEvalManager(deps.engine)
    _persistent_manager.rag_engine = deps.engine
    return _persistent_manager


def _run_persistent_action(action, task_id: str, msg: str):
    """执行持久化评估动作，统一校验 + 异常处理。"""
    deps.require_engine()
    try:
        action(task_id)
        return {"task_id": task_id, "message": msg, "status": "running"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluation/persistent/create")
async def create_persistent_task(
    rag_mode: str = None,
    sample_count: int = None,
    question_type: str = None,
    sample_offset: int = None,
    eval_subset: str = None,
    difficulty: str = None,
):
    """创建断点续评任务（不立即执行）。"""
    deps.require_engine()
    task_id = _get_persistent_manager().create_task(
        rag_mode=rag_mode, sample_count=sample_count, question_type=question_type,
        sample_offset=sample_offset, eval_subset=eval_subset, difficulty=difficulty,
    )
    return {"task_id": task_id, "message": "任务已创建"}


@router.post("/evaluation/persistent/start")
async def start_persistent_task(task_id: str):
    """首次启动评估任务（后台异步）。"""
    return _run_persistent_action(_get_persistent_manager().start, task_id, "评估已启动")


@router.post("/evaluation/persistent/resume")
async def resume_persistent_task(task_id: str):
    """断点续评：从上次中断处继续。"""
    return _run_persistent_action(_get_persistent_manager().resume, task_id, "续评已启动")


@router.post("/evaluation/persistent/restart")
async def restart_persistent_task(task_id: str):
    """重新评估：清空缓存，从头开始。"""
    return _run_persistent_action(_get_persistent_manager().restart, task_id, "重新评估已启动")


@router.post("/evaluation/persistent/retry-failed")
async def retry_failed_persistent_task(task_id: str):
    """仅重试之前失败的题目。"""
    return _run_persistent_action(_get_persistent_manager().retry_failed, task_id, "失败条目重试已启动")


@router.post("/evaluation/persistent/force-stop")
async def force_stop_persistent_task(task_id: str):
    """强制停止运行中的任务。"""
    try:
        _get_persistent_manager().force_stop(task_id)
        return {"task_id": task_id, "message": "任务已停止", "status": "paused"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/evaluation/persistent/delete")
async def delete_persistent_task(task_id: str):
    """删除任务及其所有文件。"""
    try:
        _get_persistent_manager().delete(task_id)
        return {"task_id": task_id, "message": "任务已删除"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/evaluation/persistent/tasks")
async def list_persistent_tasks():
    """列出所有断点续评任务。"""
    return {"tasks": sanitize_floats(_get_persistent_manager().list_tasks())}


@router.get("/evaluation/persistent/progress")
async def get_persistent_progress(task_id: str):
    """获取任务进度。"""
    return sanitize_floats(_get_persistent_manager().get_task_progress(task_id))


@router.get("/evaluation/persistent/report")
async def get_persistent_report(task_id: str):
    """获取/生成任务评估报告。"""
    return sanitize_floats(_get_persistent_manager().get_report(task_id))
