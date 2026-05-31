import os
import traceback
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeBaseStatus,
    UploadResponse,
    BuildRequest,
)
from app.core.rag_engine import RAGEngine
from app.evaluation.utils import sanitize_floats

# 支持的文档扩展名
SUPPORTED_EXT = {".pdf", ".docx", ".txt", ".md"}

router = APIRouter()

# RAG引擎实例（由main.py注入）
engine: RAGEngine = None

# 评估任务状态
_eval_status = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
}


def set_engine(rag_engine: RAGEngine):
    """注入RAG引擎实例。"""
    global engine
    engine = rag_engine


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """智能问答接口。

    使用同步def而非async def，FastAPI会在线程池中运行，
    避免CRAG的同步阻塞调用卡住整个事件循环。
    """
    if not engine or not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪，请先构建知识库。",
        )
    try:
        return engine.chat(request)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
async def knowledge_base_status():
    """获取知识库状态。"""
    total_chunks = 0
    vector_store_exists = False

    if engine and engine.vector_store_manager.is_ready:
        vector_store_exists = True
        try:
            total_chunks = engine.vector_store_manager.vector_store.index.ntotal
        except Exception:
            total_chunks = 0

    # 统计数据目录中的文件
    total_documents = 0
    if os.path.exists(settings.DATA_DIR):
        total_documents = sum(
            1 for f in os.listdir(settings.DATA_DIR)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        )

    return KnowledgeBaseStatus(
        total_documents=total_documents,
        total_chunks=total_chunks,
        vector_store_exists=vector_store_exists,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        llm_model=settings.LLM_MODEL_NAME,
    )


@router.post("/knowledge-base/build")
def build_knowledge_base(request: BuildRequest = None):
    """构建/重建知识库。"""
    rebuild = request.rebuild if request else False
    try:
        success = engine.build_knowledge_base(rebuild=rebuild)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="构建失败：数据目录中没有找到支持的文档文件。",
            )
        return {"message": "知识库构建成功", "rebuild": rebuild}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档到数据目录并自动加入向量库。"""
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {SUPPORTED_EXT}",
        )

    # 保存文件到数据目录
    filepath = os.path.join(settings.DATA_DIR, file.filename)
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # 添加到向量库
    try:
        chunk_count = engine.add_document(filepath)
        return UploadResponse(
            success=True,
            filename=file.filename,
            chunk_count=chunk_count,
            message=f"文档上传成功，切分为 {chunk_count} 个片段。",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/list")
async def list_documents():
    """列出数据目录中的所有文档。"""
    documents = []

    if os.path.exists(settings.DATA_DIR):
        for filename in os.listdir(settings.DATA_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXT:
                filepath = os.path.join(settings.DATA_DIR, filename)
                documents.append({
                    "filename": filename,
                    "size": os.path.getsize(filepath),
                    "type": ext.lstrip("."),
                })

    return {"documents": documents, "total": len(documents)}


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    """删除指定文档。"""
    filepath = os.path.join(settings.DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")

    os.remove(filepath)
    return {"message": f"已删除 {filename}，请重建知识库以更新向量库。"}


# ==================== V2 评估接口 ====================

def _run_eval_task(sample_count, rag_mode, question_type):
    """后台线程执行评估。"""
    from app.evaluation.eval_runner import EvalRunner
    from app.evaluation.eval_report import EvalReport

    _eval_status["running"] = True
    _eval_status["progress"] = "评估进行中..."
    _eval_status["result"] = None
    _eval_status["error"] = None

    try:
        result = EvalRunner(engine).run(sample_count=sample_count, rag_mode=rag_mode, question_type=question_type)
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
async def run_evaluation(sample_count: int = None, rag_mode: str = None, question_type: str = None):
    """启动RAGAS评估（后台异步执行）。"""
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪")
    if _eval_status["running"]:
        raise HTTPException(status_code=409, detail="评估正在进行中")

    thread = threading.Thread(
        target=_run_eval_task,
        args=(sample_count, rag_mode, question_type),
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
    from app.evaluation.eval_report import EvalReport
    report = EvalReport()
    return sanitize_floats(report.compare())


@router.get("/evaluation/observability")
async def get_observability():
    """可观测性三维度概览（检索质量 / 生成质量 / 业务指标）。"""
    from app.evaluation.eval_report import EvalReport
    report = EvalReport()
    return sanitize_floats(report.get_observability())


@router.get("/evaluation/bad-cases")
async def get_bad_cases(top_n: int = 5):
    """获取得分最低的Bad Case列表。"""
    from app.evaluation.eval_report import EvalReport
    report = EvalReport()
    return sanitize_floats(report.get_bad_cases(top_n=top_n))


# ==================== V3 断点续评接口 ====================

_persistent_manager = None


def _get_persistent_manager():
    """懒初始化持久化评估管理器。"""
    global _persistent_manager
    if _persistent_manager is None:
        from app.evaluation.eval_persistent import PersistentEvalManager
        _persistent_manager = PersistentEvalManager(engine)
    _persistent_manager.rag_engine = engine
    return _persistent_manager


def _require_engine():
    """校验引擎就绪，否则抛503。"""
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪")


def _run_persistent_action(action, task_id: str, msg: str):
    """执行持久化评估动作，统一校验+异常处理。"""
    _require_engine()
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
):
    """创建断点续评任务（不立即执行）。"""
    _require_engine()
    task_id = _get_persistent_manager().create_task(
        rag_mode=rag_mode, sample_count=sample_count, question_type=question_type,
        sample_offset=sample_offset,
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
