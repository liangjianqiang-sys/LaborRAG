import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.config import settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeBaseStatus,
    UploadResponse,
    BuildRequest,
)
from app.core.rag_engine import RAGEngine

router = APIRouter()

# RAG引擎实例（由main.py注入）
engine: RAGEngine = None


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
        import traceback
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
        supported_ext = {".pdf", ".docx", ".txt", ".md"}
        total_documents = sum(
            1 for f in os.listdir(settings.DATA_DIR)
            if os.path.splitext(f)[1].lower() in supported_ext
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
    supported_ext = {".pdf", ".docx", ".txt", ".md"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in supported_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {supported_ext}",
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
    supported_ext = {".pdf", ".docx", ".txt", ".md"}
    documents = []

    if os.path.exists(settings.DATA_DIR):
        for filename in os.listdir(settings.DATA_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_ext:
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

@router.post("/evaluation/run")
async def run_evaluation(background_tasks: BackgroundTasks, sample_count: int = None):
    """运行RAGAS评估（后台执行）。"""
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪")

    from app.evaluation.eval_runner import EvalRunner
    from app.evaluation.eval_report import EvalReport

    runner = EvalRunner(engine)
    report = EvalReport()

    # 同步运行（评估本身需要时间）
    try:
        result = runner.run(sample_count=sample_count)
        filepath = report.save_result(settings.RETRIEVER_TYPE, result)
        return {
            "message": "评估完成",
            "retriever_type": settings.RETRIEVER_TYPE,
            "scores": result["scores"],
            "sample_count": result["sample_count"],
            "report_path": filepath,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"评估失败: {str(e)}")


@router.get("/evaluation/report")
async def get_evaluation_report():
    """获取评估对比报告。"""
    from app.evaluation.eval_report import EvalReport
    report = EvalReport()
    comparison = report.compare()
    return comparison
