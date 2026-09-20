"""知识库与文档管理路由。"""
import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.auth import verify_bearer
from app.api import deps
from app.core.config import settings
from app.schemas.chat import KnowledgeBaseStatus, UploadResponse, BuildRequest

router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/knowledge-base/status", response_model=KnowledgeBaseStatus)
async def knowledge_base_status():
    """获取知识库状态。"""
    total_chunks = 0
    vector_store_exists = False

    if deps.engine and deps.engine.vector_store_manager.is_ready:
        vector_store_exists = True
        try:
            total_chunks = deps.engine.vector_store_manager.vector_store.index.ntotal
        except Exception:
            total_chunks = 0

    # 统计数据目录中的文件
    total_documents = 0
    if os.path.exists(settings.DATA_DIR):
        total_documents = sum(
            1 for f in os.listdir(settings.DATA_DIR)
            if os.path.splitext(f)[1].lower() in deps.SUPPORTED_EXT
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
        success = deps.engine.build_knowledge_base(rebuild=rebuild)
        if not success:
            raise HTTPException(
                status_code=400,
                detail="构建失败：数据目录中没有找到支持的文档文件。",
            )
        return {"message": "知识库构建成功", "rebuild": rebuild}
    except HTTPException:
        raise
    # 其余异常冒泡到 main.py 全局 handler


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档到数据目录并自动加入向量库。"""
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in deps.SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {deps.SUPPORTED_EXT}",
        )

    # 保存文件到数据目录
    filepath = os.path.join(settings.DATA_DIR, file.filename)
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # 添加到向量库（异常冒泡到 main.py 全局 handler）
    chunk_count = deps.engine.add_document(filepath)
    return UploadResponse(
        success=True,
        filename=file.filename,
        chunk_count=chunk_count,
        message=f"文档上传成功，切分为 {chunk_count} 个片段。",
    )


@router.get("/documents/list")
async def list_documents():
    """列出数据目录中的所有文档。"""
    documents = []

    if os.path.exists(settings.DATA_DIR):
        for filename in os.listdir(settings.DATA_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext in deps.SUPPORTED_EXT:
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
