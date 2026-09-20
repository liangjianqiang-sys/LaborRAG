"""聊天路由。"""
from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import verify_bearer
from app.api import deps
from app.schemas.chat import ChatRequest, ChatResponse

# 鉴权挂在 router 上：新增端点自动继承，不会漏挂。
# AUTH_SECRET 为空时 verify_bearer 直接放行，本地开发无感。
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能问答接口。

    异步处理，与 FastAPI 事件循环对齐，避免线程池开销。
    """
    if not deps.engine or not deps.engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪，请先构建知识库。")
    # 异常冒泡到 main.py 全局 handler，统一 500 格式 + request_id
    return await deps.engine.chat(request)
