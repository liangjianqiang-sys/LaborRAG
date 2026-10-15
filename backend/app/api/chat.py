"""聊天路由。"""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.auth import verify_bearer
from app.api import deps
from app.core.logging import get_logger
from app.schemas.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)

# 鉴权挂在 router 上：新增端点自动继承，不会漏挂。
# AUTH_SECRET 为空时 verify_bearer 直接放行，本地开发无感。
router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能问答接口（非流式）。

    评估链路依赖它 —— RAGAS 需要完整响应体（answer / full_contexts /
    child_contexts）才能打分，流式响应无法直接喂给评估。
    前端交互走 `/chat/stream`。

    异步处理，与 FastAPI 事件循环对齐，避免线程池开销。
    """
    if not deps.engine or not deps.engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪，请先构建知识库。")
    # 异常冒泡到 main.py 全局 handler，统一 500 格式 + request_id
    return await deps.engine.chat(request)


def _sse(event: str, data: dict) -> str:
    """把一个事件序列化成 SSE 帧。

    `ensure_ascii=False` 是必须的 —— 否则中文答案会被转义成 \\uXXXX，
    虽然客户端仍能解析，但抓包和日志里完全不可读，排查流式问题时很痛苦。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式问答接口（SSE）。

    事件类型：
        step  —— 节点进度（查询改写 / 检索法条 / 生成回答 …）
        token —— 生成节点的增量文本
        done  —— 完整 ChatResponse（含 sources / confidence / contexts）
        error —— 失败原因

    为什么不用 `response_model`：SSE 的响应体是事件流，不是单个 JSON 对象。

    为什么要显式关缓冲：反向代理（nginx 的 `proxy_buffering`）默认会攒响应，
    不关掉的话事件会被攒到最后一次性下发 —— 流式等于白做。
    """
    if not deps.engine or not deps.engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪，请先构建知识库。")

    async def event_source():
        # 流一旦开始就无法再返回 HTTP 错误码，所以这里必须自己兜住异常，
        # 以 error 事件的形式把失败原因送达客户端（否则前端只会看到连接中断）。
        try:
            async for item in deps.engine.chat_stream(request):
                yield _sse(item["event"], item["data"])
        except Exception as e:
            logger.error(f"SSE 流中断：{type(e).__name__}: {e}", exc_info=True)
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
