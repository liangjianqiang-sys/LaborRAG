"""API 路由聚合器：组合 chat / knowledge / evaluation 三个子路由。

main.py 依赖这里的 `router` 与 `set_engine`：
- `router`：合并后的根路由（main.py 以 /api/v1 前缀挂载）
- `set_engine`：来自 deps，由 lifespan 调用注入引擎
"""
from fastapi import APIRouter

from app.api.deps import set_engine  # noqa: F401  # 供 main.py 导入
from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.api.evaluation import router as evaluation_router

router = APIRouter()
router.include_router(chat_router)
router.include_router(knowledge_router)
router.include_router(evaluation_router)
