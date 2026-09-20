"""Bearer Token 鉴权。

设计取舍：
- `AUTH_SECRET` 为空时不启用鉴权，本地开发和单测零配置即可跑通；
  配置后除 `/health` 外的所有端点都要求 `Authorization: Bearer <AUTH_SECRET>`。
- 用 `auto_error=False` 自行抛 401，而不是交给 HTTPBearer 默认的 403，
  这样错误码语义正确（缺凭证是「未认证」而非「无权限」），
  且响应体走统一的 `{"detail": ...}` 格式。
"""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

_bearer = HTTPBearer(
    description="Bearer Token，取值等于后端 .env 中的 AUTH_SECRET",
    auto_error=False,
)


def verify_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """校验 Bearer Token；未配置 AUTH_SECRET 时直接放行。"""
    if not settings.AUTH_SECRET:
        return
    if not credentials or credentials.credentials != settings.AUTH_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未授权：请提供有效的访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
