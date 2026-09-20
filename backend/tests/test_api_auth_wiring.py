"""鉴权挂载的结构守卫：聚合路由下的每个端点都必须带 verify_bearer。

为什么需要它
------------
鉴权依赖是**手工挂在每个子路由**上的：

    router = APIRouter(dependencies=[Depends(verify_bearer)])

`routes.py` 拆成 chat / knowledge / evaluation 三个子路由时，靠人记得给每个
新子路由各挂一次。将来加第 4 个子路由时漏挂，就是**静默的鉴权绕过** ——
数据照常返回，没有任何报错。

而 `test_api.py::TestAuth` 打的是**具体端点**（chat / knowledge-base/status），
新增的子路由不在其中，所以它不会变红。缺的那一环正是「路由结构」本身。

所以本文件不测端点行为，而是直接读**结构**：遍历聚合器下的每条路由，
断言其 `dependencies` 里含 `verify_bearer`。

`/health` 挂在 `app` 上而非 `router` 上，因此天然不在本守卫的范围内 ——
它必须保持免鉴权（探活）。
"""
from app.api import routes as api_routes
from app.api.auth import verify_bearer


def _dependency_calls(route) -> list:
    """取出路由上挂的依赖函数。FastAPI 的 Depends 对象用 .dependency 存函数。"""
    return [getattr(d, "dependency", None) for d in getattr(route, "dependencies", [])]


def _routes_missing_auth() -> list[str]:
    missing = []
    for r in api_routes.router.routes:
        if verify_bearer not in _dependency_calls(r):
            methods = ",".join(sorted(getattr(r, "methods", []) or []))
            missing.append(f"{methods} {r.path}")
    return missing


def test_聚合路由下的每个端点都挂了鉴权():
    missing = _routes_missing_auth()
    assert not missing, (
        "以下端点没有挂 verify_bearer —— 多半是新增子路由时漏挂了，"
        "会造成**静默鉴权绕过**（AUTH_SECRET 配置后仍可匿名访问）：\n  "
        + "\n  ".join(missing)
    )


def test_守卫的判定逻辑真的能发现漏挂():
    """非空性检查：构造一个裸路由，确认判定逻辑会把它判为「缺鉴权」。

    否则上面那条测试可能因为判定逻辑恒真（比如取错了属性名）而永远变不绿。
    """
    from fastapi import APIRouter

    bare = APIRouter()

    @bare.get("/naked")
    async def naked():  # pragma: no cover - 只为构造路由对象
        return {}

    route = bare.routes[0]
    assert verify_bearer not in _dependency_calls(route), (
        "一个没挂任何依赖的路由竟被判为「已鉴权」—— 守卫的判定逻辑失效了"
    )
