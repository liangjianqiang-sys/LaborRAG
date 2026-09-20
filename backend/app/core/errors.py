"""应用级异常类型与故障分类。

为什么要把 LLM 故障单独分类
--------------------------
LLM 的**额度耗尽 / 鉴权失败 / 网络不通 / 上游 5xx** 属于运维性故障，不是代码缺陷。
若让它们和其他异常一样冒泡成 500「内部服务器错误」，运维者拿到的信息既无法定位
也无法行动 —— 而这类问题恰恰是线上最常见的一类（本项目实测就撞到过
`Free quota exhausted`：免费额度用完，表现为 chat 接口 500）。

分类之后，上层可以返回 503 + 可操作的提示（检查额度 / 密钥 / 网络），
而不是笼统的 500。

**本模块必须零依赖**：`app/main.py` 需要在模块级 import 它来注册异常处理器，
而 main 刻意不在模块级导入 `app.services.rag_engine`（那会加载 2GB 嵌入模型）。
"""
# LLM 运维性故障的异常类名。
#
# 按**类名**而非 isinstance 判断：openai / langchain 是可选依赖，而本模块要保持零依赖。
# 只收「明确属于 LLM 侧」的错误，刻意不含 BadRequestError 之类 ——
# 那通常是我们自己的 prompt/参数写错，应当照常暴露为 500，不能被掩盖。
_LLM_ERROR_NAMES = frozenset({
    "PermissionDeniedError",     # 403，常见于免费额度耗尽
    "AuthenticationError",       # 401，API Key 无效
    "RateLimitError",            # 429，限流
    "APIConnectionError",        # 连不上（网络 / 代理 / DNS）
    "APITimeoutError",           # 超时
    "InternalServerError",       # 上游 5xx
    "ServiceUnavailableError",
    "BadGatewayError",
})


class LLMUnavailableError(RuntimeError):
    """LLM 服务不可用（额度耗尽 / 鉴权失败 / 网络不通 / 上游故障）。

    与「代码缺陷」区分开：这类故障可重试、可通过配置修复，
    因此映射为 503 而非 500。
    """


def is_llm_unavailable(exc: BaseException) -> bool:
    """判断异常链里是否出现了 LLM 运维性故障。

    沿 `__cause__` / `__context__` 走完整条链 —— 真实故障往往被
    langchain 包了好几层，只看最外层类型会漏判。
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _LLM_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def llm_unavailable_message(exc: BaseException) -> str:
    """生成面向使用者的、可操作的提示（不含上游长文本，避免刷屏）。"""
    return (
        f"LLM 服务暂不可用（额度耗尽 / 鉴权失败 / 网络不通）：{type(exc).__name__}。"
        "请检查后端 .env 中的 LLM_API_KEY 与账户额度后重试。"
    )
