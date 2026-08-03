"""统一日志配置。

替代散落的 print：用 dictConfig 配 app 命名空间 logger（含时间/级别/模块），
uvicorn.access 走 PollingFilter 过滤高频轮询请求噪音。

迁移范围：仅 main.py / rag_engine.py 的启动路径 print；
评估、检索节点里的 ~79 处 print 暂留 TODO（迁移收益低、改错风险高）。
"""
import logging
import logging.config


class PollingFilter(logging.Filter):
    """过滤高频轮询请求日志（/evaluation/status 等），减少控制台噪音。

    uvicorn --reload 会 fork 子进程，子进程拿不到父进程挂的 filter，
    故需在父进程(run.py)与子进程(lifespan)各 setup_logging 一次。
    """

    _HIDDEN_PATHS = ("/evaluation/status", "/evaluation/persistent/progress")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._HIDDEN_PATHS)


_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "polling": {"()": "app.core.logging_config.PollingFilter"},
    },
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "access": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
            "filters": ["polling"],
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        # 业务日志：app.* 都走这里
        "app": {"level": "INFO", "handlers": ["console"], "propagate": False},
        # uvicorn 访问日志：带轮询过滤
        "uvicorn.access": {"level": "INFO", "handlers": ["access"], "propagate": False},
        "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
}

_configured = False


def setup_logging() -> None:
    """应用启动时调用一次，配置统一日志。幂等（reload 子进程会重新调用）。"""
    global _configured
    if _configured:
        return
    # Windows 默认用本地编码（中文系统是 GBK/cp936），日志里的 emoji(🚀/✅/👋/⏸)
    # 写入会 UnicodeEncodeError 中断该条日志。重配为 UTF-8 + errors=replace 兜底：
    # UTF-8 终端正常显示 emoji；GBK 终端也不抛错（不可编码字符替换为 ?）。
    import sys
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # 某些重定向流不支持 reconfigure，忽略
    logging.config.dictConfig(_LOGGING_CONFIG)
    # uvicorn 可能在其后重配 uvicorn.access，显式挂过滤兜底
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, PollingFilter) for f in access_logger.filters):
        access_logger.addFilter(PollingFilter())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取 app 命名空间的 logger。未配置时自动 setup 一次。"""
    if not _configured:
        setup_logging()
    if not name.startswith("app"):
        name = f"app.{name}"
    return logging.getLogger(name)
