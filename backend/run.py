"""LaborRAG 启动入口 - 直接运行此文件即可启动后端服务"""
import sys
import os
import logging

# 确保backend目录在Python路径中
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# 切换工作目录到backend，确保.env能被正确加载
os.chdir(BACKEND_DIR)


class _PollingFilter(logging.Filter):
    """过滤高频轮询请求日志，减少控制台噪音。"""
    _HIDDEN_PATHS = ("/evaluation/status", "/evaluation/persistent/progress")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._HIDDEN_PATHS)


if __name__ == "__main__":
    # 为uvicorn.access logger添加过滤
    logging.getLogger("uvicorn.access").addFilter(_PollingFilter())

    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[BACKEND_DIR])
