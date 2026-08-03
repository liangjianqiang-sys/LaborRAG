"""LaborRAG 启动入口 - 直接运行此文件即可启动后端服务"""
import sys
import os

# 确保backend目录在Python路径中
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# 切换工作目录到backend，确保.env能被正确加载
os.chdir(BACKEND_DIR)


if __name__ == "__main__":
    # 统一日志配置（uvicorn --reload 子进程会在 lifespan 再 setup 一次）
    from app.core.logging_config import setup_logging
    setup_logging()

    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[BACKEND_DIR])
