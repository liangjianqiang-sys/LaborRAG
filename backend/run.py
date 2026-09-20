"""LaborRAG 后端启动入口

启动方式（复制到独立 PowerShell 终端执行）：
    D:\Anaconda\envs\RAG\python.exe d:\毕业设计\LaborRAG\backend\run.py

环境：Conda RAG 环境（已装全部依赖）
"""
import sys
import os

# 确保backend目录在Python路径中
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# 切换工作目录到backend，确保.env能被正确加载
os.chdir(BACKEND_DIR)


if __name__ == "__main__":
    # 统一日志配置
    from app.core.logging import setup_logging
    setup_logging()

    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
