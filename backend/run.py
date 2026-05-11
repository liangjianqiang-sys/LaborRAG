"""LaborRAG 启动入口 - 直接运行此文件即可启动后端服务"""
import sys
import os

# 确保backend目录在Python路径中
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

# 切换工作目录到backend，确保.env能被正确加载
os.chdir(BACKEND_DIR)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[BACKEND_DIR])
