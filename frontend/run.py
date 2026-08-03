"""LaborRAG 前端启动入口

启动方式（复制到终端执行）：
    npm --prefix d:\毕业设计\LaborRAG\frontend run dev

（不需要 conda 环境，有 Node.js 就行）
"""
import subprocess
import sys
import os

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(FRONTEND_DIR)

print("启动前端服务...")
subprocess.run(["npm", "run", "dev"])
