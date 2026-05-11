"""LaborRAG 前端启动入口 - 直接运行此文件即可启动前端服务"""
import subprocess
import sys
import os

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(FRONTEND_DIR)

print("启动前端服务...")
subprocess.run(["npm", "run", "dev"], shell=True)
