"""LaborRAG 前端启动入口

启动方式（复制到终端执行）：
    npm --prefix d:\毕业设计\LaborRAG\frontend run dev

（不需要 conda 环境，有 Node.js 就行）
"""
import os
import shutil
import subprocess
import sys

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(FRONTEND_DIR)


def _find_npm() -> str | None:
    """解析 npm 可执行文件（走 PATHEXT，能命中 npm.cmd）。"""
    return shutil.which("npm") or shutil.which("npm.cmd")


def main() -> None:
    npm = _find_npm()
    if npm is None:
        sys.exit("未找到 npm：请先安装 Node.js（https://nodejs.org）并确认 npm 在 PATH 中。")

    print(f"启动前端服务（npm: {npm}）...")
    if os.name == "nt":
        # Windows 上 npm 实际是 npm.cmd（批处理）。subprocess 带列表参数且
        # shell=False 时，CreateProcess 只会在字面名后补 .exe 去查，不会做
        # PATHEXT 解析 —— 于是裸 "npm" 抛 FileNotFoundError（WinError 2）。
        # 经 cmd /c 启动，让 cmd 自己做 PATHEXT 解析，命中 npm.cmd。
        subprocess.run(["cmd", "/c", "npm", "run", "dev"])
    else:
        subprocess.run([npm, "run", "dev"])


if __name__ == "__main__":
    main()
