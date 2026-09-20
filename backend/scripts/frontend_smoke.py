"""前端冒烟：验证「浏览器里真的能用」，而不只是「能编译」。

为什么需要它
------------
`tsc -b` + `vite build` 只能证明**类型正确、能打包**，证明不了：

  - React 真的挂载了（而不是白屏 —— 运行时错误在构建阶段看不出来）
  - 四个路由（问答 / 知识库 / 评估 / 设置）都能渲染
  - 浏览器真的调通了后端 API（而不是卡在 CORS / 代理 / 鉴权头缺失）

本项目真实发生过：`api.ts` 的 16 处 fetch 全部漏带鉴权头，`tsc` 与 `vite build`
都正常通过 —— 因为「少一个请求头」在类型层面完全合法。

做法：用**系统已装的 Chrome** 无头渲染每个路由，断言页面出现该页的特征内容。
用系统 Chrome 而非 agent-browser：后者要额外下载 ~500MB 的 Chromium，
而 Windows 自带 Edge、多数开发机也有 Chrome，够用了。

前置：后端（:8000）与前端 dev server（:5173）都已在运行。

用法：
    cd backend
    python scripts/frontend_smoke.py
    python scripts/frontend_smoke.py --chrome "C:/Program Files/Google/Chrome/Application/chrome.exe"
    python scripts/frontend_smoke.py --base http://127.0.0.1:5173
"""
import argparse
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

# 各路由的特征内容。命中率不足即视为渲染失败。
ROUTES = [
    {
        "path": "/",
        "name": "智能问答",
        "expect": ["劳动法", "问答", "知识库"],
    },
    {
        "path": "/knowledge",
        "name": "知识库管理",
        "expect": ["知识库", "文档", "构建", "上传"],
    },
    {
        "path": "/evaluation",
        "name": "评估",
        "expect": ["评估", "指标"],
    },
    {
        "path": "/settings",
        "name": "设置",
        "expect": ["设置", "鉴权", "令牌"],
    },
]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit if pathlib.Path(explicit).exists() else None
    for c in CHROME_CANDIDATES:
        if pathlib.Path(c).exists():
            return c
    return None


def render(browser: str, url: str, budget_ms: int = 9000) -> str:
    """无头渲染并返回 DOM。--virtual-time-budget 让 JS 有机会跑完。"""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir={tmp}",
                f"--virtual-time-budget={budget_ms}",
                "--dump-dom",
                url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
    return proc.stdout or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="前端冒烟（无头浏览器）")
    parser.add_argument("--base", default="http://127.0.0.1:5173", help="前端地址")
    parser.add_argument("--chrome", default=None, help="浏览器可执行文件路径")
    parser.add_argument("--backend", default="http://127.0.0.1:8000", help="后端地址")
    args = parser.parse_args()

    browser = find_browser(args.chrome)
    if not browser:
        print("未找到 Chrome/Edge。请用 --chrome 指定路径。")
        return 1
    print(f"浏览器: {browser}")

    # 先确认两个服务都在
    for label, url in (("后端", f"{args.backend}/health"), ("前端", args.base + "/")):
        try:
            urllib.request.urlopen(url, timeout=8)
            print(f"  {label} 可达 ✓")
        except urllib.error.HTTPError:
            print(f"  {label} 可达（返回错误状态，但服务在跑）")
        except Exception as e:
            print(f"  {label} 不可达 ✗  {type(e).__name__}")
            print("  请先启动后端（python run.py）与前端（npm run dev）")
            return 1
    print()

    failed = 0
    for route in ROUTES:
        url = args.base + route["path"]
        try:
            dom = render(browser, url)
        except Exception as e:
            print(f"[{route['name']:<8}] ✗ 渲染失败: {type(e).__name__}: {e}")
            failed += 1
            continue

        if not dom:
            print(f"[{route['name']:<8}] ✗ DOM 为空（白屏）")
            failed += 1
            continue

        missing = [kw for kw in route["expect"] if kw not in dom]
        if missing:
            print(f"[{route['name']:<8}] ✗ 缺少特征内容 {missing}（DOM {len(dom)} 字节）")
            failed += 1
        else:
            print(f"[{route['name']:<8}] ✓ DOM {len(dom):>7} 字节")

    # 知识库页必须显示**来自 API 的真实文档名** —— 这是「浏览器真的调通了后端」的证据
    print()
    try:
        dom = render(browser, args.base + "/knowledge")
        names = {n.strip() for n in re.findall(r"[^<>]*\.(?:pdf|txt)[^<>]*", dom) if 0 < len(n.strip()) < 60}
        if names:
            print(f"API 连通性: ✓ 页面渲染出 {len(names)} 个真实文档名")
            for n in sorted(names)[:3]:
                print(f"    · {n}")
        else:
            print("API 连通性: ✗ 知识库页没有出现任何文档名 —— 浏览器可能没调通后端")
            failed += 1
    except Exception as e:
        print(f"API 连通性检查失败: {type(e).__name__}: {e}")
        failed += 1

    total = len(ROUTES)
    print(f"\n结果：{total - failed}/{total} 路由通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
