"""静态导入网：遍历 app/ 下全部模块逐个 import，报告失败模块与 traceback。

为什么需要它：
`pytest` 全绿**不能**证明重构没破坏东西 —— 现有测试只覆盖少数模块，
而目录重构（移动文件 + 重写 import）的典型故障模式是"某个模块再也 import 不进来"，
这种故障在测试跑到它之前完全静默。本脚本提供一道独立的静态防线。

退出码：0 全部通过；1 存在失败模块（可直接用于 CI / pre-commit）。

用法：
    cd backend
    python scripts/smoke_import.py            # 检查全部
    python scripts/smoke_import.py -v         # 打印每个成功的模块
    python scripts/smoke_import.py --module app.knowledge.store
"""
import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"
SCRIPTS_DIR = BACKEND_DIR / "scripts"

# 脚本位于 backend/scripts/ 下，而包根是 backend/，需要显式补进 sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def discover_modules(root: Path) -> list[str]:
    """把 app/ 下的 .py 文件路径映射成可导入的模块名。"""
    modules: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(BACKEND_DIR).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            modules.append(".".join(parts))
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description="静态导入网")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每个成功的模块")
    parser.add_argument(
        "--module", action="append", default=[], help="只检查指定模块，可重复"
    )
    args = parser.parse_args()

    # 同时覆盖 app/ 与 scripts/：脚本搬家同样会断 import 链，
    # 而它们不在 app/ 下，只扫 app/ 会漏掉（run_eval.py 就是从 backend/ 移过来的）。
    modules = args.module or (discover_modules(APP_DIR) + discover_modules(SCRIPTS_DIR))
    failures: list[tuple[str, str]] = []
    slow: list[tuple[str, float]] = []

    print(f"检查 {len(modules)} 个模块（包根 {BACKEND_DIR}）\n")

    for name in modules:
        started = time.perf_counter()
        try:
            importlib.import_module(name)
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
            continue
        elapsed = time.perf_counter() - started
        if elapsed >= 1.0:
            slow.append((name, elapsed))
        if args.verbose:
            print(f"  OK    {name}  ({elapsed:.2f}s)")

    print(f"\n结果：{len(modules) - len(failures)}/{len(modules)} 通过")

    if slow:
        print("\n慢导入（>=1s）：")
        for name, elapsed in sorted(slow, key=lambda x: -x[1]):
            print(f"  {elapsed:6.2f}s  {name}")

    if failures:
        print(f"\n{'=' * 70}\n失败模块 {len(failures)} 个\n{'=' * 70}")
        for name, tb in failures:
            print(f"\n### {name}\n{tb}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
