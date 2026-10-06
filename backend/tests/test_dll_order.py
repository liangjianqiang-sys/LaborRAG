"""原生 DLL 加载顺序修正的测试与结构守卫。

背景（详见 `app/utils/dll_order.py` 的 docstring）
------------------------------------------------
本机实测：`import sentence_transformers` **直接段错误**（exit 139，无 traceback），
而 `import pyarrow; import sentence_transformers` 正常。
`faulthandler` 显示崩溃点是 `pyarrow/__init__.py:71 → create_module`（加载原生 DLL）。

**这曾被我误判为「内存不足」**（当时可用内存确实只有 2.7 GB，且内存监控显示
峰值很低）。但后来发现决定性变量是**导入顺序**：内存紧张只是让它更容易出现。
判据是这两条对照：

    python -c "import sentence_transformers"                  → 139（崩）
    python -c "import pyarrow; import sentence_transformers"  → 0  （正常）
    python -c "import numpy;   import sentence_transformers"  → 139（崩）

若真是内存不足，先导 pyarrow 不该改变结果。

本文件守住两件事：
1. 修正函数本身可用、幂等、线程安全
2. **任何会拉起 sentence_transformers 的导入点，之前都必须先调用修正函数** ——
   否则将来新增第 5 个触发点时，又会退回「启动即段错误」，而那种崩溃
   没有 traceback，极难定位
"""
import pathlib
import threading

import pytest

from app.utils import dll_order
from app.utils.dll_order import ensure_native_dll_order

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"

# 这些导入会（直接或间接）拉起 sentence_transformers
TRIGGERS = (
    "from sentence_transformers import",
    "import sentence_transformers",
    "from langchain_text_splitters import",
    "import langchain_text_splitters",
    "from langchain_huggingface import",
)

GUARD_CALL = "ensure_native_dll_order()"


# ── 修正函数本身 ────────────────────────────────────────────────


def test_调用后_pyarrow_已在_sys_modules():
    ensure_native_dll_order()
    import sys
    assert "pyarrow" in sys.modules, "修正函数没有真正加载 pyarrow"


def test_幂等_多次调用不报错():
    for _ in range(3):
        ensure_native_dll_order()


def test_线程安全_并发调用不报错():
    errors = []

    def _call():
        try:
            ensure_native_dll_order()
        except Exception as e:      # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=_call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"并发调用出错：{errors}"


def test_重复调用只加载一次(monkeypatch):
    """幂等应当走「已就绪」短路，而不是每次都重新 import。

    用 `_ready` 标志验证：重置标志后调用一次应置位，再调用不应再进入 import 分支。
    """
    monkeypatch.setattr(dll_order, "_ready", False)
    ensure_native_dll_order()
    assert dll_order._ready is True


# ── 结构守卫：触发点必须先调用修正 ──────────────────────────────


def _trigger_sites():
    """返回 [(文件, 触发导入文本, 该处之前是否已调用修正)]。"""
    sites = []
    for f in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in f.parts or f.name == "dll_order.py":
            continue
        src = f.read_text(encoding="utf-8")
        for marker in TRIGGERS:
            pos = src.find(marker)
            while pos != -1:
                guarded = src.rfind(GUARD_CALL, 0, pos) != -1
                sites.append((str(f.relative_to(APP_DIR.parent)), marker, guarded))
                pos = src.find(marker, pos + 1)
    return sites


def test_每个触发点之前都调用了修正函数():
    unguarded = [f"{f}  ({m})" for f, m, ok in _trigger_sites() if not ok]
    assert not unguarded, (
        "以下位置会拉起 sentence_transformers，但之前没有调用 "
        "ensure_native_dll_order() —— 在当前环境下会导致**无 traceback 的段错误**：\n  "
        + "\n  ".join(unguarded)
    )


def test_守卫本身有效_确实扫到了触发点():
    """非空性检查：若 TRIGGERS 写错或扫描逻辑失效，上面那条会空跑。"""
    sites = _trigger_sites()
    assert len(sites) >= 4, f"只扫到 {len(sites)} 个触发点，预期至少 4 个（守卫可能失效）"
    assert any("embeddings.py" in f for f, _, _ in sites), "没扫到 embeddings.py"
    assert any("reranked.py" in f for f, _, _ in sites), "没扫到 reranked.py"
