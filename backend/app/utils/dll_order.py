"""原生 DLL 加载顺序修正 —— 一个环境相关崩溃的 workaround。

问题（2026-10-06 实测，conda `RAG` 环境）
----------------------------------------
```
python -c "import sentence_transformers"                  → 段错误 (exit 139)
python -c "import pyarrow; import sentence_transformers"  → 正常
python -c "import numpy;   import sentence_transformers"  → 段错误
```

`faulthandler` 抓到的栈显示崩溃点是：

    sentence_transformers/__init__.py:15
      → sentence_transformers/base/__init__.py:4
        → pandas/__init__.py:34 → pandas/compat/pyarrow.py:12
          → pyarrow/__init__.py:71 → create_module   ← **访问违规**

即**加载 `pyarrow.lib` 这个原生 DLL 时崩**。

`pandas` / `pyarrow` / `torch` **单独**导入都正常，`torch` 先导入再导 `pyarrow`
也正常 —— 只有「`pyarrow` 由 sentence_transformers 的链**间接**、且在 torch 之后
加载」才崩。**先显式导入 `pyarrow` 即可规避**（实测 3/3 稳定）。

## 为什么值得单独一个模块

这个函数只是把「先加载 pyarrow」这件事显式化，并把证据留在 docstring 里 ——
否则下一个人会再次把它误判成别的原因。本轮就误判过一次：最初归因于「内存不足」，
甚至写进了文档，直到发现**导入顺序才是决定性变量**（内存紧张只是让它更容易出现）。

## 成本

`import pyarrow` 实测 **0.62s**（`pandas` 要 2.65s，所以选 pyarrow）。
只在真正要加载 embedding / reranker / 文本切分器时调用，
不影响「廉价导入」那条路径（`import app.main` 仍为秒级）。

## 局限

这是**针对当前环境**的规避，不是根治。若换机器/重装依赖后仍崩，
应先确认是否同一栈（`pyarrow` 的 `create_module`），再决定是否保留本模块。
"""
import threading

_lock = threading.Lock()
_ready = False


def ensure_native_dll_order() -> None:
    """确保 `pyarrow` 已先于 sentence_transformers 的导入链加载。幂等、线程安全。

    调用点：任何会（直接或间接）拉起 `sentence_transformers` 的地方 ——
    embedding、reranker、`langchain_text_splitters`。见各调用处的注释。
    """
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:      # 双重检查：避免并发下重复导入
            return
        import pyarrow  # noqa: F401  仅为触发 DLL 加载，不使用其 API
        _ready = True
