"""重依赖必须惰性导入 —— 用 AST 扫描模块级 import 守住这条铁律。

为什么需要它
------------
项目有一条明确规则（写在 `app/services/rag_engine.py` 的注释里）：
langchain_openai / torch / transformers 这类重依赖**必须在函数内导入**，
顶层导入会让所有引用方付出几十秒代价 —— 包括整个测试会话与 smoke_import。

但这条规则此前只写在注释里，于是只在「有人恰好读到它」时生效。
结果是 **7 个模块在模块级 import `langchain_openai`**（2026-09-20 才发现），
而 `import app.agent.graph` 实测 17.5s。

本文件把规则变成会变红的检查。

为什么清单里**不含 langchain_core**
-----------------------------------
langchain_core 是基础库，但它的惰性 `__getattr__` 会连带 torch（实测）：

    from langchain_core.language_models import BaseChatModel     15.2s  torch=1
    from langchain_core.prompts import ChatPromptTemplate        16.6s  torch=1
    from langchain_core.output_parsers import StrOutputParser    15.7s  torch=1
    from langchain_core.messages import BaseMessage               0.7s  torch=0
    from langchain_core.documents import Document                 0.4s  torch=0

即 `language_models` / `prompts` / `output_parsers` 三个入口都经
`messages.block_translators` 拉到 torch。但这是 **langchain_core 的内部行为**，
不是本项目的「导入位置」问题 —— 而且 `prompts` / `language_models` 在多个模块里
是必需的（延迟导入只是把代价从启动挪到首次调用，并非净收益）。
所以本守卫只管「本项目的重依赖是否放对了位置」，不管上游库的内部代价。
"""
import ast
import pathlib

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"

HEAVY = (
    "langchain_openai",
    "torch",
    "transformers",
    "sentence_transformers",
    "datasets",
    "ragas",
    "pymupdf4llm",
    "langchain_community.document_loaders",
    "langchain_text_splitters",
    "huggingface_hub",
)


def _module_level_heavy_imports(source: str) -> list[tuple[int, str]]:
    """返回 [(行号, 模块名)]。只扫**模块级**（函数体/类体之外的）import。"""
    found: list[tuple[int, str]] = []
    for node in ast.parse(source).body:  # 只看顶层语句 → 函数内的 import 天然被排除
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module] if node.module else []
        else:
            continue
        for m in mods:
            if any(m == h or m.startswith(h + ".") for h in HEAVY):
                found.append((node.lineno, m))
    return found


def _scan_app() -> list[str]:
    hits = []
    for f in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        for lineno, mod in _module_level_heavy_imports(f.read_text(encoding="utf-8")):
            hits.append(f"{f.relative_to(APP_DIR.parent)}:{lineno}  {mod}")
    return hits


def test_app_下没有模块级重依赖导入():
    hits = _scan_app()
    assert not hits, (
        "以下位置在模块级导入了重依赖，请移进函数内"
        "（理由见 app/services/rag_engine.py 的同类注释）：\n  " + "\n  ".join(hits)
    )


def test_扫描器能检出模块级违规且不误报函数内():
    """非空性检查 —— 否则上面那条测试可能是空跑。

    同时验证「只扫模块级」这个关键性质：函数内的 import 不该被报出来，
    否则测试会逼着人把 import 提到模块级，正好做反。
    """
    src = (
        "import os\n"
        "from langchain_openai import ChatOpenAI\n"
        "import ragas.metrics\n"
        "def f():\n"
        "    from datasets import Dataset\n"
        "    import torch\n"
    )
    mods = {m for _, m in _module_level_heavy_imports(src)}
    assert "langchain_openai" in mods, f"漏检模块级 langchain_openai：{mods}"
    assert "ragas.metrics" in mods, f"漏检模块级 ragas.metrics：{mods}"
    assert "datasets" not in mods, f"误报函数内的 datasets：{mods}"
    assert "torch" not in mods, f"误报函数内的 torch：{mods}"
