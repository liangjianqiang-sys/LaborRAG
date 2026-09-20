"""BM25 法律自定义词典的加载守卫。

目录重构中最容易**静默弄坏**的一处：`law_dict.txt` 的路径。

原实现用 `os.path.dirname(__file__)` 推算词典位置，文件一移动就指向不存在的
路径，而外面包着 `if os.path.exists(...)` —— 于是加载被**静默跳过**：不报错、
不影响启动，只是 jieba 把「经济补偿金」切成「经济/补偿/金」，BM25 的召回质量
悄悄下降。这类问题不会让任何测试变红，所以必须有专门的断言盯着。

现在路径基于 `BACKEND_DIR`（见 `app/core/config.py`），且缺失时直接抛
`FileNotFoundError`。本文件再加一层：验证词典**真的生效了**，而不只是"文件存在"。
"""
import jieba
import pytest

from app.core.config import BACKEND_DIR

# 这些术语必须整体成词。若词典未加载，jieba 会把它们切碎。
LEGAL_TERMS = [
    "经济补偿金",
    "上下班途中",
    "不定时工作制",
    "不签合同",
    "工伤认定",
    "二倍工资",
]


def test_词典文件位于预期位置():
    assert (BACKEND_DIR / "data" / "law_dict.txt").is_file()


def test_词典已生效_jieba不再切碎法律术语():
    """核心断言：验证**效果**，而不是仅验证文件存在。

    只断言「文件存在」是不够的 —— 原来的 bug 正是「文件存在但路径算错，
    加载被静默跳过」。必须断言分词结果。
    """
    import app.retrieval.bm25  # noqa: F401  导入即触发 load_userdict

    for term in LEGAL_TERMS:
        tokens = jieba.lcut(term)
        assert tokens == [term], (
            f"「{term}」被切成了 {tokens}，说明法律词典未生效"
        )


def test_词典缺失时必须抛错而非静默跳过(monkeypatch):
    """回归：原实现缺失时静默跳过，导致检索质量无声下降。

    静默降级比启动失败难排查得多 —— 缺词典时应当立刻失败。
    """
    import app.retrieval.bm25 as bm25_module

    monkeypatch.setattr(bm25_module, "BACKEND_DIR", BACKEND_DIR / "no_such_dir")
    with pytest.raises(FileNotFoundError, match="法律自定义词典缺失"):
        bm25_module._law_dict_path()
