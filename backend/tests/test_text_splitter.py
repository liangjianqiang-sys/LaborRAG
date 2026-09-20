"""text_splitter 法条切分器单测。

核心切分逻辑（_sanitize_doc_id / _split_by_article_parent_child /
_split_article_into_paragraphs / ARTICLE_PATTERN）是纯函数/静态方法，
可直接测；split_documents 公开入口依赖 settings，用 monkeypatch 切换两条路径。
"""
import re

from langchain_core.documents import Document

from app.knowledge.splitter import LawArticleSplitter
from app.core.config import settings


# ── 静态/纯方法：零依赖 ──

def test_sanitize_doc_id():
    """source 路径 → 安全 doc_id，只保留字母/数字/下划线/中文。"""
    s = LawArticleSplitter
    # 纯中文名不变
    assert s._sanitize_doc_id("data/中华人民共和国劳动法.txt") == "中华人民共和国劳动法"
    # 特殊字符（-）转下划线
    assert s._sanitize_doc_id("labor-law.pdf") == "labor_law"
    # 只取文件名不含扩展名
    assert s._sanitize_doc_id("a/b/c.txt") == "c"


def test_article_pattern_edges():
    """ARTICLE_PATTERN 要求行首 + 第X条 + 空格；无空格不匹配（走 fallback）。"""
    P = LawArticleSplitter.ARTICLE_PATTERN
    # 数字/中文/大数都匹配
    assert P.match("第10条 内容") is not None
    assert P.match("第四十四条 内容") is not None
    assert P.match("第一百零八条 内容") is not None
    assert P.match("第1000条 内容") is not None
    # 无空格不匹配（条后是句号）
    assert P.match("第一条。") is None
    assert P.match("第47条") is None  # 结尾无空格


def test_split_by_article_parent_child():
    """多法条文本 → 父块数=法条数，parent_id 格式正确，父子块结构对。"""
    splitter = LawArticleSplitter()
    text = (
        "第一条 标题\n内容1行A\n内容1行B\n"
        "第二条 标题2\n内容2"
    )
    chunks = splitter._split_by_article_parent_child(text, "测试法.txt", "测试法")

    parents = [c for c in chunks if c.metadata["chunk_type"] == "parent"]
    children = [c for c in chunks if c.metadata["chunk_type"] == "child"]
    assert len(parents) == 2  # 两条法条 → 2 个父块
    assert {p.metadata["parent_id"] for p in parents} == {"测试法_第一条", "测试法_第二条"}
    assert parents[0].metadata["article"] == "第一条"
    # 父块内容含整条法条（不截断）
    assert "内容1行A" in parents[0].page_content and "内容1行B" in parents[0].page_content


def test_split_article_into_paragraphs():
    """款 （X） 起新段、标题单独成段、连续行合并。"""
    splitter = LawArticleSplitter()
    lines = ["第一条 标题", "（一）款A", "内容A", "（二）款B"]
    paras = splitter._split_article_into_paragraphs(lines)
    assert paras == ["第一条 标题", "（一）款A\n内容A", "（二）款B"]


def test_split_documents_with_monkeypatch(monkeypatch):
    """两条路径 metadata 差异：parent 模式有 chunk_type/parent_id，普通模式没有。"""
    splitter = LawArticleSplitter()
    doc = Document(page_content="第一条 标题\n内容A\n第二条 标题2\n内容B", metadata={"source": "t.txt"})

    # 父子模式
    monkeypatch.setattr(settings, "PARENT_CHILD_ENABLED", True)
    pc_chunks = splitter.split_documents([doc])
    assert all("chunk_type" in c.metadata for c in pc_chunks)
    assert all("parent_id" in c.metadata for c in pc_chunks)

    # 普通模式
    monkeypatch.setattr(settings, "PARENT_CHILD_ENABLED", False)
    plain_chunks = splitter.split_documents([doc])
    assert len(plain_chunks) >= 1
    assert all("chunk_type" not in c.metadata for c in plain_chunks)
    assert all("parent_id" not in c.metadata for c in plain_chunks)
