"""对比题上下文合并的测试 —— Bug 20 的回归网。

背景
----
对比题走 `retrieve_for_compare`，返回**两组**文档：

    {"context_docs": docs_a, "context_docs_b": docs_b}

`compare` 节点把**两组都**喂进 prompt 生成答案；但 `rag_engine.chat()` 此前只取
`result["context_docs"]`，于是 `sources` / `full_contexts` / `child_contexts`
全都只含 A 组 —— **评估只看到答案所依据的一半证据**。

20 题全量评估实测到的后果：

| 类型 | 题数 | R@5 | context_precision |
|---|---|---|---|
| retrieve | 13 | 0.8846 | — |
| calculate | 4 | 0.8750 | — |
| **compare** | **3** | **0.4444** | **0.0** |

compare 整类都是异常低点。逐题探测证实：**B 组里装着 A 组缺的那些标注法条**
（两题的 R@5 从 0.00 → 1.00）。

本文件守住三件事：
1. B 组必须进入评估上下文
2. 非对比题行为**完全不变**（没有 `context_docs_b`）
3. 两组重叠时要**去重**，否则同一篇法条会占掉两个位次、虚增参考来源
"""
import asyncio
import types

import pytest

from app.schemas.chat import ChatRequest


# ── 替身 ────────────────────────────────────────────────────────


class _Doc:
    """最小 Document 替身：合并逻辑只用到 metadata.source 与 page_content。"""

    def __init__(self, source: str, content: str):
        self.metadata = {"source": source}
        self.page_content = content

    def __repr__(self):  # pragma: no cover
        return f"_Doc({self.metadata['source']!r})"


def _pair(source: str, content: str, score: float = 0.9):
    return (_Doc(source, content), score)


class _ConvStub:
    def add_message(self, *a, **k):
        pass

    def build_context_string(self, *a, **k):
        return ""


class _GraphStub:
    """`agent_graph` 替身：`run()` 返回预置结果，不发任何请求。"""

    def __init__(self, result: dict):
        self.retriever = None
        self._result = result

    async def run(self, *a, **k):
        return self._result


@pytest.fixture
def engine(real_rag_engine_module, monkeypatch):
    """绕过 `__init__`（会加载 BGE-M3，约 60s）构造一个可用的引擎替身。"""
    mod = real_rag_engine_module
    eng = mod.RAGEngine.__new__(mod.RAGEngine)
    monkeypatch.setattr(mod, "conversation_manager", _ConvStub())
    return eng, mod


def _run_chat(eng, result):
    eng.agent_graph = _GraphStub(result)
    return asyncio.run(eng.chat(ChatRequest(question="对比一下")))


# ── 纯函数：合并与去重 ──────────────────────────────────────────


def test_对比题_B组被纳入(real_rag_engine_module):
    f = real_rag_engine_module._merge_context_docs
    result = {
        "context_docs": [_pair("A.txt", "协商解除的内容")],
        "context_docs_b": [_pair("B.txt", "单方辞退的内容")],
    }
    out = f(result)
    assert [d.metadata["source"] for d, _ in out] == ["A.txt", "B.txt"]


def test_非对比题_行为不变(real_rag_engine_module):
    """没有 `context_docs_b` 时，结果必须与改前**逐项相同**（含顺序）。"""
    f = real_rag_engine_module._merge_context_docs
    docs = [_pair("A.txt", "一"), _pair("B.txt", "二"), _pair("C.txt", "三")]
    assert f({"context_docs": docs}) == docs


def test_两组重叠时去重(real_rag_engine_module):
    """同一篇法条可能同时命中 A/B 两组 —— 不去重会占掉两个位次。"""
    f = real_rag_engine_module._merge_context_docs
    same = _pair("共同.txt", "同一篇法条")
    out = f({"context_docs": [same], "context_docs_b": [same]})
    assert len(out) == 1, f"未去重：{out}"


def test_去重保留首次出现的位次(real_rag_engine_module):
    """A 组在前 → 去重后 A 组的位次不变（影响 P@k 的排序语义）。"""
    f = real_rag_engine_module._merge_context_docs
    a1 = _pair("a1.txt", "内容1")
    dup = _pair("dup.txt", "重复")
    b_only = _pair("b1.txt", "内容2")
    out = f({"context_docs": [a1, dup], "context_docs_b": [dup, b_only]})
    assert [d.metadata["source"] for d, _ in out] == ["a1.txt", "dup.txt", "b1.txt"]


def test_内容相同但来源不同_不误去重(real_rag_engine_module):
    """去重键是 (source, content)，不同来源的同名条文不该被合并。"""
    f = real_rag_engine_module._merge_context_docs
    out = f({"context_docs": [_pair("甲法.txt", "同样的话")],
             "context_docs_b": [_pair("乙法.txt", "同样的话")]})
    assert len(out) == 2


def test_空_B组与缺失_B组都不报错(real_rag_engine_module):
    f = real_rag_engine_module._merge_context_docs
    one = [_pair("A.txt", "x")]
    assert f({"context_docs": one, "context_docs_b": []}) == one
    assert f({"context_docs": one}) == one


def test_结构异常时不丢内容(real_rag_engine_module):
    """去重逻辑不该把结构异常的元素静默丢掉 —— 宁可重复，不可丢失。"""
    f = real_rag_engine_module._merge_context_docs
    weird = "我不是 (doc, score) 元组"
    out = f({"context_docs": [_pair("A.txt", "x")], "context_docs_b": [weird]})
    assert weird in out, f"异常元素被丢弃了：{out}"


# ── 端到端：chat() 返回的评估上下文必须含 B 组 ──────────────────


def test_chat_的_full_contexts_包含_B组(engine):
    """核心行为断言：**评估看到的上下文 = 答案所依据的上下文**。"""
    eng, _mod = engine
    resp = _run_chat(eng, {
        "answer": "对比结论",
        "context_docs": [_pair("A.txt", "协商解除")],
        "context_docs_b": [_pair("B.txt", "单方辞退")],
    })
    joined = "".join(resp.full_contexts)
    assert "协商解除" in joined and "单方辞退" in joined, resp.full_contexts
    assert {s.source for s in resp.sources} == {"A.txt", "B.txt"}


def test_chat_的_child_contexts_包含_B组(engine):
    """RAGAS 的 Context Precision/Recall 用的是 child_contexts，同样必须含 B。"""
    eng, _mod = engine
    resp = _run_chat(eng, {
        "answer": "对比结论",
        "context_docs": [_pair("A.txt", "协商解除")],
        "context_docs_b": [_pair("B.txt", "单方辞退")],
    })
    joined = "".join(resp.child_contexts)
    assert "协商解除" in joined and "单方辞退" in joined, resp.child_contexts


def test_chat_非对比题不引入多余上下文(engine):
    """非空性对照：普通检索题不该因为这次改动而多出内容。"""
    eng, _mod = engine
    resp = _run_chat(eng, {
        "answer": "普通回答",
        "context_docs": [_pair("A.txt", "只有一组")],
    })
    assert len(resp.full_contexts) == 1
    assert "只有一组" in resp.full_contexts[0]
