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
import pathlib
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


# ── 契约守卫：run() 的返回键必须覆盖 chat() 读取的键 ──────────────
#
# 这条用例是补上「第一版修复为什么没生效」的教训：
#
# `AgenticRAGGraph.run()` 返回的是**固定键的 dict**。我最初只改了
# `rag_engine.chat()` 让它合并 `context_docs_b`，但 `run()` 根本没返回这个键
# → `result.get("context_docs_b")` 永远是 None → 修复静默失效。
#
# 而当时的测试没抓到：stub 的 `agent_graph.run()` **直接返回了** `context_docs_b`，
# 绕过了真正丢字段的那一层。**stub 打得太高，就没覆盖住出问题的那一层。**
#
# 这里改用源码级契约检查：把 `run()` 返回 dict 的键集合，与 `chat()` 里所有
# `result[...]` / `result.get(...)` 的键集合做包含关系断言。加键/删键都会被发现。


def _run_return_keys(src: str):
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    return {k.value for k in sub.value.keys if isinstance(k, ast.Constant)}
    return set()


def _chat_result_keys(src: str):
    """扫 `chat()` **以及它调用的 `_merge_context_docs`** 读取的 `result` 键。

    ⚠️ 必须带上 helper：合并逻辑搬进 `_merge_context_docs` 之后，
    `context_docs` / `context_docs_b` 的读取就不在 `chat()` 里了 ——
    只扫 `chat()` 会漏掉这两个（守卫会假绿）。
    """
    import ast
    tree = ast.parse(src)
    keys = set()
    for node in ast.walk(tree):
        is_target = (
            (isinstance(node, ast.AsyncFunctionDef) and node.name == "chat")
            or (isinstance(node, ast.FunctionDef) and node.name == "_merge_context_docs")
        )
        if is_target:
            for sub in ast.walk(node):
                # result.get("X")
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "get" and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "result"
                        and sub.args and isinstance(sub.args[0], ast.Constant)):
                    keys.add(sub.args[0].value)
                # result["X"]
                if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name)
                        and sub.value.id == "result" and isinstance(sub.slice, ast.Constant)):
                    keys.add(sub.slice.value)
    return keys


def test_graph_run_返回的键覆盖_rag_engine_chat_读取的键():
    """`run()` 返回的是固定键 dict —— 漏一个键就是**静默**丢数据。

    Bug 20 的第一版修复正是死在这里：`context_docs_b` 没在 `run()` 的返回里，
    `chat()` 拿到 None，修复毫无效果且不报错。
    """
    base = pathlib.Path(__file__).resolve().parent.parent / "app"
    provided = _run_return_keys((base / "agent" / "graph.py").read_text(encoding="utf-8"))
    consumed = _chat_result_keys((base / "services" / "rag_engine.py").read_text(encoding="utf-8"))

    assert provided, "没解析到 run() 的返回键 —— 契约守卫失效"
    assert consumed, "没解析到 chat() 读取的键 —— 契约守卫失效"
    missing = consumed - provided
    assert not missing, (
        f"`chat()` 读取了 run() 没有返回的键：{sorted(missing)} —— "
        f"这些会被静默丢掉（`.get()` 返回 None 不报错）。"
        f"\n  run() 提供：{sorted(provided)}"
        f"\n  chat() 需要：{sorted(consumed)}"
    )


def test_契约守卫确实能发现漏键(monkeypatch):
    """非空性检查：构造一个「run() 少返回一个键」的源码，守卫必须报出来。"""
    fake_graph = """
class G:
    async def run(self):
        return {"answer": "", "context_docs": []}
"""
    fake_engine = """
class E:
    async def chat(self):
        result = {}
        _ = result.get("context_docs_b", [])
        _ = result["answer"]
"""
    provided = _run_return_keys(fake_graph)
    consumed = _chat_result_keys(fake_engine)
    assert "context_docs_b" in consumed, "非空性检查失败：没解析到 chat() 的键"
    assert consumed - provided == {"context_docs_b"}, f"守卫没能发现漏键：{consumed - provided}"
