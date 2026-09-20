"""`rewrite` 与 `router` 节点的测试（此前零覆盖）。

为什么优先测这两个
------------------
它们是**分叉点**：改写产出检索用查询、路由决定走哪条专家路径。
错在这里的代价是全链路性的 —— 本项目已发生两次：

- **Bug 9**：`classify_complexity` 只看改写结果，而改写会剥掉口语标记
  → 对比类问题被判成 simple，退化成普通检索问答
- **Bug 12**：`retrieve_for_compare` 从改写结果提取对比概念，而对比概念靠
  **连接词**切分，改写恰恰会把连接词剥掉 → 对比路径退化成单查询检索

两次的共同根因都是「**改写节点的输出语义被下游误用**」。本文件把这两个节点的
契约钉住：改写输出什么、路由如何解析。

用桩 LLM（`prompt | llm` 会把普通可调用对象包成 RunnableLambda，
节点只读返回值的 `.content`）。零额度、确定性。
"""
import asyncio
import contextlib
import io
import json

import pytest

from app.agent.nodes.rewrite import rewrite_query, _looks_professional
from app.agent.nodes.router import route_intent, decide_intent


# ── 桩 LLM ──────────────────────────────────────────────────────


class _Msg:
    """模拟 langchain 的 AIMessage：节点只读 `.content`。"""

    def __init__(self, content: str):
        self.content = content


class _StubLLM:
    """返回固定回复的桩。可调用对象 → `prompt | llm` 包成 RunnableLambda。"""

    def __init__(self, reply: str):
        self.reply = reply

    def __call__(self, prompt_value):
        return _Msg(self.reply)


def _run(node, reply, state):
    with contextlib.redirect_stdout(io.StringIO()):
        return asyncio.run(node(_StubLLM(reply), state))


# ── rewrite：正常解析 ──────────────────────────────────────────


def test_改写_用_legal_terms_拼接且拆解子查询():
    reply = json.dumps(
        {"legal_terms": ["用人单位单方解除劳动合同", "经济补偿金"],
         "sub_queries": ["违法解除 赔偿金", "经济补偿金 计算"]},
        ensure_ascii=False,
    )
    out = _run(rewrite_query, reply, {"question": "老板把我开了有补偿吗", "steps": []})
    assert out["rewritten_question"] == "用人单位单方解除劳动合同 经济补偿金"
    assert out["sub_queries"] == ["违法解除 赔偿金", "经济补偿金 计算"]
    assert "术语" in out["steps"][-1]


def test_改写_剥离_markdown_代码块围栏():
    """真实 LLM 经常把 JSON 包在 ```json 里 —— 不剥就解析失败、静默退化成原文。"""
    reply = '```json\n{"legal_terms": ["试用期解除劳动合同"], "sub_queries": ["试用期解除"]}\n```'
    out = _run(rewrite_query, reply, {"question": "试用期被辞了", "steps": []})
    assert out["rewritten_question"] == "试用期解除劳动合同"
    assert out["sub_queries"] == ["试用期解除"]


def test_改写_legal_terms_为空时保留原问题():
    reply = json.dumps({"legal_terms": [], "sub_queries": []}, ensure_ascii=False)
    out = _run(rewrite_query, reply, {"question": "加班费怎么算", "steps": []})
    assert out["rewritten_question"] == "加班费怎么算"
    assert out["sub_queries"] == ["加班费怎么算"]


def test_改写_sub_queries_缺失时回退为改写结果():
    reply = json.dumps({"legal_terms": ["加班工资 计算"]}, ensure_ascii=False)
    out = _run(rewrite_query, reply, {"question": "加班费", "steps": []})
    assert out["sub_queries"] == ["加班工资 计算"]


def test_改写_JSON_损坏时回退用原文():
    """解析失败必须**兜住**（不能抛），且把原文当改写结果继续跑。"""
    out = _run(rewrite_query, "这不是 JSON，但我是一条改写结果", {"question": "问个问题", "steps": []})
    assert out["rewritten_question"] == "这不是 JSON，但我是一条改写结果"
    assert out["sub_queries"] == ["这不是 JSON，但我是一条改写结果"]


def test_改写_JSON_损坏且内容等于原问题时保持原问题():
    out = _run(rewrite_query, "问个问题", {"question": "问个问题", "steps": []})
    assert out["rewritten_question"] == "问个问题"
    assert "无需改写" in out["steps"][-1]


# ── rewrite：跳过路径 ──────────────────────────────────────────


def test_改写_无上下文且问题已专业时跳过():
    """跳过条件：**无对话上下文** + 长度 > 30 + 含法律术语。

    三个条件缺一不可 —— 有上下文时必须改写（追问要靠上下文补全）。
    """
    q = "劳动合同法第八十二条规定的未签订书面劳动合同应当支付二倍工资该如何计算"
    assert len(q) > 30 and _looks_professional(q)
    out = _run(rewrite_query, "（不该被调用）", {"question": q, "steps": []})
    assert out["rewritten_question"] == q
    assert out["sub_queries"] == [q]
    assert "跳过" in out["steps"][-1]


def test_改写_有对话上下文时不跳过():
    q = "劳动合同法第八十二条规定的未签订书面劳动合同应当支付二倍工资该如何计算"
    reply = json.dumps({"legal_terms": ["二倍工资"], "sub_queries": ["二倍工资"]}, ensure_ascii=False)
    out = _run(rewrite_query, reply, {"question": q, "conversation_context": "上一轮在问赔偿", "steps": []})
    assert out["rewritten_question"] == "二倍工资", "有上下文时必须走改写，不能跳过"


def test_改写_问题过短时不跳过():
    out = _run(rewrite_query, json.dumps({"legal_terms": ["加班费"], "sub_queries": ["加班费"]},
                                         ensure_ascii=False),
               {"question": "加班费怎么算", "steps": []})
    assert out["rewritten_question"] == "加班费"


@pytest.mark.parametrize("q, expected", [
    ("劳动合同法第八十二条", True),
    ("经济补偿金和赔偿金有什么区别", True),
    ("《劳动法》怎么规定", True),
    ("今天天气怎么样", False),
    ("你好", False),
])
def test_是否已专业_术语判定(q, expected):
    assert _looks_professional(q) is expected


# ── router：意图解析 ──────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("calculate", "calculate"),
    ("compare", "compare"),
    ("retrieve", "retrieve"),
])
def test_路由_三个意图(raw, expected):
    out = _run(route_intent, raw, {"question": "q", "steps": []})
    assert out["intent"] == expected
    assert out["steps"][-1] == f"意图路由: {expected}"


@pytest.mark.parametrize("raw", ["  CALCULATE \n", "Calculate", "calcuLate"])
def test_路由_大小写与空白被归一化(raw):
    assert _run(route_intent, raw, {"question": "q", "steps": []})["intent"] == "calculate"


@pytest.mark.parametrize("raw", ["不知道", "", "none", "其他"])
def test_路由_无法识别时回退_retrieve(raw):
    assert _run(route_intent, raw, {"question": "q", "steps": []})["intent"] == "retrieve"


def test_路由_同时出现时_calculate_优先():
    """**钉住优先级**：解析用的是子串包含 + if/elif 顺序，calculate 先判。

    这个顺序是有意的（计算类问题通常也涉及检索，而 calculate 路径自带检索），
    但它由代码顺序决定、不写在别处 —— 改动顺序会让意图静默漂移。
    """
    assert _run(route_intent, "compare 还是 calculate", {"question": "q", "steps": []})["intent"] == "calculate"


def test_路由_子串匹配的已知脆弱性():
    """**记录一个已知弱点**（不是 bug，但值得知道边界在哪）。

    解析是 `"calculate" in intent` —— 所以 LLM 若输出「不适用 calculate」，
    也会被判成 calculate。真实模型基本只回一个词，所以风险低；
    但若将来换成会解释理由的模型，这里会静默走错分支。
    本用例把当前行为固定下来：改动它会变红，从而迫使改动者确认这是有意的。
    """
    assert _run(route_intent, "这个不属于 calculate", {"question": "q", "steps": []})["intent"] == "calculate"


def test_路由_优先用改写结果作为问题():
    """路由应该看**改写后**的问题（法言法语，路由更准）。"""
    captured = {}

    class _CaptureLLM:
        def __call__(self, prompt_value):
            captured["text"] = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
            return _Msg("retrieve")

    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(route_intent(_CaptureLLM(),
                                 {"question": "老板把我开了", "rewritten_question": "用人单位单方解除劳动合同",
                                  "steps": []}))
    assert "用人单位单方解除劳动合同" in captured["text"]


# ── router：分发 ────────────────────────────────────────────────


@pytest.mark.parametrize("intent, expected", [
    ("calculate", "calculate_path"),
    ("compare", "compare_path"),
    ("retrieve", "retrieve_path"),
    ("未知意图", "retrieve_path"),
])
def test_分发_意图到路径(intent, expected):
    assert decide_intent({"intent": intent}) == expected


def test_分发_意图缺失时默认走检索():
    assert decide_intent({}) == "retrieve_path"
