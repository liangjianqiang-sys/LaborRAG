"""`validate` 节点的测试 —— 代码级幻觉护栏。

为什么值得专门测
----------------
这是项目的招牌功能之一（回答里引用了检索文档中不存在的法条 → 移除），
零 LLM 调用、纯规则，却在此前**一行单测都没有**。
纯规则函数是「静默逻辑错误」的高发区：不报错、不抛异常，只是删错或漏删。

最严重的一条回归见 `test_只删除引用未支撑法条的那一句` ——
原实现的正则会**吞掉前面整段正确内容**，即护栏反而在破坏正确答案。
"""
from langchain_core.documents import Document

from app.agent.nodes.validate import validate


def _doc(text: str) -> tuple:
    return (Document(page_content=text, metadata={"source": "劳动合同法.txt"}), 0.9)


def _run(answer: str, contexts: list[str]) -> dict:
    return validate(
        {"answer": answer, "context_docs": [_doc(c) for c in contexts], "steps": []}
    )


# ── 回归：误删 ────────────────────────────────────────────────


def test_只删除引用未支撑法条的那一句():
    """**核心回归**：删除必须限定在引用该法条的那一句内。

    原实现用 `《?[^》]*》?` 这种宽松前缀 + 贪婪 `[^》]*`，会从更早的位置开始吞：

        输入：根据《劳动合同法》第四十七条，经济补偿按年限算。
              根据《劳动合同法》第九十九条，应当加倍处罚。
        实际：'根据《劳动合同法》'      ← 第一句也被吞了

    即幻觉护栏会**误删大段正确内容** —— 比它要防的幻觉更糟：用户不知情地丢掉
    正确信息，且没有任何报错。
    """
    answer = (
        "根据《劳动合同法》第四十七条，经济补偿按年限算。"
        "根据《劳动合同法》第九十九条，应当加倍处罚。"
    )
    out = _run(answer, ["第四十七条 经济补偿按劳动者在本单位工作的年限……"])
    assert out["answer"] == "根据《劳动合同法》第四十七条，经济补偿按年限算。"


def test_删除一句不影响相邻句():
    answer = "第一句引用第九十九条。第二句引用第四十七条。第三句无引用。"
    out = _run(answer, ["第四十七条 经济补偿……"])
    assert "第一句" not in out["answer"]
    assert "第二句引用第四十七条" in out["answer"]
    assert "第三句无引用" in out["answer"]


def test_分号分隔的条款同样按句处理():
    answer = "依据第四十七条应付补偿；依据第九十九条应加倍处罚；以上。"
    out = _run(answer, ["第四十七条 经济补偿……"])
    assert "第四十七条" in out["answer"]
    assert "第九十九条" not in out["answer"]
    assert "以上" in out["answer"]


# ── 正常路径 ──────────────────────────────────────────────────


def test_全部有支撑时答案原样保留():
    answer = "根据《劳动合同法》第四十七条，经济补偿按工作年限计算。"
    out = _run(answer, ["第四十七条 经济补偿按劳动者在本单位工作的年限……"])
    assert out["answer"] == answer
    assert "通过" in out["steps"][-1]


def test_无回答或无文档时跳过():
    out = _run("", ["第四十七条 x"])
    assert "跳过" in out["steps"][-1]
    assert "answer" not in out  # 跳过时不改答案

    out2 = _run("根据第四十七条 x", [])
    assert "跳过" in out2["steps"][-1]


def test_回答里没有法条引用时原样保留():
    answer = "建议您收集好证据后向劳动监察大队投诉。"
    out = _run(answer, ["第四十七条 经济补偿……"])
    assert out["answer"] == answer


# ── 归一化：两种写法必须等价 ──────────────────────────────────


def test_阿拉伯写法与中文写法等价():
    """回答用「第47条」、上下文用「第四十七条」，归一化后应判为有支撑。"""
    out = _run("依据《劳动合同法》第47条，应当支付经济补偿。", ["第四十七条 经济补偿……"])
    assert "第47条" in out["answer"]
    assert "通过" in out["steps"][-1]


def test_含零条号_〇写法_也能正确匹配():
    """回归 Bug 6 的连带影响：字符类漏「〇」会让「第一百〇一条」提取失败，
    进而被误判为幻觉、把正确引用删掉。"""
    out = _run(
        "根据《劳动法》第一百零一条，用人单位不得阻挠。",
        ["第一百〇一条 用人单位无理阻挠劳动行政部门行使监督检查权的。"],
    )
    assert "第一百零一条" in out["answer"]
    assert "通过" in out["steps"][-1]


# ── 取舍：混合引用的句子 ──────────────────────────────────────


def test_一句同时引用支撑与未支撑法条时整句丢弃():
    """保守取舍：护栏的职责是不让无依据的法律主张流出去。

    保留半句会留下断章取义的断言，故整句丢弃。
    """
    answer = "依据第四十七条与第九十九条，应当支付经济补偿并加倍处罚。"
    out = _run(answer, ["第四十七条 经济补偿……"])
    assert out["answer"] == ""
    assert "第99条" in out["steps"][-1]


def test_多处未支撑法条都会被记录():
    answer = "依据第九十九条处罚。依据第八十八条处理。依据第四十七条补偿。"
    out = _run(answer, ["第四十七条 经济补偿……"])
    step = out["steps"][-1]
    assert "第99条" in step and "第88条" in step
    assert out["answer"] == "依据第四十七条补偿。"


# ── 回归：删除不得破坏 markdown 结构 ──────────────────────────


def test_删除句子时保留_markdown_段落空行():
    """**核心回归（修复自身引入的）**：删句不能把段落空行一起吃掉。

    `_split_sentences` 的切分符含 `\\n`，所以段落空行（`\\n\\n`）会被切出来
    成为**独立的 `"\\n"` 片段**。若按 `s.strip()` 过滤，这些片段会被当成空白丢弃，
    `\\n\\n` 就塌成 `\\n` —— 而 markdown 里**单换行渲染成空格**，
    于是 `**小标题**` 被并进上一段正文。

    实测（2026-09-17，真实链路）护栏触发后答案里出现过：
        `两者不能同时主张。**对比分析：**`
    """
    answer = (
        "**结论：**\n"
        "经济补偿金是合法解除时的补偿。\n"
        "\n"
        "**对比分析：**\n"
        "赔偿金依据第九十九条计算。\n"
    )
    out = _run(answer, ["第四十七条 经济补偿……"])

    assert "第九十九条" not in out["answer"]
    assert "补偿。**对比分析：**" not in out["answer"], (
        f"小标题被并进上一段 —— 段落空行被吃掉了：{out['answer']!r}"
    )
    assert "\n\n**对比分析：**" in out["answer"], (
        f"段落空行未保留，markdown 结构被破坏：{out['answer']!r}"
    )
