"""对比路径的概念提取：改写切不开时**必须回退原始问题**。

背景（2026-09-17 用真实 LLM 跑通评估后发现）
---------------------------------------------
`retrieve_for_compare` 从 `rewritten_question` 提取两个对比概念，而
`extract_compare_concepts` 靠**连接词**（和 / 与 / 及）切分。改写节点的职责是
产出**检索用查询**，会把连接词一并剥掉 —— 于是切分失败，对比路径**退化成单查询
检索**，丢掉「两组分别检索」这一核心设计（对比 prompt 的「禁止合成 / 禁止泛化」
约束正建立在两组独立检索之上）。

实测：

    原始「协商解除和公司单方辞退有什么区别？」→ ('协商解除', '公司单方辞退') ✓
    改写「协商解除 单方辞退」                  → 切分失败 ✗
    原始「经济补偿金和赔偿金有什么区别？」      → ('经济补偿金', '赔偿金') ✓
    改写「经济补偿金 赔偿金」                  → 切分失败 ✗

与 Bug 9（`classify_complexity` 只看改写结果而丢掉「区别」这个对比信号）
是**同一根因**：拿「检索查询」当「规则提取」的输入，而改写**有意**剥掉的正是
这些规则依赖的对话性标记。
"""
from app.agent.nodes.retrieve import _extract_concepts_with_fallback


def test_改写可切分时优先用改写结果():
    """改写保留了连接词 → 直接用改写结果（法言法语，检索更准）。"""
    a, b = _extract_concepts_with_fallback(
        rewritten="经济补偿金和赔偿金有什么区别？",
        original="经济补偿金和赔偿金有什么区别？",
    )
    assert (a, b) == ("经济补偿金", "赔偿金")


def test_改写切不开时回退原始问题():
    """回归：这是真实发生的情形 —— 改写把连接词剥掉了。"""
    a, b = _extract_concepts_with_fallback(
        rewritten="协商解除 单方辞退",              # 改写结果，连接词被剥掉
        original="协商解除和公司单方辞退有什么区别？",  # 原始问题，连接词完整
    )
    assert (a, b) == ("协商解除", "公司单方辞退")


def test_两处都切不开时返回失败信号():
    """两个概念相等即「提取失败」，调用方据此走单查询兜底。"""
    a, b = _extract_concepts_with_fallback(
        rewritten="经济补偿金 赔偿金",
        original="经济补偿金 赔偿金",
    )
    assert a == b == "经济补偿金 赔偿金"


def test_原始问题为空时不崩():
    a, b = _extract_concepts_with_fallback(rewritten="协商解除 单方辞退", original="")
    assert a == b == "协商解除 单方辞退"


# ── 调用点行为（只测 helper 是不够的）────────────────────────────
#
# 教训：最初只测 `_extract_concepts_with_fallback` 本身，结果把调用点退回修复前
# 的写法时测试**依然全绿** —— 那是空跑守卫。必须测调用点的实际行为。


class _RecordingRetriever:
    """只记录 query、返回空结果的检索器桩。"""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=5, score_threshold=0.2, **kwargs):
        self.queries.append(query)
        return []

    def is_ready(self):
        return True


def _run_compare(state):
    import asyncio
    import contextlib
    import io

    from app.agent.nodes.retrieve import retrieve_for_compare

    retriever = _RecordingRetriever()
    with contextlib.redirect_stdout(io.StringIO()):
        result = asyncio.run(retrieve_for_compare(retriever, state))
    return retriever.queries, result


def test_对比检索节点_改写切不开时用原始问题分别检索两组():
    """回归：改写剥掉连接词后，节点必须回退原始问题，而不是拿整句去检索。

    修复前的行为：concept_a == concept_b == 整句改写结果 → 两次检索同一个查询
    → 对比路径退化成单查询检索。
    """
    queries, _ = _run_compare(
        {
            "question": "协商解除和公司单方辞退有什么区别？",
            "rewritten_question": "协商解除 单方辞退",  # 连接词被剥掉
            "steps": [],
        }
    )

    assert "协商解除" in queries, f"未按概念切分检索：{queries}"
    assert "公司单方辞退" in queries, f"未按概念切分检索：{queries}"
    assert "协商解除 单方辞退" not in queries, (
        f"整句被当成查询 —— 对比路径退化成单查询检索：{queries}"
    )


def test_对比检索节点_改写可切分时沿用改写结果():
    """改写保留了连接词 → 用改写结果（法言法语，检索更准），不回退。"""
    queries, _ = _run_compare(
        {
            "question": "经济补偿金和赔偿金有什么区别？",
            "rewritten_question": "经济补偿金和赔偿金有什么区别？",
            "steps": [],
        }
    )
    assert queries == ["经济补偿金", "赔偿金"], queries
