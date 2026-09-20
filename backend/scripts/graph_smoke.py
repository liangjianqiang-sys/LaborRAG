"""Agent 图离线冒烟测试：用桩 LLM 把完整链路跑通。

为什么需要它
------------
现有验证网覆盖了「导入」「检索质量」「单元/契约」，但**完整链路
（改写 → 分类 → 路由 → 检索 → 生成 → 校验）从未端到端跑过** ——
它依赖真实 LLM，而 LLM 有额度、网络、成本三重不确定性
（本项目实测就撞到过免费额度耗尽，导致 `/chat` 完全不可用）。

本脚本把 LLM 换成**确定性桩**：按 prompt 内容返回预置回复。于是整条图可以在
**离线、零成本、可重复**的前提下跑通，回答两个问题：

  1. 节点之间的 state 传递有没有断（字段名写错、类型不符、缺失键）
  2. 检索到的法条有没有真的进入下游上下文

它**不能**替代真实 LLM 的质量评估（答案好不好、RAGAS 多少分）——
只能证明**管线是通的**。两者互补：本脚本保证"不会 500"，真实评估保证"答得对"。

用法：
    cd backend
    python scripts/graph_smoke.py              # 覆盖 retrieve / calculate / compare 三条路径
    python scripts/graph_smoke.py --verbose    # 显示节点内部 [Agent] 日志
"""
import argparse
import asyncio
import contextlib
import io
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ── 桩 LLM ────────────────────────────────────────────────────


class _Msg:
    """模拟 langchain 的 AIMessage：节点只读 `.content`。"""

    def __init__(self, content: str):
        self.content = content


def _prompt_text(prompt_value) -> str:
    if hasattr(prompt_value, "to_string"):
        return prompt_value.to_string()
    return str(prompt_value)


class StubLLM:
    """确定性桩 LLM：按 prompt 内容返回预置回复。

    为什么「按内容」而不是「按调用顺序」：节点执行顺序随问题类型变化
    （simple 短路 / retrieve / calculate / compare 四条路径），按顺序返回
    会在不同路径下错位。

    它必须是**可调用对象**：`prompt | llm` 会经 `coerce_to_runnable` 把普通
    callable 包成 `RunnableLambda`，其返回值即节点读到的 `result.content`。
    """

    # 识别语 → 固定回复。按顺序匹配，**更具体的放前面**。
    # 改写与路由两个节点需要「按问题作答」，不走这张表（见 __call__）。
    _RULES: list[tuple[str, str]] = [
        (
            "你是劳动法律师，请根据用户问题写一段假设性的专业回答",
            "根据《劳动合同法》第八十二条，用人单位自用工之日起超过一个月不满一年"
            "未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。",
        ),
        ("判断检索到的法律资料能否充分回答用户问题", "充分"),
        ("你是一位严格的法律答案质量评审专家", "充分"),
        ("你是一位资深的劳动法律专家", "（桩：修正后回答）"),
        (
            "额外规则（计算场景）",
            "计算结果：689.66 元。法条依据：《劳动法》第四十四条。",
        ),
        (
            "【对比场景核心约束",
            "第一组：《劳动合同法》第四十七条——经济补偿按工作年限计算。\n"
            "第二组：《劳动合同法》第八十七条——违法解除应支付二倍赔偿金。",
        ),
        (
            "参考资料：",
            "根据《劳动合同法》第八十二条与第十条，未签订书面劳动合同的，"
            "用人单位应当向劳动者每月支付二倍的工资。",
        ),
    ]

    _FALLBACK = "（桩 LLM：未识别的 prompt）"

    # 路由桩的判定规则：模拟真实 LLM 在 route_intent 里做的事。
    # 写成规则而非固定值，是为了能覆盖 retrieve / calculate / compare 三条路径 ——
    # 本脚本要验的是**图的接线**，不是 LLM 的判断力。
    _COMPARE_HINTS = ("区别", "对比", "不同", "比较", "差异", "分别")
    _CALCULATE_HINTS = ("多少", "怎么算", "计算", "几天", "倍数")

    def __init__(self):
        self.question = ""      # 由脚本在每次 run 前设置
        self.rewritten = ""     # 同上：该用例的「真实改写结果」
        self.unmatched: list[str] = []

    def _rewrite_reply(self) -> str:
        """改写桩：返回该用例预先声明的「真实改写结果」。

        刻意**不**简单回显原问题 —— 真实 LLM 的改写会滤掉关系性措辞
        （实测「经济补偿金和赔偿金有什么区别？」→「经济补偿金 赔偿金」），
        而这正是 `classify_complexity` 丢信号的来源。桩若回显原问题，
        就会把这类问题掩盖过去（本项目真实发生过一次：对比类问题被判成 simple，
        退化成普通检索问答）。
        """
        import json as _json

        terms = self.rewritten.split()
        return _json.dumps({"legal_terms": terms, "sub_queries": terms}, ensure_ascii=False)

    def _route_reply(self) -> str:
        if any(k in self.question for k in self._COMPARE_HINTS):
            return "compare"
        if any(k in self.question for k in self._CALCULATE_HINTS):
            return "calculate"
        return "retrieve"

    def __call__(self, prompt_value):
        text = _prompt_text(prompt_value)
        if "你是一个专业的中国劳动法律师" in text:      # 改写节点
            return _Msg(self._rewrite_reply())
        if "你是劳动法问答系统的意图路由器" in text:      # 路由节点
            return _Msg(self._route_reply())
        for marker, reply in self._RULES:
            if marker in text:
                return _Msg(reply)
        self.unmatched.append(text[:80])
        return _Msg(self._FALLBACK)


# ── 待测问题（覆盖三条 agent 路径）────────────────────────────

CASES = [
    {
        "question": "劳动合同法第八十二条规定了什么？",
        "rewritten": "劳动合同法第八十二条",
        "expect_intent": "",  # 含条号 → medium → 快速路径，不设 intent
        "why": "含条号 → medium → 快速路径（注意：快速路径**仍做检索+概念注入**）",
    },
    {
        "question": "哪种情况下可以主张二倍工资？",
        "rewritten": "二倍工资 主张情形",
        "expect_intent": "retrieve",
        "why": "含「哪种」→ complex → 检索路径",
    },
    {
        "question": "加班工资怎么算？",
        "rewritten": "加班工资 计算标准",
        "expect_intent": "calculate",
        "why": "含「怎么算」→ complex → 计算路径",
    },
    {
        "question": "经济补偿金和赔偿金有什么区别？",
        "rewritten": "经济补偿金 赔偿金",
        "expect_intent": "compare",
        "why": "含「区别」→ complex → 对比路径（**改写会把「区别」滤掉，"
               "分类必须同时看原问题**）",
    },
]


def _check(result: dict, case: dict) -> list[str]:
    """返回该用例的问题列表（空列表 = 通过）。"""
    problems = []
    q = case["question"]

    if not result.get("answer"):
        problems.append("answer 为空 —— 生成节点没产出内容")
    if result.get("intent") != case["expect_intent"]:
        problems.append(
            f"intent 期望 {case['expect_intent']!r}，实际 {result.get('intent')!r}"
        )

    # Bug 5 守卫：rewritten_question 不得等于 intent 值
    rq = result.get("rewritten_question", "")
    if not rq:
        problems.append("rewritten_question 为空")
    elif rq in ("retrieve", "calculate", "compare"):
        problems.append(f"rewritten_question 是意图值 {rq!r}（Bug 5 复发）")

    steps = result.get("steps", [])
    if not steps:
        problems.append("steps 为空 —— 节点没有记录流程")

    docs = result.get("context_docs", [])
    if not docs:
        problems.append("context_docs 为空 —— 检索结果没传到下游")
    else:
        missing_src = [d for d, _ in docs if not d.metadata.get("source")]
        if missing_src:
            problems.append(f"{len(missing_src)} 个文档缺 source 元数据")

    return problems


async def _run_case(graph, case: dict, verbose: bool) -> tuple[dict, str]:
    buf = io.StringIO()
    if verbose:
        return await graph.run(case["question"]), ""
    # 节点的 [Agent] 日志会淹没结果，默认吞掉
    with contextlib.redirect_stdout(buf):
        return await graph.run(case["question"]), buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 图离线冒烟")
    parser.add_argument("--verbose", action="store_true", help="显示节点内部 [Agent] 日志")
    args = parser.parse_args()

    print("加载向量库与检索器（首次需加载 BGE-M3，请稍候）...")
    from app.services.rag_engine import RAGEngine

    engine = RAGEngine()
    if not engine.initialize():
        print("向量库初始化失败：请先在 backend/ 下构建知识库")
        return 1

    graph = engine.agent_graph
    stub = StubLLM()
    graph.grader_llm = stub
    graph.gen_llm = stub
    print("已把 grader_llm / gen_llm 替换为确定性桩（零网络、零成本）\n")

    failed = 0
    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] {case['question']}   （{case['why']}）")
        # 桩需要知道当前问题与它的真实改写结果，才能「按问题作答」
        stub.question = case["question"]
        stub.rewritten = case["rewritten"]
        try:
            result, _ = asyncio.run(_run_case(graph, case, args.verbose))
        except Exception as e:
            print(f"    ✗ 图执行抛异常：{type(e).__name__}: {e}")
            failed += 1
            continue

        problems = _check(result, case)
        if problems:
            failed += 1
            print("    ✗ " + "\n      ".join(problems))
        else:
            docs = result.get("context_docs", [])
            print(
                f"    ✓ intent={result['intent']}  "
                f"改写={result['rewritten_question'][:34]!r}  "
                f"文档={len(docs)}  步骤={len(result.get('steps', []))}"
            )

    if stub.unmatched:
        print(f"\n⚠ 有 {len(stub.unmatched)} 次 LLM 调用未匹配到任何规则（返回了兜底文案）：")
        for t in stub.unmatched[:3]:
            print(f"    {t!r}")

    total = len(CASES)
    print(f"\n结果：{total - failed}/{total} 通过")
    if failed:
        print("注意：本脚本只证明「管线是通的」，不评估答案质量（那需要真实 LLM）。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
