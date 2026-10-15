"""验证计算工具在典型场景下是否被模型正确调用。

为什么需要这个脚本
------------------
「工具定义是否正确」可以离线单测（`tests/test_labor_tools.py`），
但「**模型会不会去调它、参数填得对不对**」是**模型行为**，只能真实调用才能验证。
离线测试覆盖不到这一层 —— 而这一层恰恰是工具调用方案能否成立的关键。

需要真实 LLM，会消耗少量额度（每个场景 1 次调用）。

用法：
    cd backend && NO_PROXY="localhost,127.0.0.1" python scripts/check_tool_calling.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings


# (场景说明, 期望行为, 问题)
CASES = [
    (
        "完整参数",
        "调 1 个工具，参数齐全",
        "月薪8000，工作5年，被辞退，经济补偿金是多少？",
    ),
    (
        "估算场景（给了月薪、缺年限）",
        "代入典型年限多次调用，金额全部来自工具",
        "月薪8000，被辞退了，能拿多少补偿？",
    ),
    (
        "缺月薪",
        "不应调用工具（月薪不可假设），应向用户索要",
        "被辞退了，能拿多少补偿？",
    ),
    (
        "加班类型识别",
        "传 rate_type=休息日加班",
        "月薪8000，周末加班10小时，加班费多少？",
    ),
    (
        "同时问两种",
        "调 2 个工具（补偿金 + 赔偿金）",
        "月薪8000工作5年被违法辞退，经济补偿金和赔偿金分别是多少？",
    ),
]


async def main() -> int:
    from langchain_openai import ChatOpenAI

    from app.agent.nodes.calculate import _calculate_via_tools

    print(f"模型: {settings.LLM_MODEL_NAME}")
    print("=" * 74)

    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        openai_api_key=settings.LLM_API_KEY,
        openai_api_base=settings.LLM_API_BASE,
        temperature=settings.LLM_TEMPERATURE,
        extra_body={"enable_thinking": False},
    )

    for label, expectation, question in CASES:
        print(f"\n【{label}】")
        print(f"  问题      : {question}")
        print(f"  期望      : {expectation}")
        try:
            result, trace = await _calculate_via_tools(llm, question, "")
        except Exception as exc:
            print(f"  ✗ 调用失败 : {type(exc).__name__}: {exc}")
            continue

        if trace:
            for item in trace:
                print(f"  ✓ 工具调用 : {item}")
        else:
            print("  — 未调用任何工具")

        if result:
            print(f"  结果首行   : {result.strip().splitlines()[0]}")

    print("\n" + "=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
