"""验证意图路由对「有数字但缺参数」的问题是否归到 calculate。

背景（2026-10-15）
------------------
缺年限的估算场景（"月薪8000，被辞退了，能拿多少补偿？"）曾被路由判为
`retrieve` —— 于是根本不会进入 calculate 节点，工具调用的「估算例外」
成了**死代码**。这个脚本专门守住这条路由边界。

需要真实 LLM（每例 1 次调用，很便宜）。

用法：
    cd backend && NO_PROXY="localhost,127.0.0.1" python scripts/check_intent_routing.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (问题, 期望意图, 说明)
CASES = [
    ("月薪8000，被辞退了，能拿多少补偿？", "calculate", "有月薪、缺年限，明确问金额"),
    ("月薪8000加班10小时加班费多少？", "calculate", "参数齐全"),
    ("工作5年被违法辞退赔偿金多少？", "calculate", "只有年限、缺月薪"),
    # 「完全没有数字、但明确问金额」→ calculate：
    # 进计算节点后，工具层会因缺参数拒绝调用，再由 CALCULATOR_PROMPT 的
    # 「不要代为假设数值」兜底，向用户索要参数 —— 比走 retrieve 让生成
    # 模型自由举例更可控（举例金额是模型心算的，不可追溯）。
    ("被辞退了，能拿多少补偿？", "calculate", "无数字但问金额 → 进计算节点索要参数"),
    ("经济补偿金怎么算？", "retrieve", "问方法、无数字"),
    ("经济补偿金和赔偿金有什么区别？", "compare", "对比类"),
    ("劳动合同法第47条怎么规定的？", "retrieve", "法条查询"),
]


async def main() -> int:
    from langchain_openai import ChatOpenAI

    from app.agent.nodes.router import route_intent
    from app.core.config import settings

    llm = ChatOpenAI(
        model=settings.EVAL_LLM_MODEL_NAME or settings.LLM_MODEL_NAME,
        openai_api_key=settings.EVAL_LLM_API_KEY or settings.LLM_API_KEY,
        openai_api_base=settings.EVAL_LLM_API_BASE or settings.LLM_API_BASE,
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    print(f"路由模型: {settings.EVAL_LLM_MODEL_NAME or settings.LLM_MODEL_NAME}")
    print("=" * 78)

    failed = 0
    for question, expect, note in CASES:
        state = {
            "question": question,
            "conversation_context": "",
            "rewritten_question": question,
            "steps": [],
        }
        try:
            result = await route_intent(llm, state)
            got = result.get("intent", "?")
        except Exception as exc:
            print(f"  ✗ 调用失败 {question!r}: {type(exc).__name__}: {exc}")
            failed += 1
            continue

        ok = got == expect
        failed += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} [{got:<9}] 期望 {expect:<9} | {question}   （{note}）")

    print("=" * 78)
    print(f"  通过 {len(CASES) - failed}/{len(CASES)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
