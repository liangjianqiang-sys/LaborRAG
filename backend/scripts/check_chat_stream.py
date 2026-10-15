"""验证 RAGEngine.chat_stream() 的事件流。

不经过 HTTP —— 直接消费 async generator。要验证四件事：

1. **step 事件覆盖完整节点序列**（前端进度条的数据源）
2. **token 只来自生成节点** —— 不能混入 rewrite_query / route_intent 的输出
3. **done 事件含完整 ChatResponse**（sources / confidence / contexts）
4. **流式内容与 done.answer 一致** —— 差异只应来自护栏收尾（追加注意事项）

会消耗一次完整链路的 LLM 调用。

用法：
    cd backend && NO_PROXY="localhost,127.0.0.1" python scripts/check_chat_stream.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas.chat import ChatRequest

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "月薪8000，工作5年，被辞退，经济补偿金是多少？"


async def main() -> int:
    from app.services.rag_engine import RAGEngine

    print("正在初始化引擎（加载向量库与模型）...")
    engine = RAGEngine()
    if not engine.initialize():
        print("✗ 知识库未就绪")
        return 1

    print(f"\n问题: {QUESTION}")
    print("=" * 70)

    steps: list[str] = []
    tokens: list[str] = []
    done: dict | None = None
    errors: list[str] = []

    async for item in engine.chat_stream(ChatRequest(question=QUESTION)):
        ev, data = item["event"], item["data"]
        if ev == "step":
            steps.append(data["label"])
            print(f"  [step ] {data['label']}")
        elif ev == "token":
            tokens.append(data["text"])
        elif ev == "done":
            done = data
        elif ev == "error":
            errors.append(data["message"])
            print(f"  [error] {data['message']}")

    print("\n=== 结果 ===")
    print(f"  step 事件     : {len(steps)} 个 → {' → '.join(steps)}")
    print(f"  token 片段    : {len(tokens)} 个")

    if errors:
        print(f"  ✗ 收到 error  : {errors}")
        return 1
    if done is None:
        print("  ✗ 没有收到 done 事件")
        return 1

    streamed = "".join(tokens)
    answer = done.get("answer", "")
    print(f"  流式拼接长度  : {len(streamed)}")
    print(f"  done.answer   : {len(answer)}")
    print(f"  流式是答案前缀: {answer.startswith(streamed) if streamed else '（无 token）'}")
    print(f"  sources       : {len(done.get('sources', []))} 条")
    print(f"  confidence    : {done.get('confidence')}")
    print(f"  full_contexts : {len(done.get('full_contexts', []))} 条")
    print(f"  rag_steps     : {len(done.get('rag_steps', []))} 条")

    print("\n  answer 前 140 字：")
    print("   ", answer[:140].replace("\n", " "))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
