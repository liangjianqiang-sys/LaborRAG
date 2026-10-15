"""通过 ASGI 直连验证 `/chat/stream` 的 SSE 输出。

为什么不用 curl 起服务测
------------------------
起真实服务要等模型加载（约 80s）且会占用端口，而这里要验证的只是
「SSE 帧格式 + 事件序列 + 鉴权挂载」这几件事 —— 用 httpx 的 ASGITransport
直接打 ASGI app 就够了，不起端口、不进网络栈。

会消耗一次完整链路的 LLM 调用。

用法：
    cd backend && NO_PROXY="localhost,127.0.0.1" python scripts/check_sse_endpoint.py
"""
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

QUESTION = "月薪8000加班10小时加班费多少？"
URL = "/api/v1/chat/stream"


async def main() -> int:
    from app.api.routes import set_engine
    from app.main import app
    from app.services.rag_engine import RAGEngine

    print("初始化引擎（加载向量库与模型）...")
    engine = RAGEngine()
    if not engine.initialize():
        print("✗ 知识库未就绪")
        return 1
    set_engine(engine)

    transport = httpx.ASGITransport(app=app)
    counts: Counter = Counter()
    steps: list[str] = []
    token_len = 0
    done_payload: dict | None = None
    errors: list[str] = []

    print(f"\nPOST {URL}  question={QUESTION!r}")
    print("=" * 70)

    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=300) as client:
        async with client.stream("POST", URL, json={"question": QUESTION}) as resp:
            print(f"HTTP {resp.status_code}  content-type={resp.headers.get('content-type')}")
            if resp.status_code != 200:
                print("✗ 非 200，SSE 未建立")
                return 1

            event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                    counts[event] += 1
                elif line.startswith("data:") and event:
                    raw = line[len("data:"):].lstrip()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        errors.append(f"数据帧不是合法 JSON：{raw[:80]}")
                        continue
                    if event == "step":
                        steps.append(payload.get("label", ""))
                    elif event == "token":
                        token_len += len(payload.get("text", ""))
                    elif event == "done":
                        done_payload = payload
                    elif event == "error":
                        errors.append(payload.get("message", ""))

    print("\n=== 事件统计 ===")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    print("\n=== step 序列 ===")
    print("  " + " → ".join(steps))

    print("\n=== 结果 ===")
    print(f"  token 累计字符: {token_len}")
    if done_payload:
        answer = done_payload.get("answer", "")
        print(f"  done.answer 长度: {len(answer)}")
        print(f"  sources: {len(done_payload.get('sources', []))} 条")
        print(f"  confidence: {done_payload.get('confidence')}")
        print(f"  conversation_id: {done_payload.get('conversation_id', '')[:8]}…")
    else:
        print("  ✗ 没有收到 done 事件")

    if errors:
        print(f"\n  ✗ 错误：{errors}")

    # step 是否去重（每个节点只应出现一次）
    dupes = [s for s, c in Counter(steps).items() if c > 1]
    print(f"\n  step 去重检查: {'✗ 有重复 ' + str(dupes) if dupes else '✓ 每个节点只出现一次'}")

    print("=" * 70)
    return 0 if (done_payload and not errors and not dupes) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
