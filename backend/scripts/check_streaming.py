"""探测 LangGraph 的 astream_events 事件结构。

用途
----
设计 `/chat/stream`（SSE）端点前，先确认两件事，避免按猜测实现：

1. **节点级进度** —— 能否实时拿到「改写 → 分类 → 路由 → 检索 → 生成」，
   用于给前端推送 step 事件（解决"等待时毫无反馈"）。
2. **生成节点的 token 级事件** —— 能否拿到 `on_chat_model_stream`，
   并按 `metadata.langgraph_node` 过滤出**生成节点**的 token。
   注意：辅助 LLM（改写 / 路由 / HyDE）也会产生 token，必须能区分，
   否则前端会把"改写过程"当成"答案"流出来。

需要真实 LLM + 已构建的向量库。

用法：
    cd backend && NO_PROXY="localhost,127.0.0.1" python scripts/check_streaming.py
"""
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复杂问题 —— 走完整 agent 链路（改写→分类→路由→检索→生成→校验）
QUESTION = "经济补偿金和赔偿金有什么区别？"


async def main() -> int:
    from app.agent.graph import AgenticRAGGraph
    from app.knowledge.store import VectorStoreManager
    from app.retrieval import build_retriever
    from app.retrieval.bm25 import BM25Retriever

    vsm = VectorStoreManager()
    if not vsm.load():
        print("✗ 向量库未就绪，无法探测")
        return 1
    bm25 = BM25Retriever()
    bm25.load_index()
    graph = AgenticRAGGraph(build_retriever(vsm, bm25))

    counts: Counter = Counter()
    node_events: list[str] = []
    token_nodes: Counter = Counter()
    sample_tokens: dict[str, list[str]] = {}

    print(f"问题: {QUESTION}")
    print("=" * 70)

    async for ev in graph.graph.astream_events(
        {"question": QUESTION, "conversation_context": "", "steps": []},
        version="v2",
    ):
        kind = ev["event"]
        counts[kind] += 1
        node = ev.get("metadata", {}).get("langgraph_node")

        if kind == "on_chain_start" and node:
            node_events.append(node)
        elif kind == "on_chat_model_stream":
            chunk = ev.get("data", {}).get("chunk")
            text = getattr(chunk, "content", "") or ""
            if text:
                key = node or "(无 node)"
                token_nodes[key] += 1
                sample_tokens.setdefault(key, [])
                if len(sample_tokens[key]) < 3:
                    sample_tokens[key].append(text)

    print("\n=== 事件类型统计 ===")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")

    print("\n=== 节点执行顺序（on_chain_start）===")
    seen = []
    for n in node_events:
        if not seen or seen[-1] != n:
            seen.append(n)
    print("  " + " → ".join(seen))

    print("\n=== token 来源节点（on_chat_model_stream 按 node 分组）===")
    if not token_nodes:
        print("  ✗ 没有捕获到任何 token —— 该版本可能不支持 astream_events 的流式")
    for n, c in token_nodes.most_common():
        print(f"  {n}: {c} 个 token 片段   样例={sample_tokens[n]}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
