"""完整链路检索评估（官方口径）：复现 README 里那组 P@1/R@5 数字。

和 `retrieval_baseline.py` 的区别（**两个数字不一样是正常的，口径不同**）
------------------------------------------------------------------------
| | `retrieval_baseline.py` | 本脚本 |
|---|---|---|
| 链路 | **零 LLM 子集**：向量+BM25+RRF → 父子分块 → 知识图谱注入 → 概念置顶 | **完整链路**：LLM 改写 + HyDE + 多查询扩展 → 重排 → 注入 → 置顶 |
| 检索器 | 默认 hybrid（可 `--retriever reranked` 消融） | 强制 `use_reranker=True`（与官方评估一致） |
| 成本 | 零 | 每题一次完整 `chat()` |
| 确定性 | 是（可复现，Δ 恒为 0） | 否（LLM 改写不确定） |
| 用途 | **重构回归网**：证明「没弄坏」 | **对外指标**：论文/README 里那组数 |

两者不可互相替代：前者要的是可复现，后者要的是与官方评估同口径。

本脚本做的是官方 `EvalRunner` 的**阶段 1 + 阶段 2**（生成答案 → 检索指标），
**跳过阶段 3 的 RAGAS** —— RAGAS 占 token 消耗 30–40% 且与检索指标无关。
所以它比跑完整评估便宜得多，但仍用官方的 `compute_retrieval_metrics` 计算，
保证数字与官方一致。

用法：
    cd backend
    python scripts/retrieval_eval_full.py                 # 20 题 Full Set
    python scripts/retrieval_eval_full.py --subset smoke  # 5 题冒烟
    python scripts/retrieval_eval_full.py -o baselines/agentic_eval.json
"""
import argparse
import asyncio
import contextlib
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.evaluation.datasets.golden_set import EVAL_DATASET  # noqa: E402
from app.evaluation.metrics.retrieval import compute_retrieval_metrics  # noqa: E402

def select_items(subset: str) -> list[dict]:
    if subset == "all":
        return list(EVAL_DATASET)
    return [item for item in EVAL_DATASET if subset in item.get("eval_subset", [])]


async def eval_one(engine, item: dict) -> dict | None:
    """跑一题完整链路，返回官方口径的 sources 结构。"""
    from app.schemas.chat import ChatRequest

    try:
        resp = await engine.chat(
            ChatRequest(question=item["question"], skip_guardrails=True),
            use_reranker=True,
        )
    except Exception as e:
        print(f"    ✗ 生成失败: {type(e).__name__}: {str(e)[:90]}")
        return None

    # 与 EvalRunner._gen_answer 完全一致的结构（content 已被截断到 200 字符）
    return {
        "question": item["question"],
        "sources": [{"source": s.source, "content": s.content} for s in resp.sources],
        "answer_len": len(resp.answer or ""),
        "rewritten_question": resp.rewritten_question,
    }


async def run_all(engine, items: list[dict], quiet: bool) -> list[dict | None]:
    out = []
    for i, item in enumerate(items, 1):
        print(f"  [{i:>2}/{len(items)}] {item['question'][:36]}")
        buf = io.StringIO()
        if quiet:
            with contextlib.redirect_stdout(buf):
                r = await eval_one(engine, item)
        else:
            r = await eval_one(engine, item)
        out.append(r)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="完整链路检索评估（官方口径）")
    parser.add_argument("--subset", choices=["smoke", "full", "all"], default="full")
    parser.add_argument("-o", "--output", default="baselines/agentic_eval.json")
    parser.add_argument("--verbose", action="store_true", help="显示节点内部日志")
    args = parser.parse_args()

    items = select_items(args.subset)
    if not items:
        print(f"数据集为空（subset={args.subset}）")
        return 1

    print(f"题量      : {len(items)}（subset={args.subset}）")
    print("链路      : 完整（LLM 改写 + HyDE + 多查询 + 重排）")
    print("计算      : 官方 compute_retrieval_metrics，k=5")
    print("跳过      : 阶段 3 的 RAGAS（与检索指标无关，且占 30–40% token）")
    print("\n加载引擎（首次需加载 BGE-M3，请稍候）...")

    started = time.perf_counter()
    from app.services.rag_engine import RAGEngine

    engine = RAGEngine()
    if not engine.initialize():
        print("向量库初始化失败")
        return 1
    print(f"就绪，用时 {time.perf_counter() - started:.1f}s\n")

    t0 = time.perf_counter()
    results = asyncio.run(run_all(engine, items, quiet=not args.verbose))
    elapsed = time.perf_counter() - t0

    ok = [r for r in results if r is not None]
    if not ok:
        print("\n全部失败 —— 请检查 LLM 是否可用（额度 / 密钥 / 网络）")
        return 1

    # 官方口径：sources_raw 与 relevant_articles 一一对应
    sources_raw = [r["sources"] for r in results if r is not None]
    relevant = [item.get("relevant_articles", []) for item, r in zip(items, results) if r is not None]

    metrics = compute_retrieval_metrics(sources_raw, relevant, k=5)

    # ⚠️ 官方的 compute_retrieval_metrics **只产出 precision@1 与 recall@k**
    # （见其文档字符串）。P@3/P@5/MRR/MAP 需要自己用 per_query 里的
    # retrieved_ids / relevant_ids 另算 —— 本脚本不这么做，以保持与官方口径一致。
    # 那四个指标在零 LLM 的 retrieval_baseline.py 里有覆盖。
    print(f"\n{'指标':<14}{'值':>8}")
    for key in ("precision@1", "recall@5"):
        print(f"  {key:<12}{metrics.get(key, 0.0):>8.4f}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = BACKEND_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "scope": "full-agentic（LLM 改写 + HyDE + 多查询 + 重排）",
                    "subset": args.subset,
                    "dataset_size": len(items),
                    "succeeded": len(ok),
                    "elapsed_sec": round(elapsed, 1),
                },
                "aggregate": {k: metrics.get(k) for k in ("precision@1", "recall@5")},
                "per_query": metrics.get("per_query", []),
                "raw": [{"question": r["question"], "rewritten_question": r["rewritten_question"],
                         "answer_len": r["answer_len"]} for r in ok],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n成功 {len(ok)}/{len(items)} 题，用时 {elapsed:.0f}s")
    print(f"结果已写入 {out_path}")
    print("\n⚠️ 本口径含 LLM，**不可复现**（改写每次不同）。做重构回归请用")
    print("   scripts/retrieval_baseline.py（零 LLM、确定性、Δ 必须为 0）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
