"""离线检索基线：零 LLM、确定性、可复现的检索质量度量。

为什么需要它
------------
目录重构（移动文件 + 重写 import）与去重（合并多份重复实现）都声称"零行为变化"，
但现有测试覆盖不到检索层 —— 全绿不等于没坏。本脚本在重构**前后各跑一次**，
逐项比对六个指标，把"行为未变"从口头承诺变成可核验的数字。

它调用的是**真实生产节点**，不是复刻版
------------------------------------------
早期版本直接调 `retriever.retrieve()`，只覆盖了 FAISS + BM25 + RRF 这一层，
把整个知识图谱层漏掉了。而概念映射注入（`concept_lookup` + 强制置顶 0.95）
恰恰是 P@1 的主要来源 —— 实测「没签合同被辞了」经 `concept_lookup`
可精确命中 `劳动合同法第82条`，绕过它则整题判 MISS。用那版基线做回归，
重构把 `law_graph.py` 改坏了也不会报警。

因此本脚本直接调用 `app.agent.nodes.retrieve.retrieve_docs`
（生产检索节点本身）与 `classify_complexity`（纯规则复杂度分类），
覆盖范围即生产链路中**不依赖 LLM 的全部环节**：

  已覆盖：复杂度分类 → 向量+BM25+RRF 融合 → 父子分块提升
          → 知识图谱伴生法条注入 → 概念映射提升/注入并置顶 → 截断 → 兜底
  未覆盖：LLM 查询改写（用原问题代替）、HyDE、多查询扩展、生成、RAGAS

刻意只测检索、不测生成：生成要调 LLM，不确定、慢、还受网络影响，
无法作为回归基准。上表中未覆盖的几项同样依赖 LLM，故一律关闭 ——
传入的 `_NoLLM` 会让 HyDE 与多查询扩展在 try/except 中降级为"不做增强"。

用法：
    cd backend
    python scripts/retrieval_baseline.py                        # 跟随 .env 的 RETRIEVER_TYPE
    python scripts/retrieval_baseline.py --retriever reranked   # 消融对比
    python scripts/retrieval_baseline.py -o baselines/pre_refactor.json
    # 重构后：跑一次并与重构前比对
    python scripts/retrieval_baseline.py --compare baselines/pre_refactor.json \
        -o baselines/post_refactor.json
    python scripts/retrieval_baseline.py --verbose              # 显示节点内部日志
"""
import argparse
import asyncio
import contextlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.evaluation.datasets.golden_set import EVAL_DATASET  # noqa: E402
from app.evaluation.metrics.retrieval import (  # noqa: E402
    _normalize_article,
    average_precision,
    extract_article_id,
    mrr,
    precision_at_k,
    recall_at_k,
)

METRIC_KEYS = ["precision@1", "precision@3", "precision@5", "recall@5", "mrr", "map"]


class _NoLLM:
    """占位 LLM：任何形式的调用都抛错。

    `retrieve_docs` 里两处依赖 LLM 的增强（HyDE、多查询扩展）都有 try/except
    兜底，抛错即降级为"不做增强"。这保证基线是纯确定性、零成本、零网络的。
    """

    def __getattr__(self, name):
        def _boom(*args, **kwargs):
            raise RuntimeError("retrieval_baseline: 基线不调用 LLM")

        return _boom


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BACKEND_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def select_items(subset: str) -> list[dict]:
    if subset == "all":
        return list(EVAL_DATASET)
    return [item for item in EVAL_DATASET if subset in item.get("eval_subset", [])]


def score_one(retrieved_ids: list[str], relevant_ids: list[str]) -> dict:
    return {
        "precision@1": round(precision_at_k(retrieved_ids, relevant_ids, 1), 4),
        "precision@3": round(precision_at_k(retrieved_ids, relevant_ids, 3), 4),
        "precision@5": round(precision_at_k(retrieved_ids, relevant_ids, 5), 4),
        "recall@5": round(recall_at_k(retrieved_ids, relevant_ids, 5), 4),
        "mrr": round(mrr(retrieved_ids, relevant_ids), 4),
        "map": round(average_precision(retrieved_ids, relevant_ids), 4),
    }


async def run_one(retriever, item: dict, verbose: bool) -> dict:
    """跑一题：走生产节点的确定性路径，返回该题的检索结果与指标。"""
    from app.agent.nodes.classify import classify_complexity
    from app.agent.nodes.retrieve import retrieve_docs

    question = item["question"]

    # 离线无 LLM 改写，rewritten_question 退化为原问题；
    # 预置 sub_queries 使节点跳过 _expand_queries
    state = {
        "question": question,
        "rewritten_question": question,
        "sub_queries": [question],
        "steps": [],
    }
    state.update(classify_complexity(state))

    buf = io.StringIO()
    if verbose:
        docs = (await retrieve_docs(retriever, _NoLLM(), state))["context_docs"]
    else:
        # 节点的 [Agent] 日志会淹没进度条，默认吞掉（--verbose 可放出）
        with contextlib.redirect_stdout(buf):
            docs = (await retrieve_docs(retriever, _NoLLM(), state))["context_docs"]

    retrieved_ids = []
    for doc, _score in docs:
        aid = extract_article_id(doc.metadata.get("source", ""), doc.page_content)
        if aid:
            retrieved_ids.append(aid)

    relevant_ids = [_normalize_article(a) for a in item.get("relevant_articles", [])]
    metrics = score_one(retrieved_ids, relevant_ids)

    return {
        "question": question,
        "question_type": item.get("question_type"),
        "difficulty": item.get("difficulty"),
        "complexity": state.get("complexity"),
        "relevant_ids": relevant_ids,
        "retrieved_ids": retrieved_ids,
        "hit@1": metrics["precision@1"],
        "retrieved_count": len(docs),
        "metrics": metrics,
    }


async def run_all(retriever, items: list[dict], verbose: bool) -> list[dict]:
    results = []
    for idx, item in enumerate(items, 1):
        r = await run_one(retriever, item, verbose)
        results.append(r)
        mark = "OK " if r["metrics"]["precision@1"] else "MISS"
        print(f"  [{idx:>2}/{len(items)}] {mark} {r['question'][:34]}")
    return results


def print_diff(old_path: Path, payload: dict) -> None:
    """与一份历史基线逐项比对，并列出 rank-1 命中发生变化的题目。"""
    try:
        old = json.loads(old_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"\n[compare] 读取 {old_path} 失败：{e}")
        return

    old_agg, new_agg = old.get("aggregate", {}), payload["aggregate"]
    print(f"\n{'指标':<14}{'旧':>10}{'新':>10}{'Δ':>10}")
    for key in METRIC_KEYS:
        a, b = old_agg.get(key), new_agg.get(key)
        if a is None:
            continue
        delta = b - a
        flag = "" if abs(delta) < 1e-9 else ("  ⬆" if delta > 0 else "  ⬇")
        print(f"  {key:<12}{a:>10.4f}{b:>10.4f}{delta:>+10.4f}{flag}")

    old_q = {q["question"]: q for q in old.get("per_question", [])}
    changed = [
        (q["question"], old_q[q["question"]]["hit@1"], q["hit@1"])
        for q in payload["per_question"]
        if q["question"] in old_q and old_q[q["question"]]["hit@1"] != q["hit@1"]
    ]
    if changed:
        print(f"\nrank-1 命中变化（{len(changed)} 题）：")
        for question, before, after in changed:
            arrow = "MISS→OK" if after > before else "OK→MISS"
            print(f"  {arrow}  {question}")
    else:
        print("\nrank-1 命中：逐题一致")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线检索基线")
    parser.add_argument(
        "--retriever",
        choices=["vector", "hybrid", "reranked"],
        default=None,
        help="覆盖 .env 的 RETRIEVER_TYPE，用于消融对比",
    )
    parser.add_argument("--subset", choices=["smoke", "full", "all"], default="full")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 settings.TOP_K")
    parser.add_argument("-o", "--output", default="baselines/pre_refactor.json")
    parser.add_argument(
        "--compare",
        default=None,
        help="与一份历史基线比对，打印六个指标的 Δ 与 rank-1 变化题目",
    )
    parser.add_argument("--verbose", action="store_true", help="显示节点内部 [Agent] 日志")
    args = parser.parse_args()

    if args.retriever:
        settings.RETRIEVER_TYPE = args.retriever
    if args.top_k:
        settings.TOP_K = args.top_k

    items = select_items(args.subset)
    if not items:
        print(f"数据集为空（subset={args.subset}）")
        return 1

    print(f"检索器      : {settings.RETRIEVER_TYPE}")
    print(f"TOP_K       : {settings.TOP_K}")
    print(f"向量库      : {settings.VECTOR_STORE_PATH}")
    print(f"题量        : {len(items)}（subset={args.subset}）")
    print("\n加载向量库与检索器（首次需加载 BGE-M3，请稍候）...")

    started = time.perf_counter()
    from app.services.rag_engine import RAGEngine

    engine = RAGEngine()
    if not engine.initialize():
        print("向量库初始化失败：请先在 backend/ 下构建知识库")
        return 1
    retriever = engine.retriever
    print(f"就绪，用时 {time.perf_counter() - started:.1f}s\n")

    per_question = asyncio.run(run_all(retriever, items, args.verbose))

    aggregate = {
        key: round(sum(q["metrics"][key] for q in per_question) / len(per_question), 4)
        for key in METRIC_KEYS
    }

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "pipeline": "classify_complexity + retrieve_docs（零 LLM 确定性路径）",
            "covered": [
                "复杂度分类（纯规则）",
                "向量 + BM25 + RRF 融合",
                "父子分块提升",
                "知识图谱伴生法条注入",
                "概念映射提升/注入并置顶",
                "截断与兜底",
            ],
            "not_covered": ["LLM 查询改写", "HyDE", "多查询扩展", "生成", "RAGAS"],
            "retriever": settings.RETRIEVER_TYPE,
            "top_k": settings.TOP_K,
            "score_threshold": settings.SCORE_THRESHOLD,
            "subset": args.subset,
            "dataset_size": len(items),
            "vector_store": settings.VECTOR_STORE_PATH,
        },
        "aggregate": aggregate,
        "per_question": per_question,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = BACKEND_DIR / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n{'指标':<14}{'值':>8}")
    for key in METRIC_KEYS:
        print(f"  {key:<12}{aggregate[key]:>8.4f}")
    print(f"\n基线已写入 {out_path}")

    if args.compare:
        cmp_path = Path(args.compare)
        if not cmp_path.is_absolute():
            cmp_path = BACKEND_DIR / cmp_path
        print_diff(cmp_path, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
