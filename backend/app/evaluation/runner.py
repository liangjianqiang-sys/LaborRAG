"""评估执行器：三阶段执行流程（生成答案 → 检索指标 → RAGAS+响应指标）。

三元组评估体系：
  - triad.context_relevancy:  上下文相关性（检索质量）
  - triad.faithfulness:       忠实度（生成质量）
  - triad.answer_relevancy:   答案相关性（端到端）
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from app.evaluation.datasets.golden_set import EVAL_DATASET
from app.evaluation.metrics.retrieval import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_by_type,
)
from app.evaluation.metrics.response import (
    compute_response_metrics,
    compute_response_metrics_by_type,
)
from app.evaluation.utils import sanitize_floats, strip_per_query, parallel_map
from app.core.config import settings
from app.knowledge.embeddings import get_embeddings

_SKIP_COLS = frozenset(('question', 'answer', 'contexts', 'ground_truth', 'question_type'))

# 中间结果保存目录（相对于backend根目录）
_BACKEND_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CACHE_DIR = os.path.join(_BACKEND_ROOT, "evaluation_reports", "cache")


def _mean_scores(df, skip_extra=frozenset()) -> Dict[str, float]:
    """从DataFrame提取指标列均值。"""
    scores = {}
    skip = _SKIP_COLS | skip_extra
    for col in df.columns:
        if col in skip:
            continue
        try:
            scores[col] = round(float(df[col].mean()), 4)
        except (TypeError, ValueError):
            continue
    return scores


class EvalRunner:
    """评估执行器，对RAG系统进行三元组量化评估。"""

    # RAGAS 阶段的最大重试次数（**不是**单次 LLM 调用的重试）。
    #
    # 2026-10-06 由 3 改为 1。理由：
    #
    # ① 成功路径完全不受影响 —— 成功时 `return` 在 try 块内，第一次就返回，
    #    MAX_RETRIES=1 与 =3 都是「只跑一次」。
    # ② 这个重试是**整轮重跑整个 Phase 3**（两个 evaluate() 全跑一遍），
    #    而不是只重试失败的那一项。按 20 题估算，一次重跑 ≈ 330k token；
    #    配 30s/60s 退避，失败时要多烧 3 倍额度 + 90 秒等待。
    # ③ 它能拦住的失败（配额耗尽、模型 400）**重试也没用**；而真正的瞬时故障
    #    （网络抖动、限流）已由内层 `RunConfig(max_retries=3, max_wait=300)`
    #    兜住 —— 改这里不影响那一层保护。
    #
    # 全失败时返回空结果 `{}, {}, []`，报告里 RAGAS 三项缺失；但 Phase 1/2
    # 的答案与检索指标已由 `_save_cache` 落盘，不会白跑，手动重跑即可。
    MAX_RETRIES = 1

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        # 关闭思考模式 —— **无条件传，不要按模型名做条件判断**。
        #
        # 这里原本写成「仅当模型名以 qwen 开头时才关思考」，理由是「DeepSeek 等
        # 其他模型不需要」。实测推翻了这个假设（2026-09-17，RAGAS 槽 =
        # deepseek-v4.1-flash）：
        #     不传 extra_body        : 2.6s，输出 115 tok，其中 **113 tok 是思考**（98%）
        #     enable_thinking=False  : 0.9s，输出 **1** tok，答案仍正确
        # 即非 qwen 模型同样默认满血思考（该模型的 reasoning_effort 默认 max）。
        # RAGAS 每题要发几十次调用，这个浪费会被放大成千上万倍。
        #
        # 另外三处调用点（agent/graph.py ×2、main.py、services/rag_engine.py）
        # 本来就是无条件传的，此处保持一致。
        # 回归守卫见 tests/test_llm_call_sites.py。
        from langchain_openai import ChatOpenAI  # 见文件顶部 TYPE_CHECKING 处的说明

        base_llm = ChatOpenAI(
            model=settings.RAGAS_MODEL_NAME,
            openai_api_key=settings.RAGAS_API_KEY,
            openai_api_base=settings.RAGAS_API_BASE,
            temperature=0,
            max_tokens=4096,
            request_timeout=180,
            extra_body={"enable_thinking": False},
        )
        # RAGAS 0.4.3: 直接传 ChatOpenAI，不做 n=1 包装。
        # 原因：RAGAS 内部用 LangchainLLMWrapper 调 agenerate_prompt，而 n=1
        # 包装器只覆写同步 _generate，异步调用会失败。当前 metrics 不含
        # answer_relevancy（唯一会请求 n>1 的指标），故无需包装。
        # 历史：曾有一个 `_N1ChatModel(BaseChatModel)` 实现，已随死代码清理删除；
        # 它同时是「模块级 import BaseChatModel」的唯一原因，而那次导入实测 15s。
        # 当前metrics不含answer_relevancy（不需要n>1），无需n=1包装
        self.ragas_llm = base_llm
        self.ragas_embeddings = get_embeddings()

    def run(self, sample_count: int = None, rag_mode: str = None, question_type: str = None,
            eval_subset: str = None, difficulty: str = None) -> Dict[str, Any]:
        """运行评估（三阶段：生成答案 → 检索指标 → RAGAS+响应指标）。

        Args:
            sample_count: 评估样本数量，None表示全部
            rag_mode: 评估标签（仅用于报告标识，不影响引擎行为）
            question_type: 过滤问题类型(retrieve/calculate/compare/out_of_scope)，None表示全部
            eval_subset: 子集过滤(smoke/full)，None表示全部
            difficulty: 难度过滤(easy/medium/hard)，None表示全部
        """
        dataset = EVAL_DATASET
        if question_type:
            dataset = [d for d in dataset if d.get("question_type") == question_type]
        if eval_subset:
            dataset = [d for d in dataset if eval_subset in d.get("eval_subset", ["full"])]
        if difficulty:
            dataset = [d for d in dataset if d.get("difficulty") == difficulty]
        dataset = dataset[:sample_count] if sample_count else dataset

        return self._execute(dataset, rag_mode or "agent")

    def _gen_answer(self, item: dict) -> dict:
        """单题答案生成（供并发调用）。返回结果dict，异常时返回None。"""
        import asyncio
        from app.schemas.chat import ChatRequest
        try:
            # 在 ThreadPoolExecutor 中需要新建事件循环，不能复用 uvicorn 的
            loop = asyncio.new_event_loop()
            try:
                response = loop.run_until_complete(
                    self.rag_engine.chat(ChatRequest(question=item["question"], skip_guardrails=True), use_reranker=True)
                )
            finally:
                loop.close()
            # 父块contexts（供Faithfulness评估）：完整法条，LLM基于此生成
            contexts = response.full_contexts if response.full_contexts else [src.content for src in response.sources]
            # 子块contexts（供Context Precision/Recall评估）：精准段落，反映检索器真实命中
            child_contexts = response.child_contexts if response.child_contexts else contexts
            return {
                "question": item["question"],
                "answer": response.answer,
                "contexts": contexts,
                "child_contexts": child_contexts,
                "ground_truth": item["ground_truth"],
                "relevant_articles": item.get("relevant_articles", []),
                "question_type": item.get("question_type", "retrieve"),
                "difficulty": item.get("difficulty", ""),
                "test_dimension": item.get("test_dimension", ""),
                "eval_subset": item.get("eval_subset", ["full"]),
                "law": item.get("law", ""),
                "sources": [{"source": s.source, "content": s.content} for s in response.sources],
                "source_count": len(response.sources),
                "rag_mode": "agent",
            }
        except Exception as e:
            print(f"  生成失败: {item['question'][:30]}... → {e}")
            return None

    def _execute(self, dataset: list, rag_mode_label: str) -> Dict[str, Any]:
        """三阶段执行核心逻辑。"""

        total = len(dataset)
        max_workers = settings.EVAL_MAX_CONCURRENT
        expected_questions = [d["question"] for d in dataset]

        # ── 阶段1：生成答案（token消耗60-70%，并发加速）──
        # 先尝试复用上次的缓存（2026-10-06 补）。Phase 1 是 20 题里最贵的一段
        # （约 230k token）；而 Phase 3 的 RAGAS 在内存紧张时容易崩 ——
        # 崩了却因为缓存「只写不读」而必须重跑 Phase 1，等于白烧一次。
        cached = self._load_cache(rag_mode_label, expected_questions)
        if cached is not None:
            print(f"[Phase 1] 命中缓存 {cached['_cache_file']}，**跳过生成**"
                  f"（复用 {len(cached['questions'])} 题的答案与上下文，省约 230k token）")
            questions = cached["questions"]
            answers = cached["answers"]
            contexts = cached["contexts"]
            ground_truths = cached["ground_truths"]
            child_contexts_list = cached["child_contexts"]
            sources_raw = cached["sources_raw"]
            details = cached["details"]
            # 这两个可从 details 还原，无需单独入缓存
            relevant_articles_list = [d.get("relevant_articles", []) for d in details]
            question_types = [d.get("question_type", "retrieve") for d in details]
            gen_results = None
        else:
            print(f"[Phase 1] Generating answers on {total} samples "
                  f"(mode={rag_mode_label}, concurrent={max_workers})...")

            gen_results = parallel_map(
                self._gen_answer, dataset,
                max_workers=max_workers, desc="Phase1",
            )

            # 按顺序收集成功结果，跳过失败题目
            questions, answers, contexts, ground_truths = [], [], [], []
            sources_raw: List[List[Dict]] = []
            relevant_articles_list: List[List[str]] = []
            question_types: List[str] = []
            details = []
            # 子块上下文：供 RAGAS 的 Context Precision/Recall（2026-10-06 起一并入缓存）
            child_contexts_list: List[List[str]] = []

            for i, result in enumerate(gen_results):
                if result is None or isinstance(result, Exception):
                    # 失败题目用占位数据，保证索引对齐
                    item = dataset[i]
                    questions.append(item["question"])
                    answers.append("")
                    contexts.append([])
                    ground_truths.append(item["ground_truth"])
                    relevant_articles_list.append(item.get("relevant_articles", []))
                    question_types.append(item.get("question_type", "retrieve"))
                    sources_raw.append([])
                    child_contexts_list.append([])
                    details.append({
                        "question": item["question"],
                        "question_type": item.get("question_type", "retrieve"),
                        "difficulty": item.get("difficulty", ""),
                        "test_dimension": item.get("test_dimension", ""),
                        "eval_subset": item.get("eval_subset", ["full"]),
                        "law": item.get("law", ""),
                        "answer": "", "ground_truth": item["ground_truth"],
                        "relevant_articles": item.get("relevant_articles", []),
                        "source_count": 0, "rag_mode": rag_mode_label,
                    })
                    continue
                questions.append(result["question"])
                answers.append(result["answer"])
                contexts.append(result["contexts"])
                ground_truths.append(result["ground_truth"])
                relevant_articles_list.append(result["relevant_articles"])
                question_types.append(result["question_type"])
                sources_raw.append(result["sources"])
                child_contexts_list.append(result.get("child_contexts") or result["contexts"])
                details.append({
                    "question": result["question"],
                    "question_type": result["question_type"],
                    "difficulty": result["difficulty"],
                    "test_dimension": result["test_dimension"],
                    "eval_subset": result["eval_subset"],
                    "law": result["law"],
                    "answer": result["answer"],
                    "ground_truth": result["ground_truth"],
                    "relevant_articles": result["relevant_articles"],
                    "source_count": result["source_count"],
                    "rag_mode": result["rag_mode"],
                })

            # 立即保存中间结果，评分失败时可直接重试
            self._save_cache(rag_mode_label, questions, answers, contexts, ground_truths, details,
                            sources_raw=sources_raw, child_contexts=child_contexts_list)
        print(f"[Phase 1] Done. Answers cached. Safe to retry if scoring fails.")

        # ── 阶段2：检索指标计算（零LLM调用）──
        print(f"[Phase 2] Computing retrieval metrics (zero LLM calls)...")
        retrieval = compute_retrieval_metrics(sources_raw, relevant_articles_list, k=5)
        type_retrieval = compute_retrieval_metrics_by_type(
            sources_raw, relevant_articles_list, question_types, k=5
        )
        # 将每题检索指标写入details
        for i, pq in enumerate(retrieval.get("per_query", [])):
            details[i]["retrieval"] = {k: v for k, v in pq.items() if k not in ("retrieved_ids", "relevant_ids")}
            details[i]["retrieved_ids"] = pq.get("retrieved_ids", [])
            details[i]["relevant_ids"] = pq.get("relevant_ids", [])
        print(f"[Phase 2] Done. retrieval={_summary(retrieval)}")

        # ── 阶段3：RAGAS双轨评分（token消耗30-40%，可重试）──
        # child_contexts 已在 Phase 1 一并收集（并随缓存落盘），此处不再重算。
        # 2026-10-06 改：此前这里无条件 `for i, result in enumerate(gen_results)`，
        # 一旦走缓存路径（gen_results 为 None）就会 TypeError；而且它收集的东西
        # 与 Phase 1 完全重复。合并到 Phase 1 后两条路径共用同一份数据。
        scores, type_scores, faithfulness_per_query = self._score_with_retry(
            questions, answers, contexts, child_contexts_list, ground_truths, details
        )

        # 响应指标（ROUGE-L/BLEU零成本，幻觉率复用faithfulness，完整性需LLM）
        print(f"[Phase 3] Computing response metrics...")
        # 拼接contexts为字符串列表，供完整性评估使用
        contexts_str_list = ["\n".join(ctx) if isinstance(ctx, list) else str(ctx) for ctx in contexts]
        response_metrics = compute_response_metrics(
            answers, ground_truths,
            faithfulness_scores=faithfulness_per_query,
            llm=self.ragas_llm,  # 复用评估LLM计算完整性
            contexts_list=contexts_str_list,
        )
        type_response = compute_response_metrics_by_type(
            answers, ground_truths, question_types,
            faithfulness_scores=faithfulness_per_query,
            llm=self.ragas_llm,
            contexts_list=contexts_str_list,
        )
        # 将每题faithfulness写入details
        for i, f in enumerate(faithfulness_per_query):
            details[i]["faithfulness"] = f
        # 将每题响应指标写入details
        for i, pq in enumerate(response_metrics.get("per_query", [])):
            details[i]["response"] = pq
        print(f"[Phase 3] Done. response={_summary(response_metrics)}")

        # ── 按难度分组统计 ──
        difficulty_scores = {}
        difficulty_retrieval = {}
        for diff in ("easy", "medium", "hard"):
            idxs = [i for i, d in enumerate(details) if d.get("difficulty") == diff]
            if not idxs:
                continue
            # 检索指标
            diff_sources = [sources_raw[i] for i in idxs]
            diff_articles = [relevant_articles_list[i] for i in idxs]
            difficulty_retrieval[diff] = strip_per_query(
                compute_retrieval_metrics(diff_sources, diff_articles, k=5)
            )
            # RAGAS指标
            if scores:
                diff_faith = [details[i].get("faithfulness") for i in idxs]
                diff_faith_vals = [v for v in diff_faith if v is not None]
                difficulty_scores[diff] = {
                    "faithfulness": round(sum(diff_faith_vals) / len(diff_faith_vals), 4) if diff_faith_vals else None,
                    "count": len(idxs),
                }

        # ── 构建核心指标 ──
        triad = {
            "faithfulness": scores.get("faithfulness") or 0.0,
        }

        return {
            "triad": triad,
            "retrieval": strip_per_query(retrieval),
            "response": strip_per_query(response_metrics),
            "scores": scores,
            "type_scores": type_scores,
            "type_retrieval": {k: strip_per_query(v) for k, v in type_retrieval.items()},
            "type_response": {k: strip_per_query(v) for k, v in type_response.items()},
            "difficulty_scores": difficulty_scores,
            "difficulty_retrieval": difficulty_retrieval,
            "rag_mode": rag_mode_label,
            "sample_count": total,
            "details": details,
        }

    def _score_with_retry(self, questions, answers, parent_contexts, child_contexts, ground_truths, details) -> tuple:
        """RAGAS双轨评分，带重试逻辑。返回 (scores, type_scores, faithfulness_per_query)。

        双轨策略：
        - Faithfulness用parent_contexts（完整法条，LLM基于此生成）
        - Context Precision/Recall用child_contexts（精准段落，反映检索器真实命中）
        """
        # 重依赖一律在函数内导入：datasets / ragas 顶层导入会连带 torch（实测该模块
        # 顶层导入约 48s）。本模块只有 api/evaluation.py 一个引用方且本就是延迟导入，
        # 但保持规则无例外，守卫测试才能用「零豁免」的形态（见 tests/test_heavy_imports.py）。
        from datasets import Dataset
        from ragas import evaluate, RunConfig
        from ragas.metrics import faithfulness, context_precision, context_recall

        # Faithfulness数据集：用父块contexts
        ds_faith = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": parent_contexts, "ground_truth": ground_truths,
        })
        # Precision/Recall数据集：用子块contexts
        ds_prec = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": child_contexts, "ground_truth": ground_truths,
        })

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"[Phase 3] Computing RAGAS metrics (attempt {attempt}/{self.MAX_RETRIES})...")
                run_config = RunConfig(
                    timeout=180,
                    max_retries=3,
                    max_wait=300,
                    max_workers=4,
                )

                # 1. Faithfulness（用父块contexts）
                print(f"[Phase 3]   Faithfulness: parent contexts ({len(parent_contexts)} items)...")
                faith_result = evaluate(
                    ds_faith, metrics=[faithfulness],
                    llm=self.ragas_llm, embeddings=self.ragas_embeddings,
                    batch_size=3,
                    run_config=run_config,
                )
                faith_df = faith_result.to_pandas()

                # 2. Context Precision/Recall（用子块contexts）
                print(f"[Phase 3]   Precision/Recall: child contexts ({len(child_contexts)} items)...")
                prec_result = evaluate(
                    ds_prec, metrics=[context_precision, context_recall],
                    llm=self.ragas_llm, embeddings=self.ragas_embeddings,
                    batch_size=3,
                    run_config=run_config,
                )
                prec_df = prec_result.to_pandas()

                # 合并结果
                scores = _mean_scores(faith_df)
                prec_scores = _mean_scores(prec_df)
                scores.update(prec_scores)

                type_scores = {}
                if any(d["question_type"] != "retrieve" for d in details):
                    faith_df["question_type"] = [d["question_type"] for d in details]
                    prec_df["question_type"] = [d["question_type"] for d in details]
                    for qtype in set(d["question_type"] for d in details):
                        ts = _mean_scores(faith_df[faith_df["question_type"] == qtype])
                        ts.update(_mean_scores(prec_df[prec_df["question_type"] == qtype]))
                        type_scores[qtype] = ts

                # 提取每题faithfulness（用于幻觉率计算）
                faithfulness_per_query = []
                if "faithfulness" in faith_df.columns:
                    for val in faith_df["faithfulness"]:
                        try:
                            faithfulness_per_query.append(float(val))
                        except (TypeError, ValueError):
                            faithfulness_per_query.append(None)

                return scores, type_scores, faithfulness_per_query
            except Exception as e:
                print(f"[Phase 3] RAGAS attempt {attempt} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    wait = attempt * 30
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[Phase 3] All {self.MAX_RETRIES} attempts failed. Returning partial results.")
                    return {}, {}, []

    def _save_cache(self, rag_mode_label, questions, answers, contexts, ground_truths, details,
                    sources_raw=None, child_contexts=None):
        """保存中间结果（答案生成后立即保存，评分失败可重试）。

        `sources_raw` / `child_contexts` 是 2026-10-06 补的 —— 此前只存了
        questions/answers/contexts/ground_truths/details，缺这两个就无法真正
        跳过 Phase 1：

        - `sources_raw` 供 Phase 2 算检索指标（`compute_retrieval_metrics`）
        - `child_contexts` 供 RAGAS 的 Context Precision/Recall

        缺它们的后果是 `_load_cache` 只能"读回一半数据"，Phase 1 还是得重跑 ——
        而 Phase 1 正是 20 题里最贵的部分（约 230k token）。
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(_CACHE_DIR, f"answers_{rag_mode_label}_{ts}.json")
        data = sanitize_floats({
            "rag_mode": rag_mode_label,
            "timestamp": ts,
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "child_contexts": child_contexts or [],
            "ground_truths": ground_truths,
            "sources_raw": sources_raw or [],
            "details": details,
        })
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Cache saved: {cache_file}")

    def _load_cache(self, rag_mode_label: str, expected_questions: List[str]) -> Optional[Dict[str, Any]]:
        """按 label 找最近一份**题目完全一致**的缓存，命中则返回其内容，否则 None。

        为什么要校验题目集
        ------------------
        缓存文件名只带 label 与时间戳，**不带筛选条件**。若上一次跑的是
        `--difficulty easy`（6 题）、这次跑 `--subset full`（20 题），
        直接复用就会把 6 题的答案当成 20 题的结果 —— 而且**不会报错**，
        只会产出一份看起来正常的错报告。

        所以这里要求 `questions` 列表**逐项完全相等**（含顺序）。不匹配就
        当作没有缓存，老老实实重跑。

        另外：2026-10-06 之前写的缓存没有 `sources_raw` / `child_contexts`，
        无法真正跳过 Phase 1，这里一并视为无效（宁可重跑，也不要半吊子复用）。
        """
        if not os.path.isdir(_CACHE_DIR):
            return None

        prefix = f"answers_{rag_mode_label}_"
        candidates = sorted(
            (f for f in os.listdir(_CACHE_DIR) if f.startswith(prefix) and f.endswith(".json")),
            reverse=True,   # 文件名含时间戳，倒序即最新优先
        )
        for name in candidates:
            path = os.path.join(_CACHE_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  [cache] 跳过损坏文件 {name}: {e}")
                continue

            if data.get("questions") != expected_questions:
                continue
            if not data.get("sources_raw") or not data.get("child_contexts"):
                print(f"  [cache] {name} 是旧格式（缺 sources_raw/child_contexts），不复用")
                continue

            data["_cache_file"] = name
            return data
        return None


def _summary(metrics: Dict) -> str:
    """生成指标摘要字符串，用于日志输出。"""
    items = [f"{k}={v}" for k, v in metrics.items() if k != "per_query" and isinstance(v, (int, float))]
    return "{" + ", ".join(items) + "}"
