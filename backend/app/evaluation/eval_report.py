"""评估报告生成器：支持三元组/检索/响应三维度对比。"""
import json
import os
from typing import Dict, Any, List
from datetime import datetime
from app.evaluation.utils import sanitize_floats


class EvalReport:
    """评估报告，存储和对比多次评估结果，支持三元组/检索/响应三维度。"""

    def __init__(self, report_dir: str = None):
        self.report_dir = report_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "evaluation_reports"
        )
        os.makedirs(self.report_dir, exist_ok=True)

    def save_result(self, retriever_type: str, result: Dict[str, Any]) -> str:
        """保存评估结果到文件。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{retriever_type}_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)

        data = {
            "retriever_type": retriever_type,
            "timestamp": timestamp,
            **result,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sanitize_floats(data), f, ensure_ascii=False, indent=2)

        return filepath

    def load_result(self, filepath: str) -> Dict[str, Any]:
        """加载评估结果。"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_results(self) -> List[Dict[str, Any]]:
        """获取所有评估结果。"""
        results = []
        if os.path.exists(self.report_dir):
            for filename in sorted(os.listdir(self.report_dir)):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.report_dir, filename)
                    try:
                        results.append(self.load_result(filepath))
                    except (json.JSONDecodeError, OSError):
                        continue
        return results

    def compare(self) -> Dict[str, Any]:
        """对比所有评估结果（三维度：三元组 + 检索 + 响应）。"""
        results = self.get_all_results()
        if not results:
            return {"message": "暂无评估结果"}

        # 按retriever_type分组，保留每组最新结果
        grouped = {}
        for r in results:
            rtype = r.get("retriever_type", r.get("rag_mode", "unknown"))
            if rtype not in grouped or r["timestamp"] > grouped[rtype]["timestamp"]:
                grouped[rtype] = r

        # 构建三维度对比
        comparison = {
            "triad": {},
            "metrics": [],
            "results": {},
            "retrieval": {},
            "response": {},
        }

        # 收集所有metric key（兼容旧报告只有scores的情况）
        first_data = list(grouped.values())[0]
        if "scores" in first_data:
            comparison["metrics"] = list(first_data["scores"].keys())

        for rtype, data in grouped.items():
            # 三维度指标统一收集
            for dim in ("scores", "triad", "retrieval", "response"):
                if dim in data:
                    target_key = "results" if dim == "scores" else dim
                    comparison[target_key][rtype] = data[dim]

        return comparison

    def _get_latest(self) -> Dict[str, Any]:
        """获取最新一次评估结果，无结果返回空dict。"""
        results = self.get_all_results()
        return max(results, key=lambda r: r.get("timestamp", "")) if results else {}

    def get_bad_cases(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """获取得分最低的Bad Case列表。

        按响应指标综合分排序，返回得分最低的题目。
        """
        latest = self._get_latest()
        details = latest.get("details", [])
        if not details:
            return []

        # 计算每题综合分
        for d in details:
            resp = d.get("response", {})
            d["_score"] = (
                resp.get("rouge_l", 0) * 0.3
                + resp.get("completeness", 0) * 0.4
                + (1 - (resp.get("hallucination_rate") or 1)) * 0.3
            ) if resp else 0.5

        bad_cases = sorted(details, key=lambda d: d.get("_score", 0))[:top_n]
        for d in bad_cases:
            d.pop("_score", None)
        return bad_cases

    def get_observability(self) -> Dict[str, Any]:
        """获取可观测性三维度概览（检索/生成/业务）。"""
        latest = self._get_latest()
        if not latest:
            return {"message": "暂无评估结果"}

        # 检索质量
        ret = latest.get("retrieval", {})
        retrieval_quality = {
            "precision@1": ret.get("precision@1", 0),
            "precision@5": ret.get("precision@5", 0),
            "recall@5": ret.get("recall@5", 0),
            "mrr": ret.get("mrr", 0),
            "corpus_coverage": latest.get("scores", {}).get("context_recall", 0),
        }

        # 生成质量
        triad = latest.get("triad", {})
        resp = latest.get("response", {})
        generation_quality = {
            "faithfulness": triad.get("faithfulness", 0),
            "hallucination_rate": resp.get("hallucination_rate", 0),
            "completeness": resp.get("completeness", 0),
        }

        # 业务指标
        details = latest.get("details", [])
        total = len(details)
        business = {
            "resolution_rate": round(sum(1 for d in details if d.get("answer", "").strip()) / total, 4) if total else 0,
            "first_answer_usability": round(sum(1 for d in details if "警告" not in d.get("answer", "") and "无法" not in d.get("answer", "")) / total, 4) if total else 0,
        }

        return {
            "retrieval_quality": retrieval_quality,
            "generation_quality": generation_quality,
            "business": business,
        }
