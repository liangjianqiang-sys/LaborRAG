"""评估报告生成器：对比不同检索策略的评估结果。"""
import json
import os
from typing import Dict, Any, List
from datetime import datetime


class EvalReport:
    """评估报告，存储和对比多次评估结果。"""

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
            json.dump(data, f, ensure_ascii=False, indent=2)

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
                    results.append(self.load_result(filepath))
        return results

    def compare(self) -> Dict[str, Any]:
        """对比所有评估结果。"""
        results = self.get_all_results()
        if not results:
            return {"message": "暂无评估结果"}

        # 按retriever_type分组
        grouped = {}
        for r in results:
            rtype = r.get("retriever_type", "unknown")
            if rtype not in grouped or r["timestamp"] > grouped[rtype]["timestamp"]:
                grouped[rtype] = r

        # 生成对比表
        comparison = {
            "metrics": list(list(grouped.values())[0]["scores"].keys()),
            "results": {},
        }
        for rtype, data in grouped.items():
            comparison["results"][rtype] = data["scores"]

        return comparison
