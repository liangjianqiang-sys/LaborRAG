"""Golden Set 评估脚本：支持 Smoke(5题) / Full(20题) 双轨评估。

用法：
  python run_eval.py                          # 默认 Smoke(5题)
  python run_eval.py --subset full            # Full(20题)
  python run_eval.py --subset smoke --label v5
  python run_eval.py --difficulty hard        # 只跑 hard 题
  python run_eval.py --type calculate         # 只跑计算类
"""
import argparse
import json
import time
import requests

BASE = "http://localhost:8000/api/v1"


def run_eval(subset: str, label: str, qtype: str, difficulty: str):
    """触发评估并等待完成。"""
    params = {}
    if subset:
        params["eval_subset"] = subset
    if label:
        params["rag_mode"] = label
    if qtype:
        params["question_type"] = qtype
    if difficulty:
        params["difficulty"] = difficulty

    print(f"[run_eval] 启动评估: subset={subset}, label={label}, type={qtype}, difficulty={difficulty}")
    r = requests.post(f"{BASE}/evaluation/run", params=params, timeout=15)
    if r.status_code != 200:
        print(f"[run_eval] 启动失败: {r.text}")
        return

    print("[run_eval] 评估已启动，等待完成...")
    while True:
        time.sleep(5)
        status = requests.get(f"{BASE}/evaluation/status", timeout=10).json()
        if status.get("error"):
            print(f"[run_eval] 评估失败: {status['error']}")
            return
        if status.get("result"):
            result = status["result"]
            scores = result.get("scores", {})
            retrieval = result.get("retrieval", {})
            response = result.get("response", {})
            diff_scores = result.get("difficulty_scores", {})

            print("\n" + "=" * 60)
            print(f"  Golden Set 评估结果 ({subset})")
            print("=" * 60)
            print(f"  Faithfulness:     {scores.get('faithfulness', 'N/A')}")
            print(f"  Context Precision:{scores.get('context_precision', 'N/A')}")
            print(f"  Context Recall:   {scores.get('context_recall', 'N/A')}")
            print(f"  幻觉率:           {response.get('hallucination_rate', 'N/A')}")
            print(f"  完整性:           {response.get('completeness', 'N/A')}")
            # ⚠️ 直接遍历响应里的键，**不要硬编码指标名**。
            # 这里曾写成 retrieval.get("P@1") / ("R@5") / ("MRR")，而指标层的实际键是
            # precision@1 / recall@5 / mrr —— 于是**检索指标永远显示 N/A**，
            # 而数据其实就躺在同一个响应体里。键名写错不报错、不抛异常，只静默丢数据，
            # 看输出的人会以为「检索指标没算出来」，进而去查错方向。
            # 遍历的额外好处：指标层将来新增指标会自动出现在报告里。
            print("  检索指标:")
            for key, value in retrieval.items():
                if key == "per_query":  # 逐题明细，汇总时跳过
                    continue
                print(f"    {key:<14}{value}")

            if diff_scores:
                print("\n  按难度分组:")
                for diff, ds in diff_scores.items():
                    print(f"    {diff}: faithfulness={ds.get('faithfulness', 'N/A')} (n={ds.get('count', '?')})")

            print("=" * 60)
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden Set 评估脚本")
    parser.add_argument("--subset", default="smoke", choices=["smoke", "full"],
                        help="评估子集: smoke(5题) / full(20题)")
    parser.add_argument("--label", default="", help="评估标签")
    parser.add_argument("--type", default="", choices=["retrieve", "calculate", "compare"],
                        help="问题类型过滤")
    parser.add_argument("--difficulty", default="", choices=["easy", "medium", "hard"],
                        help="难度过滤")
    args = parser.parse_args()
    run_eval(args.subset, args.label, args.type, args.difficulty)
