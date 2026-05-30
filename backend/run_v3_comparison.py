"""V1-V4对比实验脚本：运行四组RAGAS评估并对比结果。

用法：
    cd backend
    python run_v3_comparison.py              # 运行全部四组
    python run_v3_comparison.py --only agent # 只运行V4(Agentic RAG)
    python run_v3_comparison.py --only vector crag agent  # 指定多组

实验配置：
    V1: RETRIEVER_TYPE=vector, RAG_MODE=simple
    V2: RETRIEVER_TYPE=reranked, RAG_MODE=simple
    V3: RETRIEVER_TYPE=reranked, RAG_MODE=crag
    V4: RETRIEVER_TYPE=reranked, RAG_MODE=agent
"""
import os
import sys
import json
import argparse
import time

# 确保backend目录在路径中
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

# 加载.env
from dotenv import load_dotenv
load_dotenv()

from app.config import settings
from app.core.rag_engine import RAGEngine
from app.evaluation.eval_runner import EvalRunner
from app.evaluation.eval_report import EvalReport


# 三组实验配置
EXPERIMENTS = {
    "vector": {
        "label": "V1 纯向量检索",
        "RETRIEVER_TYPE": "vector",
        "RAG_MODE": "simple",
    },
    "reranked": {
        "label": "V2 混合检索+重排序",
        "RETRIEVER_TYPE": "reranked",
        "RAG_MODE": "simple",
    },
    "crag": {
        "label": "V3 CRAG自我纠错",
        "RETRIEVER_TYPE": "reranked",
        "RAG_MODE": "crag",
    },
    "agent": {
        "label": "V4 Agentic RAG",
        "RETRIEVER_TYPE": "reranked",
        "RAG_MODE": "agent",
    },
}


def run_experiment(exp_key: str, exp_config: dict, sample_count: int = None) -> dict:
    """运行单组实验。"""
    print(f"\n{'='*60}")
    print(f"  实验: {exp_config['label']}")
    print(f"  RETRIEVER_TYPE={exp_config['RETRIEVER_TYPE']}, RAG_MODE={exp_config['RAG_MODE']}")
    print(f"{'='*60}")

    # 临时修改settings
    original_retriever = settings.RETRIEVER_TYPE
    original_rag_mode = settings.RAG_MODE

    settings.RETRIEVER_TYPE = exp_config["RETRIEVER_TYPE"]
    settings.RAG_MODE = exp_config["RAG_MODE"]

    try:
        # 重建RAG引擎（使用新配置）
        engine = RAGEngine()
        if not engine.initialize():
            print("  ❌ 引擎初始化失败！跳过此实验。")
            return None

        # 运行评估
        runner = EvalRunner(engine)
        start_time = time.time()
        result = runner.run(sample_count=sample_count)
        elapsed = time.time() - start_time

        # 保存结果（用 exp_key 作为标识）
        report = EvalReport()
        filepath = report.save_result(exp_key, result)

        print(f"  ✅ 完成！耗时 {elapsed:.1f}s")
        print(f"  报告: {filepath}")
        for metric, score in result["scores"].items():
            print(f"    {metric}: {score}")

        return {
            "key": exp_key,
            "label": exp_config["label"],
            "scores": result["scores"],
            "sample_count": result["sample_count"],
            "elapsed": round(elapsed, 1),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ❌ 实验失败: {e}")
        return None
    finally:
        # 恢复原始配置
        settings.RETRIEVER_TYPE = original_retriever
        settings.RAG_MODE = original_rag_mode


def print_comparison(results: list):
    """打印对比表格。"""
    if not results:
        print("\n没有成功的实验结果。")
        return

    print(f"\n{'='*70}")
    print("  📊 V1 vs V2 vs V3 vs V4 对比实验结果")
    print(f"{'='*70}")

    # 收集所有指标
    all_metrics = []
    for r in results:
        for m in r["scores"]:
            if m not in all_metrics:
                all_metrics.append(m)

    # 表头
    header = f"{'指标':<25}"
    for r in results:
        header += f" {r['label']:>18}"
    print(header)
    print("-" * len(header))

    # 每行指标
    for metric in all_metrics:
        row = f"{metric:<25}"
        for r in results:
            score = r["scores"].get(metric, "N/A")
            if isinstance(score, float):
                row += f" {score:>18.4f}"
            else:
                row += f" {str(score):>18}"
        print(row)

    # 耗时
    row = f"{'耗时(秒)':<25}"
    for r in results:
        row += f" {r['elapsed']:>18.1f}"
    print(row)

    print(f"{'='*70}")

    # 分析
    if len(results) >= 2:
        print("\n📈 分析：")
        for metric in all_metrics:
            best = max(results, key=lambda r: r["scores"].get(metric, 0))
            worst = min(results, key=lambda r: r["scores"].get(metric, 0))
            best_val = best["scores"].get(metric, 0)
            worst_val = worst["scores"].get(metric, 0)
            if isinstance(best_val, (int, float)) and isinstance(worst_val, (int, float)):
                diff = best_val - worst_val
                print(f"  {metric}: {best['label']} 最优 ({best_val:.4f}), "
                      f"领先 {worst['label']} +{diff:.4f}")


def main():
    parser = argparse.ArgumentParser(description="V1/V2/V3对比实验")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS.keys()),
                        help="只运行指定实验，如 --only crag")
    parser.add_argument("--sample", type=int, default=None,
                        help="评估样本数量，默认全部")
    args = parser.parse_args()

    # 选择要运行的实验
    if args.only:
        exp_keys = args.only
    else:
        exp_keys = list(EXPERIMENTS.keys())

    print(f"🔬 将运行 {len(exp_keys)} 组实验: {', '.join(exp_keys)}")
    if args.sample:
        print(f"   每组样本数: {args.sample}")

    # 依次运行
    results = []
    for key in exp_keys:
        result = run_experiment(key, EXPERIMENTS[key], sample_count=args.sample)
        if result:
            results.append(result)

    # 打印对比
    print_comparison(results)

    # 保存对比结果
    if results:
        report = EvalReport()
        comparison_path = os.path.join(
            report.report_dir,
            f"comparison_v1v2v3_{int(time.time())}.json"
        )
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 对比结果已保存: {comparison_path}")


if __name__ == "__main__":
    main()
