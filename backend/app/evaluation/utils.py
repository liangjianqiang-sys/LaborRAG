"""评估体系公共工具函数。"""
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

# 注：file_lock 已下沉到 app.utils.files（通用文件写入原语，不属于评估层）。
# 它的旧位置在本模块，导致 app/core/vectorstore.py 反向依赖评估层。


def parallel_map(
    fn: Callable,
    items: List,
    max_workers: int = 4,
    desc: str = "",
) -> List:
    """并发执行 fn(item)，返回按原始顺序的结果列表。

    每个item独立执行，单题失败不阻塞其他。
    异常以 Exception 对象存入结果列表对应位置，由调用方处理。

    Args:
        fn: 单项处理函数
        items: 待处理列表
        max_workers: 最大并发线程数
        desc: 日志描述前缀
    """
    n = len(items)
    if n == 0:
        return []
    results: List[Optional[Any]] = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            done_count += 1
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = e
                if desc:
                    print(f"  [{desc}] item {idx} failed: {e}")
            if desc and done_count % max(1, n // 4) == 0:
                print(f"  [{desc}] {done_count}/{n} done")
    return results


def sanitize_floats(obj: Any) -> Any:
    """递归清洗NaN/inf浮点值为None，使其可JSON序列化。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj


def mean_of_key(dicts: list, key: str, round_digits: int = 4) -> float:
    """从字典列表中提取指定key的均值。"""
    values = [d[key] for d in dicts if key in d]
    return round(sum(values) / len(values), round_digits) if values else 0.0


def strip_per_query(metrics: dict) -> dict:
    """移除per_query字段，用于汇总输出。"""
    return {k: v for k, v in metrics.items() if k != "per_query"}
