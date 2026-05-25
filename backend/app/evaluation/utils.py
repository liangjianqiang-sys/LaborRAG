"""评估体系公共工具函数。"""
import math
from typing import Any


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
