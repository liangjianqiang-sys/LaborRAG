"""evaluation.utils 公共工具的单测。

纯函数模块（sanitize_floats / mean_of_key / strip_per_query / parallel_map），
此前零测试覆盖。其中 sanitize_floats 负责把 NaN/inf 清洗成 None：
若漏掉，下游 json.dump 会抛 ValueError（NaN 不是合法 JSON 数值），
而这类问题只会在「恰好出现一个 NaN 指标」时才暴露 —— 典型静默边界。
"""
from app.evaluation.utils import (
    sanitize_floats,
    mean_of_key,
    strip_per_query,
    parallel_map,
)


def test_sanitize_floats_非浮点原样保留():
    assert sanitize_floats(3) == 3           # int 不是 float
    assert sanitize_floats("str") == "str"
    assert sanitize_floats(None) is None


def test_sanitize_floats_正常浮点保留():
    assert sanitize_floats(1.5) == 1.5
    assert sanitize_floats(0.0) == 0.0


def test_sanitize_floats_nan_inf_转_None():
    assert sanitize_floats(float("nan")) is None
    assert sanitize_floats(float("inf")) is None
    assert sanitize_floats(float("-inf")) is None


def test_sanitize_floats_递归清洗嵌套结构():
    dirty = {
        "a": float("nan"),
        "b": 1.5,
        "c": [float("inf"), {"d": float("-inf")}],
    }
    assert sanitize_floats(dirty) == {"a": None, "b": 1.5, "c": [None, {"d": None}]}


def test_mean_of_key():
    assert mean_of_key([{"a": 1}, {"a": 3}], "a") == 2.0
    assert mean_of_key([{"a": 1}, {"b": 99}], "a") == 1.0  # 缺键跳过
    assert mean_of_key([], "a") == 0.0


def test_strip_per_query():
    assert strip_per_query({"per_query": [1, 2], "precision@1": 0.9}) == {"precision@1": 0.9}


def test_parallel_map_顺序正确():
    assert parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=2) == [2, 4, 6]
    assert parallel_map(lambda x: x, []) == []


def test_parallel_map_单题异常不阻塞其他():
    def f(x):
        if x == 2:
            raise ValueError("boom")
        return x * 10

    results = parallel_map(f, [1, 2, 3], max_workers=2)
    assert results[0] == 10
    assert isinstance(results[1], ValueError)  # 异常以 Exception 对象存入对应位置
    assert results[2] == 30
