"""LLM 调用点的统一约定：必须**无条件**关闭思考模式。

为什么需要这条规则
------------------
辅助/评估槽要的是「一个词」「JSON」「是否忠于资料」这类短结构化输出，
思考模式对它们是纯浪费。实测（2026-09-17，RAGAS 槽 = deepseek-v4.1-flash）：

    无 extra_body         : 2.6s，输出 115 tok，其中 **113 tok 是思考**（98%）
    enable_thinking=False : 0.9s，输出   1 tok，答案仍正确

而 `runner.py` 曾写成按模型名判断：

    extra_body = {"enable_thinking": False} if "qwen" in model_name else {}

理由写的是「DeepSeek 等其他模型不需要」—— 实测正好推翻。
**这类「按模型名猜行为」的写法会在换模型时静默失效**：功能照常工作、
测试照常通过，只是每次调用白烧 98% 的输出 token。

本文件用 AST 扫描所有 `extra_body=` 调用点，要求：
  1. 值必须是 **dict 字面量**，且含 `enable_thinking: False`
  2. **不得是条件表达式** —— 条件式正是当初出问题的形态
并额外扫描「按模型名判断行为」的写法。
"""
import ast
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BACKEND_DIR / "app"


def _iter_extra_body():
    """产出所有 `extra_body=` 关键字参数的 (文件, 行号, 值节点)。"""
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "extra_body":
                        yield path, node.lineno, kw.value


def test_存在_extra_body_调用点_防扫描器失效():
    sites = list(_iter_extra_body())
    assert sites, "一个 extra_body 调用点都没找到 —— 扫描器失效了，下面的用例是空跑"


def test_所有_extra_body_都是无条件关闭思考():
    problems = []
    for path, lineno, value in _iter_extra_body():
        rel = path.relative_to(BACKEND_DIR).as_posix()

        if isinstance(value, ast.IfExp):
            problems.append(f"{rel}:{lineno} 用了条件表达式 —— 换模型会静默失效")
            continue
        if not isinstance(value, ast.Dict):
            problems.append(f"{rel}:{lineno} 不是 dict 字面量（无法静态校验）")
            continue

        keys = [k.value for k in value.keys if isinstance(k, ast.Constant)]
        if "enable_thinking" not in keys:
            problems.append(f"{rel}:{lineno} 缺少 enable_thinking")
            continue
        for k, v in zip(value.keys, value.values):
            if isinstance(k, ast.Constant) and k.value == "enable_thinking":
                if not (isinstance(v, ast.Constant) and v.value is False):
                    problems.append(f"{rel}:{lineno} enable_thinking 的值不是 False")

    assert not problems, "LLM 调用点的关思考配置有问题：\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("marker", ['"qwen"', "'qwen'"])
def test_没有按模型名猜行为的写法(marker):
    """扫描模型名判断 —— 当初的 bug 就长这样。

    只匹配**独立成串**的模型名（`"qwen"`），不会误伤 `"qwen3.7-max"` 这类具体型号。
    """
    hits = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if marker in line:
                hits.append(f"{path.relative_to(BACKEND_DIR).as_posix()}:{i}  {line.strip()[:70]}")
    assert not hits, (
        "出现了按模型名判断行为的写法 —— 换模型时会静默失效，"
        "应当改成对所有模型统一的配置：\n  " + "\n  ".join(hits)
    )
