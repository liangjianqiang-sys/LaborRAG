"""劳动法计算工具集 —— 供 LLM 通过 Function Calling 调用。

与 `labor_calculator.py` 的分工
--------------------------------
    labor_calculator.py   纯函数：入参是数字、出参是 dict，不含任何 LLM 概念
    labor_tools.py        工具封装：把纯函数暴露成模型可调用的工具，
                          并把 dict 结果渲染成模型易读的文本

即：`labor_calculator` 负责「怎么算」，本模块负责「怎么让模型调得动」。

分层约束
--------
本模块属 `tools` 层，**不得 import `app.agent` / `app.services`**
（`tests/test_layering.py` 用 AST 扫描，函数体内的延迟导入同样会被扫到）。
因此给模型的系统提示词不放在这里 —— 它属于 agent 层，见
`app/agent/nodes/calculate.py` 的 `_TOOL_CALL_SYSTEM_PROMPT`。

docstring 就是给模型看的工具描述
--------------------------------
`@tool` 会把 docstring 作为 tool description 发给模型，所以每个 docstring
都要回答「**什么时候该用这个工具**」，而不只是「它做什么」——
模型选错工具通常不是能力问题，而是描述里没写清适用边界。
"""
from typing import Any, Dict, List, Literal

from langchain_core.tools import tool

# 底层纯函数与工具同名，用别名导入避免遮蔽（工具对象在下面重新定义）
from app.tools.labor_calculator import (
    calculate_damage_pay as _calc_damage_pay,
    calculate_overtime_pay as _calc_overtime_pay,
    calculate_severance_pay as _calc_severance_pay,
)


# 计算函数返回的字段名 → 中文标签，渲染时按此顺序输出
_FIELD_LABELS = (
    ("monthly_salary", "月薪"),
    ("hourly_wage", "时薪"),
    ("overtime_hours", "加班小时数"),
    ("rate_type", "加班类型"),
    ("rate", "倍率"),
    ("working_years", "工作年限"),
    ("compensation_months", "补偿月数"),
    ("severance_base", "经济补偿金基数"),
)


def _render(result: Dict[str, Any]) -> str:
    """把计算函数的 dict 结果渲染成模型易读的文本。

    渲染成文本而不是回传 dict，是为了让模型整合回答时能**直接引用**
    公式、法条依据和代入过程，减少它自己复述时编造数字的空间。
    """
    lines = [
        f"【{result['type']}】",
        f"法条依据：{result['law_ref']}",
        f"计算公式：{result['formula']}",
    ]
    for key, label in _FIELD_LABELS:
        if key in result:
            lines.append(f"{label}：{result[key]}")
    lines.append(f"计算结果：{result['result']} 元")
    return "\n".join(lines)


# ─── 工具定义 ──────────────────────────────────────────────

@tool
def calculate_overtime_pay(
    monthly_salary: float,
    hours: float,
    rate_type: Literal["工作日延时", "休息日加班", "法定节假日"] = "工作日延时",
) -> str:
    """计算加班费金额。

    何时使用：用户给出了**具体的月薪和加班小时数**，问「加班费是多少 / 能拿多少」。
    何时不用：只问加班费的计算标准或倍率规定、没有给具体数字 —— 那属于法条查询，
    应走检索而不是计算。

    Args:
        monthly_salary: 月薪（元），例如 8000。
        hours: 加班小时数，例如 10。
        rate_type: 加班类型 —— 工作日延时按 1.5 倍、休息日加班按 2 倍、
            法定节假日按 3 倍。用户未指明时按「工作日延时」处理。
    """
    return _render(_calc_overtime_pay(monthly_salary, hours, rate_type))


@tool
def calculate_severance_pay(monthly_salary: float, years: float) -> str:
    """计算经济补偿金（N）金额。

    何时使用：用户给出月薪和工作年限，问「经济补偿金 / 补偿金能拿多少」。
    何时不用：问的是**违法解除的赔偿金（2N）** —— 那要用 calculate_damage_pay。
    两者依据的法条不同（第 47 条 vs 第 87 条），不能混用。

    Args:
        monthly_salary: 离职前 12 个月的平均月工资（元），例如 8000。
        years: 工作年限（年），支持小数，例如 3.5。
            每满 1 年算 1 个月工资；6 个月以上不满 1 年按 1 年算；不满 6 个月算半个月。
    """
    return _render(_calc_severance_pay(monthly_salary, years))


@tool
def calculate_damage_pay(monthly_salary: float, years: float) -> str:
    """计算违法解除劳动合同的赔偿金（2N）金额。

    何时使用：用户被**违法解除**（无理由辞退、辞退理由不合法、程序违法），
    问「赔偿金 / 2N 能拿多少」。赔偿金 = 经济补偿金 × 2。
    何时不用：问的是合法解除情形下的经济补偿金（N）—— 那要用 calculate_severance_pay。

    Args:
        monthly_salary: 离职前 12 个月的平均月工资（元），例如 8000。
        years: 工作年限（年），支持小数，例如 3.5。
    """
    return _render(_calc_damage_pay(monthly_salary, years))


LABOR_TOOLS = [
    calculate_overtime_pay,
    calculate_severance_pay,
    calculate_damage_pay,
]

_TOOL_BY_NAME = {t.name: t for t in LABOR_TOOLS}


# ─── 执行器 ────────────────────────────────────────────────

def execute_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """执行 LLM 返回的 tool_calls，返回 [{name, args, content}, ...]。

    单个工具失败**不影响其余工具**：错误以文本形式回传给模型，让它有机会
    换个参数重试、或如实告知用户缺参数 —— 而不是让整条链路 500。
    """
    executed: List[Dict[str, Any]] = []
    for call in tool_calls:
        name = call.get("name") or ""
        args = call.get("args") or {}
        target = _TOOL_BY_NAME.get(name)
        if target is None:
            content = f"[错误] 未知工具「{name}」。可用工具：{'、'.join(_TOOL_BY_NAME)}"
        else:
            try:
                content = target.invoke(args)
            except Exception as exc:  # 错误要回传给模型，不能中断链路
                content = f"[错误] 工具「{name}」执行失败：{exc}"
        executed.append({"name": name, "args": args, "content": content})
    return executed


def format_tool_results(executed: List[Dict[str, Any]]) -> str:
    """把工具执行结果拼成一段文本，供生成 Prompt 引用。"""
    return "\n\n".join(item["content"] for item in executed)


def summarize_tool_calls(executed: List[Dict[str, Any]]) -> List[str]:
    """产出简短的工具调用轨迹，写入 AgentState.steps 供前端展示。"""
    return [f"{item['name']}({_brief(item['args'])})" for item in executed]


def _brief(args: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items())
