"""劳动法计算工具集测试。

不需要 LLM —— 工具定义、参数 schema、执行器全是本地的，可离线跑。
（工具**被模型选中**这件事依赖服务端 function calling，只能端到端验证，
不在本文件覆盖范围。）
"""
import pytest

from app.tools.labor_tools import (
    LABOR_TOOLS,
    execute_tool_calls,
    format_tool_results,
    summarize_tool_calls,
)

EXPECTED_NAMES = {
    "calculate_overtime_pay",
    "calculate_severance_pay",
    "calculate_damage_pay",
}


# ─── 工具定义 ──────────────────────────────────────────────

def test_工具清单与名称():
    assert {t.name for t in LABOR_TOOLS} == EXPECTED_NAMES


def test_每个工具都有足够详细的描述():
    """docstring 就是发给模型的 tool description —— 空描述会让模型不会选它。"""
    for t in LABOR_TOOLS:
        assert t.description and len(t.description.strip()) > 30, (
            f"{t.name} 的 docstring 太短或为空，模型无法判断何时该用它"
        )


def test_参数_schema_包含预期字段():
    """工具参数名必须与底层计算函数的签名一致（否则 invoke 会校验失败）。"""
    def _props(t) -> set:
        schema = t.args_schema
        if hasattr(schema, "model_json_schema"):
            return set(schema.model_json_schema().get("properties", {}))
        return set(schema.schema().get("properties", {}))

    by_name = {t.name: _props(t) for t in LABOR_TOOLS}
    assert by_name["calculate_overtime_pay"] == {"monthly_salary", "hours", "rate_type"}
    assert by_name["calculate_severance_pay"] == {"monthly_salary", "years"}
    assert by_name["calculate_damage_pay"] == {"monthly_salary", "years"}


# ─── 工具行为（金额正确性）──────────────────────────────────

def test_加班费_默认按工作日延时一点五倍():
    out = LABOR_TOOLS[0].invoke({"monthly_salary": 8000, "hours": 10})
    # 8000 / 21.75 / 8 × 10 × 1.5 = 689.66
    assert "689.66" in out
    assert "《劳动法》第44条" in out


def test_加班费_法定节假日三倍():
    out = LABOR_TOOLS[0].invoke(
        {"monthly_salary": 8000, "hours": 10, "rate_type": "法定节假日"}
    )
    # 8000 / 21.75 / 8 × 10 × 3 = 1379.31
    assert "1379.31" in out


def test_经济补偿金_整年():
    out = LABOR_TOOLS[1].invoke({"monthly_salary": 8000, "years": 5})
    assert "40000.0" in out
    assert "《劳动合同法》第47条" in out


def test_经济补偿金_不满半年按半个月():
    """年限进位边界：0.3 年 → floor=0，小数部分 0.3 < 0.5 但 > 0 → 加半个月。"""
    out = LABOR_TOOLS[1].invoke({"monthly_salary": 8000, "years": 0.3})
    assert "4000.0" in out


def test_赔偿金_是经济补偿金的两倍():
    out = LABOR_TOOLS[2].invoke({"monthly_salary": 8000, "years": 5})
    assert "80000.0" in out
    assert "《劳动合同法》第87条" in out


# ─── 执行器 ────────────────────────────────────────────────

def test_执行器_按名称分发到对应工具():
    executed = execute_tool_calls([
        {"name": "calculate_severance_pay", "args": {"monthly_salary": 10000, "years": 2}},
    ])
    assert len(executed) == 1
    assert executed[0]["name"] == "calculate_severance_pay"
    assert "20000.0" in executed[0]["content"]


def test_执行器_支持一次调用多个工具():
    executed = execute_tool_calls([
        {"name": "calculate_severance_pay", "args": {"monthly_salary": 10000, "years": 2}},
        {"name": "calculate_damage_pay", "args": {"monthly_salary": 10000, "years": 2}},
    ])
    assert len(executed) == 2
    assert "20000.0" in executed[0]["content"]
    assert "40000.0" in executed[1]["content"]


def test_执行器_未知工具返回错误文本而不抛异常():
    executed = execute_tool_calls([{"name": "no_such_tool", "args": {}}])
    assert "未知工具" in executed[0]["content"]
    assert "calculate_overtime_pay" in executed[0]["content"]


def test_执行器_参数非法返回错误文本而不抛异常():
    """缺参数 / 类型错误都不该让链路炸掉，而是把错误回传给模型。"""
    executed = execute_tool_calls([{"name": "calculate_severance_pay", "args": {}}])
    assert executed[0]["content"].startswith("[错误]")


def test_执行器_单个工具失败不影响其他工具():
    executed = execute_tool_calls([
        {"name": "calculate_severance_pay", "args": {}},  # 缺参数 → 失败
        {"name": "calculate_damage_pay", "args": {"monthly_salary": 10000, "years": 2}},
    ])
    assert executed[0]["content"].startswith("[错误]")
    assert "40000.0" in executed[1]["content"], "第二个工具应照常执行"


def test_执行器_容忍缺失的_args_字段():
    executed = execute_tool_calls([{"name": "calculate_damage_pay"}])
    assert executed[0]["args"] == {}
    assert executed[0]["content"].startswith("[错误]")


# ─── 结果格式化 ────────────────────────────────────────────

def test_format_tool_results_拼接多个结果():
    executed = execute_tool_calls([
        {"name": "calculate_severance_pay", "args": {"monthly_salary": 10000, "years": 2}},
        {"name": "calculate_damage_pay", "args": {"monthly_salary": 10000, "years": 2}},
    ])
    text = format_tool_results(executed)
    assert "【经济补偿金】" in text
    assert "【赔偿金】" in text


def test_summarize_tool_calls_带参数轨迹():
    executed = execute_tool_calls([
        {"name": "calculate_overtime_pay", "args": {"monthly_salary": 8000, "hours": 10}},
    ])
    trace = summarize_tool_calls(executed)
    assert trace == ["calculate_overtime_pay(monthly_salary=8000, hours=10)"]


@pytest.mark.parametrize("index,keyword", [(0, "加班费"), (1, "经济补偿金"), (2, "赔偿金")])
def test_渲染结果包含类型_法条_公式_金额(index, keyword):
    args = {"monthly_salary": 8000, "hours": 10} if index == 0 else {"monthly_salary": 8000, "years": 3}
    out = LABOR_TOOLS[index].invoke(args)
    assert keyword in out
    assert "法条依据" in out
    assert "计算公式" in out
    assert "计算结果" in out
