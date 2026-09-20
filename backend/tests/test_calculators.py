"""calculators 计算引擎单测。

这些是纯函数，不依赖 LLM/检索/settings，可直接 import 裸调。
重点覆盖：加班费倍率、经济补偿金年限进位（最易错）、赔偿金=补偿金×2、
中文薪资解析、缺参引导分支。
"""
import math

from app.tools.labor_calculator import (
    DAILY_WORK_HOURS,
    MONTHLY_WORK_DAYS,
    _parse_cn_salary,
    auto_calculate,
    calculate_damage_pay,
    calculate_overtime_pay,
    calculate_severance_pay,
)


def test_overtime_three_rates():
    """三档倍率 1.5/2.0/3.0，且 result 严格等于公式独立计算值。"""
    salary, hours = 8000, 8
    hourly = salary / MONTHLY_WORK_DAYS / DAILY_WORK_HOURS  # 未取整的时薪

    for rate_type, expected_rate in [("工作日延时", 1.5), ("休息日加班", 2.0), ("法定节假日", 3.0)]:
        r = calculate_overtime_pay(salary, hours, rate_type)
        assert r["rate"] == expected_rate
        assert r["rate_type"] == rate_type
        # result 用未取整时薪计算（代码里 overtime_pay 用未取整 hourly_wage）
        assert r["result"] == round(hourly * hours * expected_rate, 2)
        # 展示用时薪取整
        assert r["hourly_wage"] == round(hourly, 2)
        assert r["law_ref"] == "《劳动法》第44条"


def test_severance_years_rounding():
    """经济补偿金年限进位边界（math.floor + 0.5 阈值，最易错）。

    真实逻辑：
      years=0.3 → 0.5 月（不满6个月）
      years=0.5 → 1 月（>=0.5 进1）
      years=0.6 → 1 月
      years=1.0 → 1 月（满1年）
      years=1.4 → 1.5 月（1年 + 不满6个月的0.5）
      years=1.5 → 2 月（1年 + >=0.5进1）
      years=2.0 → 2 月
    """
    cases = {0.3: 0.5, 0.5: 1, 0.6: 1, 1.0: 1, 1.4: 1.5, 1.5: 2, 2.0: 2}
    salary = 10000
    for years, expected_months in cases.items():
        r = calculate_severance_pay(salary, years)
        assert r["compensation_months"] == expected_months, (
            f"years={years} 期望 {expected_months} 月，实际 {r['compensation_months']}"
        )
        assert r["result"] == round(salary * expected_months, 2)


def test_damage_is_double_severance():
    """赔偿金 = 经济补偿金 × 2，且 severance_base 字段一致。"""
    for salary, years in [(8000, 1.5), (12000, 3.0), (6500, 0.5)]:
        sev = calculate_severance_pay(salary, years)
        dmg = calculate_damage_pay(salary, years)
        assert dmg["result"] == round(sev["result"] * 2, 2)
        assert dmg["severance_base"] == sev["result"]
        assert dmg["law_ref"] == "《劳动合同法》第87条"


def test_parse_cn_salary():
    """中文薪资表达解析。"""
    assert _parse_cn_salary("1万5") == 15000
    assert _parse_cn_salary("2万") == 20000
    assert _parse_cn_salary("1万8千") == 18000
    assert _parse_cn_salary("3万2百") == 30200
    assert _parse_cn_salary("无匹配文本") == 0.0


def test_auto_calculate_missing_param():
    """缺月薪时返回缺参引导，而非崩溃或编造金额。"""
    result = auto_calculate("加班10小时")  # 只给了小时数，没月薪
    assert "⚠️ 缺少参数" in result
    assert "月薪" in result
    # 不应出现具体金额数字作为结果
    assert "➜ 加班费 = " not in result
