"""Calculators — 劳动法计算引擎，不感知 AgentState，不依赖 LLM/Retriever。"""
import re
import math

from app.core.generator.constants import MONTHLY_WORK_DAYS, DAILY_WORK_HOURS


# ─── 计算公式字典 ──

LABOR_CALCULATORS = {
    "加班费": {
        "formula": "加班费 = 月薪 ÷ 21.75 ÷ 8 × 加班小时数 × 倍率",
        "rates": {"工作日延时": 1.5, "休息日加班": 2.0, "法定节假日": 3.0},
        "law_ref": "《劳动法》第44条",
    },
    "经济补偿金": {
        "formula": "经济补偿金 = 工作年限 × 月工资",
        "law_ref": "《劳动合同法》第47条",
        "note": "每满1年支付1个月工资，6个月以上不满1年按1年算，不满6个月支付半个月",
    },
    "赔偿金": {
        "formula": "赔偿金 = 经济补偿金 × 2",
        "law_ref": "《劳动合同法》第87条",
        "note": "用人单位违反本法规定解除或终止劳动合同",
    },
    "未签合同双倍工资": {
        "formula": "双倍工资差额 = 月工资 × 应签未签月数(最多11个月)",
        "law_ref": "《劳动合同法》第82条",
        "note": "超过1个月不满1年未签书面合同，每月支付2倍工资",
    },
    "年休假工资": {
        "formula": "年休假工资 = 日工资 × 300% × 未休天数",
        "law_ref": "《职工带薪年休假条例》第5条",
        "note": "日工资 = 月薪 ÷ 21.75",
    },
}


# ─── 辅助函数 ──

def _extract_number(text: str, pattern: str) -> float:
    """从文本中提取数字。"""
    match = re.search(pattern, text)
    if match:
        return float(match.group(1).replace(",", ""))
    return 0.0


def _first_match(patterns: list, text: str) -> float:
    """按顺序尝试多个正则，返回第一个匹配的数值。"""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1).replace(",", ""))
    return 0.0


def _parse_cn_salary(text: str) -> float:
    """解析中文薪资表达，如"1万5"→15000。"""
    m = re.search(r"(\d+)万(\d*[千百]?)", text)
    if not m:
        return 0.0
    wan = float(m.group(1))
    rest_str = m.group(2)
    if not rest_str:
        return wan * 10000
    if "千" in rest_str:
        return wan * 10000 + float(rest_str.replace("千", "")) * 1000
    if "百" in rest_str:
        return wan * 10000 + float(rest_str.replace("百", "")) * 100
    return wan * 10000 + float(rest_str) * 1000


# ─── 计算函数 ──

def calculate_overtime_pay(monthly_salary: float, hours: float,
                           rate_type: str = "工作日延时") -> dict:
    """计算加班费。"""
    rate = LABOR_CALCULATORS["加班费"]["rates"].get(rate_type, 1.5)
    hourly_wage = monthly_salary / MONTHLY_WORK_DAYS / DAILY_WORK_HOURS
    overtime_pay = round(hourly_wage * hours * rate, 2)
    return {
        "type": "加班费",
        "formula": LABOR_CALCULATORS["加班费"]["formula"],
        "law_ref": LABOR_CALCULATORS["加班费"]["law_ref"],
        "monthly_salary": monthly_salary,
        "hourly_wage": round(hourly_wage, 2),
        "overtime_hours": hours,
        "rate_type": rate_type,
        "rate": rate,
        "result": overtime_pay,
    }


def calculate_severance_pay(monthly_salary: float, years: float) -> dict:
    """计算经济补偿金。"""
    months = math.floor(years)
    if years - months >= 0.5:
        months += 1
    elif years - months > 0:
        months += 0.5
    severance = round(monthly_salary * months, 2)
    return {
        "type": "经济补偿金",
        "formula": LABOR_CALCULATORS["经济补偿金"]["formula"],
        "law_ref": LABOR_CALCULATORS["经济补偿金"]["law_ref"],
        "monthly_salary": monthly_salary,
        "working_years": years,
        "compensation_months": months,
        "result": severance,
    }


def calculate_damage_pay(monthly_salary: float, years: float) -> dict:
    """计算赔偿金（违法解除）。"""
    severance = calculate_severance_pay(monthly_salary, years)
    damage = round(severance["result"] * 2, 2)
    return {
        "type": "赔偿金",
        "formula": LABOR_CALCULATORS["赔偿金"]["formula"],
        "law_ref": LABOR_CALCULATORS["赔偿金"]["law_ref"],
        "monthly_salary": monthly_salary,
        "working_years": years,
        "severance_base": severance["result"],
        "result": damage,
    }


def auto_calculate(question: str, conversation_context: str = "") -> str:
    """根据问题自动识别计算类型并执行计算。

    尝试从问题中提取月薪、年限、小时数等参数。
    当追问缺少关键词时，从对话上下文继承计算类型和缺失参数。
    """
    combined = f"{conversation_context}\n{question}" if conversation_context else question

    # 提取月薪：优先当前问题，其次上下文，最后中文表达
    salary = _first_match(
        [r"月薪[为约]?(\d+[,.]?\d*)", r"工资[为约]?(\d+[,.]?\d*)", r"(\d+[,.]?\d*)元[/月]"],
        question,
    ) or _first_match([r"月薪[为约]?(\d+[,.]?\d*)"], combined) or _parse_cn_salary(question)

    # 提取年限：优先当前问题，其次上下文
    years = _first_match(
        [r"工作[了为约]?(\d+\.?\d*)年", r"(\d+\.?\d*)年"],
        question,
    ) or (conversation_context and _first_match([r"工作[了为约]?(\d+\.?\d*)年"], combined))

    # 提取加班小时：优先当前问题，其次上下文
    hours = _first_match(
        [r"加班[了为约]?(\d+\.?\d*)小时", r"(\d+\.?\d*)小时"],
        question,
    ) or (conversation_context and _first_match([r"加班[了为约]?(\d+\.?\d*)小时"], combined))

    # 判断加班类型
    rate_type = "工作日延时"
    if "休息日" in combined or "周末" in combined:
        rate_type = "休息日加班"
    elif "法定节假日" in combined or "节假日" in combined or "国庆" in combined or "春节" in combined:
        rate_type = "法定节假日"

    # 从上下文继承计算类型关键词
    has_overtime = "加班" in combined or "延时" in combined
    has_severance = "经济补偿" in combined or "补偿金" in combined
    has_damage = "赔偿金" in combined or "违法解除" in combined or "违法辞退" in combined

    results = []
    missing = []

    # 加班费计算
    if has_overtime:
        if salary > 0 and hours > 0:
            r = calculate_overtime_pay(salary, hours, rate_type)
            results.append(
                f"【{r['type']}】\n"
                f"  法条依据：{r['law_ref']}\n"
                f"  计算公式：{r['formula']}\n"
                f"  月薪：{r['monthly_salary']}元 → 时薪：{r['hourly_wage']}元\n"
                f"  加班{r['overtime_hours']}小时 × {r['rate']}倍({r['rate_type']})\n"
                f"  ➜ 加班费 = {r['result']}元"
            )
        else:
            if salary == 0: missing.append("月薪")
            if hours == 0: missing.append("加班小时数")
            results.append(
                f"【加班费】\n"
                f"  法条依据：{LABOR_CALCULATORS['加班费']['law_ref']}\n"
                f"  计算公式：{LABOR_CALCULATORS['加班费']['formula']}\n"
                f"  ⚠️ 缺少参数：{', '.join(missing)}，无法计算具体金额"
            )
            missing.clear()

    # 经济补偿金
    if has_severance:
        if salary > 0 and years > 0:
            r = calculate_severance_pay(salary, years)
            results.append(
                f"【{r['type']}】\n"
                f"  法条依据：{r['law_ref']}\n"
                f"  计算公式：{r['formula']}\n"
                f"  月薪：{r['monthly_salary']}元 × {r['compensation_months']}个月\n"
                f"  ➜ 经济补偿金 = {r['result']}元"
            )
        else:
            if salary == 0: missing.append("月薪")
            if years == 0: missing.append("工作年限")
            results.append(
                f"【经济补偿金】\n"
                f"  法条依据：{LABOR_CALCULATORS['经济补偿金']['law_ref']}\n"
                f"  计算公式：{LABOR_CALCULATORS['经济补偿金']['formula']}\n"
                f"  ⚠️ 缺少参数：{', '.join(missing)}，无法计算具体金额"
            )
            missing.clear()

    # 赔偿金
    if has_damage:
        if salary > 0 and years > 0:
            r = calculate_damage_pay(salary, years)
            results.append(
                f"【{r['type']}】\n"
                f"  法条依据：{r['law_ref']}\n"
                f"  计算公式：{r['formula']}\n"
                f"  经济补偿金基数：{r['severance_base']}元 × 2倍\n"
                f"  ➜ 赔偿金 = {r['result']}元"
            )
        else:
            if salary == 0: missing.append("月薪")
            if years == 0: missing.append("工作年限")
            results.append(
                f"【赔偿金】\n"
                f"  法条依据：{LABOR_CALCULATORS['赔偿金']['law_ref']}\n"
                f"  计算公式：{LABOR_CALCULATORS['赔偿金']['formula']}\n"
                f"  ⚠️ 缺少参数：{', '.join(missing)}，无法计算具体金额"
            )
            missing.clear()

    if not results:
        return "未能识别计算类型或缺少必要参数（月薪/年限/小时数）。"

    return "\n\n".join(results)
