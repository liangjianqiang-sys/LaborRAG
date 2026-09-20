"""常量配置 — 复杂度分类关键词、法条编号归一化映射、计算常量。

零依赖：只依赖标准库与同为零依赖的 app.utils.law_refs。
"""
from app.utils.law_refs import CN_DIGIT_MAP  # noqa: F401  （再导出，见文件末尾说明）

# ── 复杂度分类关键词 ──

COMPLEX_KEYWORDS = frozenset({
    "区别", "不同", "对比", "比较", "差异", "分别", "哪个", "哪种",
})

CALCULATE_KEYWORDS = frozenset({
    "多少", "怎么算", "计算", "赔偿金计算", "补偿金计算", "加班费计算",
    "工资计算", "经济补偿计算", "年假", "产假", "婚假", "丧假",
    "带薪", "天数", "比例", "倍数", "标准", "基数",
})

SIMPLE_KEYWORDS = frozenset({
    "是什么", "定义", "含义", "包括", "包含", "适用", "范围",
    "规定", "哪些", "什么", "是否", "可以", "能否", "多久", "几条",
})

# ── 中文数字→阿拉伯数字映射 ──
# CN_DIGIT_MAP 的唯一真相源在 app.utils.law_refs，此处仅再导出，
# 保留「常量去 constants.py 找」的习惯路径。不要在别处重新定义它。

# 注：劳动法计算常量（月计薪天数 / 日工时）已移到 app/tools/labor_calculator.py
# —— 它们只服务于计算，放在 agent 层会让 tools 反向依赖 agent。
