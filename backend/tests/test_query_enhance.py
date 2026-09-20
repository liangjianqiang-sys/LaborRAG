"""query_enhance 查询增强的单测。

纯规则模块（70 条口语→法言法语硬编码映射），零 LLM、此前零测试覆盖。
本文件补两类用例：

1. 行为用例 —— 无匹配原样返回、命中追加、只取一条不堆砌；
2. **「短词不得遮蔽长词」的客观扫描** —— 映射表会随维护不断加词，任何一次
   「短关键词排在长关键词之前」的笔误都会立刻变红，而不是靠人肉发现。
"""
from app.retrieval.query_enhance import enhance_query, _KEYWORD_MAP


def test_无匹配原样返回():
    assert enhance_query("今天天气怎么样") == "今天天气怎么样"
    assert enhance_query("") == ""


def test_命中追加法言法语_保留原问题():
    # 「拖欠工资」的映射值是「拖欠劳动报酬…」，不会重复原词
    assert enhance_query("拖欠工资") == "拖欠工资 拖欠劳动报酬 劳动合同法第八十五条"
    assert enhance_query("试用期最长多久") == "试用期最长多久 试用期期限 劳动合同法第十九条"


def test_关键词表无重复():
    """重复键会让维护者误以为两处独立、改一处漏一处。"""
    keys = [k for k, _ in _KEYWORD_MAP]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    assert not dup, f"存在重复关键词：{dup}"


def test_只取一条匹配不堆砌():
    # 「经济补偿金和赔偿金区别」同时含「经济补偿金」「赔偿金」，
    # 但应只命中最长的那条（第一条），不把三条都堆上去。
    out = enhance_query("经济补偿金和赔偿金区别")
    assert out == "经济补偿金和赔偿金区别 经济补偿金 赔偿金 劳动合同法第四十七条 第八十七条"


def test_短词不得遮蔽长词():
    """客观扫描：若某关键词是另一关键词的子串且排在前面，会遮蔽后者。

    例：「工伤认定」排在「工伤认定时限」之前 → 查询含「工伤认定时限」时
    会先命中短的「工伤认定」，拿到第十四/十五条（认定情形），
    而拿不到第十七条（申请时限）—— 静默地检索错法条，不报错。
    """
    violations = []
    for i in range(len(_KEYWORD_MAP)):
        ki, _ = _KEYWORD_MAP[i]
        for j in range(i + 1, len(_KEYWORD_MAP)):
            kj, _ = _KEYWORD_MAP[j]
            if ki != kj and ki in kj:
                violations.append((i, ki, j, kj))
    assert not violations, (
        "以下短关键词排在长关键词之前，会遮蔽后者：\n"
        + "\n".join(f"  [{i}] {ki!r} 遮蔽 [{j}] {kj!r}" for i, ki, j, kj in violations)
    )


def test_工伤认定时限命中第十七条():
    """阴性对照：锁定上面扫描发现的具体缺陷。

    查询「工伤认定时限」应增强出「第十七条」（申请时限），
    而不是「第十四/十五条」（认定情形）。
    """
    out = enhance_query("工伤认定时限是多久")
    assert "第十七条" in out, f"应命中第十七条，实际：{out}"
    assert "第十四条" not in out, f"不应命中第十四条，实际：{out}"
