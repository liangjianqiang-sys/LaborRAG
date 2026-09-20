"""guardrails 回答护栏的单测。

纯正则规则（14 条），零 LLM、此前零测试覆盖。本文件补行为用例：
触发判定、前提已满足时不追加、多规则触发、幂等（二次调用不重复追加）。
"""
from app.agent.guardrails import AnswerGuardrails


def test_无关键词原样返回():
    """未命中任何 trigger 时，必须原样返回，不能加空的分隔符。"""
    assert AnswerGuardrails.check("这是普通回答") == "这是普通回答"
    assert AnswerGuardrails.check("") == ""


def test_触发且缺前提_追加提示():
    out = AnswerGuardrails.check("加班费根据法律规定支付")
    assert "**注意事项：**" in out
    assert "150%" in out  # 加班费 hint 的关键内容（三种倍率）


def test_触发且前提已满足_不追加():
    # 已经说清三种情形与倍率 → 前提满足，不应再画蛇添足
    ans = "加班费：工作日延时150%、休息日200%、法定节假日300%。"
    assert AnswerGuardrails.check(ans) == ans


def test_多规则触发_追加多条():
    # 「经济补偿金」与「赔偿金」是两个独立 trigger，应各追加一条
    out = AnswerGuardrails.check("经济补偿金和赔偿金怎么算")
    assert "第46条" in out  # 经济补偿 hint
    assert "第87条" in out  # 赔偿金 hint
    assert out.count("经济补偿金仅在") == 1  # 每条 hint 恰好出现一次


def test_二次调用不重复追加_幂等():
    ans = "经济补偿金怎么算"
    once = AnswerGuardrails.check(ans)
    twice = AnswerGuardrails.check(once)
    assert twice == once
