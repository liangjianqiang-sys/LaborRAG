"""法条编号解析的边界测试。

为什么**不用**「中文数字往返属性测试」：
`_cn_to_int` 的解析是按位累加的宽松实现，`_int_to_cn(101)` 曾错误地输出「一百一」，
而 `_cn_to_int("一百一")` 恰好也返回 101 —— 往返一致，但两头都错。
往返测试对这类 bug 是**假阴性**，所以这里直接断言权威中文格式。

覆盖的故障类：劳动法第 101-107 条写作「第一百零X条」，
任何一处字符类漏「零」，都会让这些条号在查询过滤或评估打分时静默失效
（不报错，只是结果悄悄变差）。2026-09 曾真实发生过：当时只修了
text_splitter 一处，vectorstore 与 retrieval_metrics 两处带病三个月。

2026-09 复查又发现另两处（classify 漏「零」、retrieve 漏「千」），
共 7 处正则字符类互不一致。现已全部收敛到 app.utils.law_refs，
本文件末尾的「消费者一致性」用例用于防止再次分裂。
"""
import pytest

from app.utils.law_refs import (
    ARTICLE_REF_PATTERN,
    ARTICLE_REF_RE,
    CN_DIGIT_MAP,
    article_to_cn,
    cn_to_int,
    int_to_cn,
    normalize_article,
)
from app.agent.nodes.classify import classify_complexity
from app.knowledge.splitter import LawArticleSplitter
from app.knowledge.store import (
    _ARTICLE_REF_PATTERN,
    _normalize_article_num,
    extract_article_ref,
)
from app.evaluation.metrics.retrieval import (
    _ARTICLE_PATTERN,
    _normalize_article,
    extract_article_id,
)


# ── 阿拉伯 → 中文：权威格式断言 ──────────────────────────────


@pytest.mark.parametrize(
    "num,expected",
    [
        (1, "一"),
        (10, "十"),
        (11, "十一"),
        (20, "二十"),
        (44, "四十四"),
        (47, "四十七"),
        (99, "九十九"),
        (100, "一百"),
        # 十位为 0 要补「零」
        (101, "一百零一"),
        (105, "一百零五"),
        (109, "一百零九"),
        # 十位非 0 时「一十」不能省「一」
        (110, "一百一十"),
        (115, "一百一十五"),
        (119, "一百一十九"),
        (120, "一百二十"),
        (200, "二百"),
        (201, "二百零一"),
        (999, "九百九十九"),
    ],
)
def test_int_to_cn_权威格式(num, expected):
    assert int_to_cn(num) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("第47条", "第四十七条"),
        ("第101条", "第一百零一条"),
        ("第105条", "第一百零五条"),
        ("第107条", "第一百零七条"),
        ("第110条", "第一百一十条"),
        ("第四十七条", "第四十七条"),
    ],
)
def test_normalize_article_num(raw, expected):
    assert _normalize_article_num(raw) == expected


# ── 中文数字解析：权威格式 + 解析失败语义 ─────────────────────


@pytest.mark.parametrize(
    "cn,expected",
    [
        ("四十七", 47),
        ("十", 10),
        ("一百", 100),
        ("一百零一", 101),
        ("一百〇一", 101),      # 「〇」是「零」的合法异体写法
        ("一百一十", 110),
        ("九百九十九", 999),
        ("一千零一", 1001),      # 解析能力不止于 999
        ("47", 47),             # 纯阿拉伯数字串
    ],
)
def test_cn_to_int_权威格式(cn, expected):
    assert cn_to_int(cn) == expected


@pytest.mark.parametrize("bad", ["一百X", "四十七abc", "", "第"])
def test_cn_to_int_解析失败返回_None(bad):
    """无法解析必须返回 None，不能返回"尽力而为"的半截值。

    原实现遇到未知字符是 `break` 后返回累计值 —— 「一百X」会静默变成 100，
    调用方拿错条号却不自知。这类失真不抛异常。
    """
    assert cn_to_int(bad) is None


@pytest.mark.parametrize(
    "num,expected",
    [(1, "第一条"), (47, "第四十七条"), (101, "第一百零一条"), (110, "第一百一十条")],
)
def test_article_to_cn_权威格式(num, expected):
    assert article_to_cn(str(num)) == expected


def test_article_to_cn_超过99不得回落成阿拉伯写法():
    """回归：law_graph 原用 1–99 的查表，条号 >99 会回落成「第101条」。

    而向量库元数据里的 article 是中文（「第一百零一条」），
    于是精确查找永远匹配不上 —— 知识图谱注入会静默失效。
    现有图谱条号最大 91，所以这是个潜伏缺陷，不是活 bug。
    """
    assert article_to_cn("101") == "第一百零一条"
    assert article_to_cn("110") == "第一百一十条"


def test_int_to_cn_边界行为是显式契约():
    """int_to_cn 刻意只支持 1–999，超出原样返回数字串。

    这是**有意**的边界：现行法律条文数远小于 999，而「千」位段的中文规则
    （1010 应写「一千零一十」而非「一千零十」）需要另一套进位处理。
    固定住这个契约，避免有人误以为它支持任意范围。
    """
    assert int_to_cn(999) == "九百九十九"
    assert int_to_cn(1000) == "1000"
    assert int_to_cn(0) == "0"


# ── 查询侧：元数据过滤 ────────────────────────────────────────


def test_extract_article_ref_含零条号():
    """「第一百零一条」必须能提取出条号，否则过滤退化成只有法律名。"""
    assert extract_article_ref("劳动法第一百零一条怎么规定") == {
        "source": "劳动法.txt",
        "article": "第一百零一条",
    }


def test_extract_article_ref_阿拉伯写法归一():
    """「第101条」与「第一百零一条」必须归一到同一个 article 值。

    这是本条链路的契约：向量库元数据存的是中文格式，
    用户用阿拉伯数字提问时也要能命中同一个过滤条件。
    """
    cn = extract_article_ref("劳动法第一百零一条")
    ar = extract_article_ref("劳动法第101条")
    assert cn["article"] == ar["article"] == "第一百零一条"


def test_extract_article_ref_无条号时只按法律名过滤():
    assert extract_article_ref("劳动法相关问题") == {"source": "劳动法.txt"}
    assert extract_article_ref("加班费怎么算") is None


# ── 评估侧：检索结果标识提取与标准化 ──────────────────────────


def test_extract_article_id_含零条号():
    """检索命中的文档若以「第一百零X条」开头，必须能取到标识。

    取不到就会返回空串，该文档在 P@1/MRR/MAP 里一律算未命中 ——
    即使它其实检索正确。
    """
    assert (
        extract_article_id("劳动法.txt", "第一百零一条 用人单位无理阻挠劳动行政部门行使监督检查权的")
        == "劳动法第101条"
    )
    assert extract_article_id("劳动法.txt", "第一百零七条 本法自1995年1月1日起施行") == "劳动法第107条"


def test_normalize_article_含零条号():
    """标注侧与检索侧都归一到阿拉伯格式，才能相互匹配。"""
    assert _normalize_article("劳动法第四十四条") == "劳动法第44条"
    assert _normalize_article("劳动法第一百零一条") == "劳动法第101条"
    assert _normalize_article("劳动法第一百零七条") == "劳动法第107条"


def test_评估链路端到端可匹配():
    """模拟真实打分：检索结果 → 标识提取 → 与标注比对，必须判为命中。"""
    relevant = {"劳动法第101条"}
    retrieved_id = extract_article_id(
        "劳动法.txt", "第一百零一条 用人单位无理阻挠劳动行政部门行使监督检查权的"
    )
    assert retrieved_id in relevant


# ── 防复发：唯一真相源 + 消费者一致性 ──────────────────────────

# 带尾随空白，兼容 text_splitter 的 `^第…条\s`（用于识别法条标题行）
_SAMPLES = [
    "第一百零一条 用人单位无理阻挠劳动行政部门行使监督检查权的",
    "第一百零五条 违反本法规定侵害劳动者合法权益的",
    "第一百零七条 本法自1995年1月1日起施行",
    "第一百〇一条 用「〇」写零的变体",
    "第一千条 千位边界",
    "第31条 阿拉伯数字写法",
]


def test_唯一真相源覆盖全部合法写法():
    """canonical 模式必须覆盖 中文数字 / 零 / 〇 / 千 / 阿拉伯数字。"""
    for sample in _SAMPLES:
        assert ARTICLE_REF_RE.search(sample), f"canonical 模式未能匹配：{sample}"


def test_全部消费者共用同一模式():
    """6 处消费点必须全部指向 app.utils.law_refs，不得各自实现。

    2026-09 的真实故障：7 处正则各自维护字符类，只修了 1 处，
    另 2 处（classify / retrieve）带病。身份断言让「再抄一份」立刻变红。
    """
    # 直接复用同一编译对象
    assert _ARTICLE_REF_PATTERN is ARTICLE_REF_RE  # vectorstore
    assert _ARTICLE_PATTERN is ARTICLE_REF_RE  # retrieval_metrics

    # text_splitter 语义不同（锚定行首 + 尾随空白），只能组合，
    # 但必须由 canonical 源串拼出，否则又是第二份字符类
    assert LawArticleSplitter.ARTICLE_PATTERN.pattern == rf"^{ARTICLE_REF_PATTERN}\s"
    for sample in _SAMPLES:
        assert LawArticleSplitter.ARTICLE_PATTERN.match(sample), (
            f"text_splitter 未能匹配：{sample}"
        )


def test_复杂度分类识别含零条号():
    """含法条编号的问题应判为 medium（快速检索），而不是 simple。

    classify 的判定顺序是 对比→计算→法条编号→简单词，
    字符类漏「零」会让「第一百零一条…」跳过法条分支、
    落到「什么/规定」等简单词上被误判为 simple，走错检索路径。
    """
    assert classify_complexity({"question": "劳动合同法第一百零一条规定了什么"})["complexity"] == "medium"
    assert classify_complexity({"question": "劳动合同法第一百〇一条规定了什么"})["complexity"] == "medium"
    assert classify_complexity({"question": "劳动合同法第101条规定了什么"})["complexity"] == "medium"

    # 负对照：不带条号时应落到 simple（证明上面的 medium 来自条号分支）
    assert classify_complexity({"question": "劳动合同法规定了什么"})["complexity"] == "simple"


def test_中文数字表只有一份():
    """CN_DIGIT_MAP 的唯一真相源在 law_refs；constants 只做再导出。

    历史上「中文数字解析」散落 6 处（helpers / law_graph 两份 / vectorstore /
    retrieval_metrics / constants），语义互不一致 —— README 的「Bug 1」正是
    这个分裂的产物。身份断言让「再抄一份」立刻变红。
    """
    from app.agent.constants import CN_DIGIT_MAP as constants_map

    assert constants_map is CN_DIGIT_MAP
    assert "〇" in CN_DIGIT_MAP, "「〇」是合法零写法，必须能被解析"


def test_中文数字转换只有一个实现():
    """各调用方拿到的是同一个函数对象，不是各自的副本。"""
    from app.utils.text import normalize_article as helpers_normalize
    from app.retrieval.domain_boost import article_to_cn as lg_article_to_cn
    from app.retrieval.domain_boost import cn_to_int as lg_cn_to_int

    assert helpers_normalize is normalize_article
    assert lg_cn_to_int is cn_to_int
    assert lg_article_to_cn is article_to_cn



def test_复杂度分类同时看原始问题与改写结果():
    """改写节点会滤掉关系性措辞，分类必须同时看原问题，否则丢信号。

    真实 LLM 的改写只保留法言法语术语：实测
    「经济补偿金和赔偿金有什么区别？」→「经济补偿金 赔偿金」——「区别」没了。
    若分类只看改写结果，本应走**对比路径**（带禁止合成/禁止泛化约束）的问题
    会被判成 simple，退化成普通检索问答。
    """
    state = {
        "question": "经济补偿金和赔偿金有什么区别？",
        "rewritten_question": "经济补偿金 赔偿金",  # 真实 LLM 的改写结果
    }
    assert classify_complexity(state)["complexity"] == "complex"


def test_复杂度分类_两处都无信号时仍判_simple():
    """负对照：原问题与改写结果都没有对比/计算信号时，不应被误升为 complex。"""
    assert (
        classify_complexity(
            {"question": "经济补偿金 赔偿金", "rewritten_question": "经济补偿金 赔偿金"}
        )["complexity"]
        == "simple"
    )
