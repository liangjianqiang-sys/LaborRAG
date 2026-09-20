"""法条引用原语的唯一真相源（single source of truth）。

覆盖三类原语，它们此前各自散落多份实现：

1. **法条编号正则**（`ARTICLE_REF_RE`）
   曾在 7 处各自维护字符类，2 处漏「零」/漏「千」，后果不是报错而是**静默失真**：
   `第一百零一条` 整段匹配失败 → 向量库条号过滤静默退化；评估提取不到条号，
   检索明明命中也算未命中，P@1 / MRR 被系统性低估。

2. **中文数字 ↔ 阿拉伯数字**（`cn_to_int` / `int_to_cn`）
   曾有 6 套实现（`helpers.normalize_article`、`law_graph._CN_NUM`、
   `law_graph._article_to_cn`、`vectorstore._int_to_cn`、
   `retrieval_metrics._cn_to_int`、`constants.CN_DIGIT_MAP`），语义互不一致：
   - `retrieval_metrics` 的字符表不含「〇」，把 `第一百〇一条` 解析成 **100**
   - `law_graph._article_to_cn` 是 1–99 的查表，条号 >99 直接回落成阿拉伯写法，
     永远匹配不上元数据里的中文条号
   README 里的「Bug 1：法条切分正则漏零」之所以存在，就是因为同一个"中文数字
   解析"散落多处，改一处漏五处。

3. **法律名 ↔ 向量库 source 文件名**（`_LAW_SOURCE_MAP` / `_SOURCE_LAW_MAP`）
   放在最底层是为了打破一个包级循环：`app/utils/text.py` 的 `_law_display()`
   需要这张表来显示法律名，而 `app/retrieval/domain_boost/` 也需要它做精确查找。
   表若留在 domain_boost，就成了 `utils → retrieval`（上层）的反向依赖。
   新增法律文档时**必须同步维护**这张表。

**本模块必须保持零依赖** —— 最底层的 `vectorstore` 与 `text_splitter` 都要引用它。
"""
import re

# ── 法条编号正则 ──────────────────────────────────────────────

# 条号中允许出现的字符：中文数字（含 零 / 〇 两种零写法）+ 阿拉伯数字
_ARTICLE_REF_CHARS = "一二三四五六七八九十百千零〇"

# 未锚定的法条编号模式：在任意文本中查找「第X条」
ARTICLE_REF_PATTERN = rf"第[{_ARTICLE_REF_CHARS}\d]+条"

# 预编译版本（查找场景直接用）
ARTICLE_REF_RE = re.compile(ARTICLE_REF_PATTERN)


# ── 中文数字 ↔ 阿拉伯数字 ────────────────────────────────────

# 数字位 → 值。零有两种合法写法：零、〇
CN_DIGIT_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}

# 生成方向只用「零」（「〇」是异体，不作为输出）
_INT_TO_CN_DIGIT = {v: k for k, v in CN_DIGIT_MAP.items() if v < 10}
_INT_TO_CN_DIGIT[0] = "零"    # 覆盖推导式里可能取到的「〇」
_INT_TO_CN_DIGIT[10] = "十"   # 「十」单用（推导式按 v<10 过滤掉了它）


def cn_to_int(cn: str) -> int | None:
    """中文数字 → 整数；无法解析时返回 None。

    支持常规中文写法（含两种零）以及纯阿拉伯数字串：
        四十七 → 47    一百零一 → 101   一百〇一 → 101
        一百一十 → 110  一千零一 → 1001   "47" → 47

    解析不了时返回 **None**，而不是"尽力而为"的半截值 ——
    静默返回部分结果（例如把 `一百〇一` 当成 100）会让调用方拿错条号却不自知，
    这类失真不会抛异常，只会让评估指标悄悄变差。
    """
    if not cn:
        return None
    if cn.isdigit():
        return int(cn)

    result = 0
    current = 0
    for ch in cn:
        val = CN_DIGIT_MAP.get(ch)
        if val is None:
            return None
        if val >= 10:
            # 「十」单用时省略前导「一」：十 → 10
            if current == 0:
                current = 1
            result += current * val
            current = 0
        else:
            current = val
    return result + current


def int_to_cn(num: int) -> str:
    """整数 → 中文数字，**支持 1–999**；超出范围原样返回数字串。

    范围刻意止于 999：现行法律条文数远小于此，而「千」位段的中文规则
    （如 1010 应写作「一千零一十」而非「一千零十」）需要另一套进位处理，
    在没有真实需求前引入只会新增 bug 面。边界行为由测试显式固定。

    两条易漏的进位规则（漏了会让生成的条号与向量库元数据对不上）：
      - 十位为 0 且个位非 0 要补「零」：101 → 一百零一（不是「一百一」）
      - 十位非 0 时「一十」不能省「一」：110 → 一百一十（不是「一百十」）
    """
    if num <= 0 or num > 999:
        return str(num)
    if num <= 10:
        return _INT_TO_CN_DIGIT[num]
    if num < 20:
        return "十" + _INT_TO_CN_DIGIT[num - 10]
    if num < 100:
        tens, ones = divmod(num, 10)
        return _INT_TO_CN_DIGIT[tens] + "十" + (_INT_TO_CN_DIGIT[ones] if ones else "")
    hundreds, rest = divmod(num, 100)
    cn = _INT_TO_CN_DIGIT[hundreds] + "百"
    if rest == 0:
        return cn
    if rest < 10:
        return cn + "零" + _INT_TO_CN_DIGIT[rest]
    if rest < 20:
        return cn + "一十" + (_INT_TO_CN_DIGIT[rest - 10] if rest > 10 else "")
    return cn + int_to_cn(rest)


def normalize_article(article: str) -> str:
    """法条编号归一化为阿拉伯格式：`第四十七条` → `第47条`。

    输入须形如「第X条」；无法解析时**原样返回**，不猜测。
    """
    if not (article.startswith("第") and article.endswith("条")):
        return article
    num = cn_to_int(article[1:-1])
    return f"第{num}条" if num is not None else article


def article_to_cn(num: str) -> str:
    """阿拉伯条号 → 中文法条编号：`'47'` → `'第四十七条'`。

    非数字输入原样包进「第…条」，不猜测。
    """
    digits = num.strip()
    if digits.isdigit():
        return f"第{int_to_cn(int(digits))}条"
    return f"第{digits}条"


# ── 法律名 ↔ 向量库 source 文件名 ─────────────────────────────
#
# 知识图谱/概念映射里用的是「法律名」（如「劳动合同法」），
# 而向量库 metadata 里存的是「source 文件名」（如「劳动合同法.txt」）。
# 这张表是两者之间唯一的桥 —— **新增法律文档时必须同步维护**。

_LAW_SOURCE_MAP = {
    "劳动合同法": "劳动合同法.txt",
    "劳动法": "劳动法.txt",
    "劳动争议调解仲裁法": "劳动争议调解仲裁法.txt",
    "社会保险法": "社会保险法.txt",
    "职工带薪年休假条例": "职工带薪年休假条例.txt",
    "工伤保险条例": "工伤保险条例.pdf",
    "女职工劳动保护特别规定": "女职工劳动保护特别规定.pdf",
    "最低工资规定": "最低工资规定.pdf",
    "工资支付暂行规定": "工资支付暂行规定.pdf",
    "失业保险条例": "失业保险条例.pdf",
    "劳动合同法实施条例": "中华人民共和国劳动合同法实施条例.pdf",
    "职业病防治法": "中华人民共和国职业病防治法.pdf",
    "住房公积金管理条例": "住房公积金管理条例.pdf",
    "司法解释（一）": "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）.pdf",
    "司法解释（二）": "最高人民法院关于审理劳动争议案件适用法律问题的解释（二）.pdf",
}

# 反向映射：source 文件名 → 法律名
_SOURCE_LAW_MAP = {v: k for k, v in _LAW_SOURCE_MAP.items()}
