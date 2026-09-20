"""检索确定性测试：同一输入必须永远得到同一输出。

为什么需要跨进程测
------------------
Python 字符串哈希带随机盐（PYTHONHASHSEED），**同一进程内** set 的迭代顺序
是固定的，所以「同进程跑两遍比对」这类测试对下面这个 bug 是**假阴性**：

    hit_articles = set()            # set[tuple[str, str]]
    ...
    for law_name, article_num in hit_articles:   # ← 顺序随进程而变
        ...
    for x in needed_articles[:max_inject]:       # ← 按序截断
        inject(x)

`max_inject` 截断让"顺序"变成"选谁"，于是同一个问题在不同进程里会注入
不同的伴生法条 —— 用户重启服务后问同样的问题，可能得到不同的答案。

2026-09 实测：离线检索基线连跑两次，MAP 在 0.7294 / 0.7301 之间抖动，
其余五项指标恰好稳定（它们只看 top-5），掩盖了问题。
固定 PYTHONHASHSEED=0 后两次结果逐题一致，从而定位到 set 迭代。

因此本文件用**两个不同哈希种子的子进程**跑同一段检索逻辑，比对输出。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent

# 在子进程里执行：用桩替换外部依赖，只暴露 inject_related_articles 的选序行为
_PROBE = r"""
import json, sys
from langchain_core.documents import Document
from app.retrieval import domain_boost as lg

# 每个命中法条都产出多于 max_inject 个互不相同的伴生法条，
# 使「处理顺序」直接决定「谁被截断保留」
def fake_related(law_name, article_num):
    return [(f"{law_name}-伴生", f"{article_num}{i}", "关联", "d") for i in range(3)]

def fake_exact(source, article, k=2):
    doc = Document(
        page_content=f"{article} 正文",
        metadata={"source": source, "article": article, "doc_id": f"{source}|{article}"},
    )
    return [(doc, 1.0)]

lg.get_related_articles = fake_related
lg._exact_lookup_from_parent_store = fake_exact

# 5 个高分命中，元数据用中文条号（_extract_law_article_from_doc 的输入约定）
HITS = [
    ("劳动合同法", "第四十七条"),
    ("劳动法", "第四十四条"),
    ("工伤保险条例", "第十四条"),
    ("社会保险法", "第十六条"),
    ("女职工劳动保护特别规定", "第六条"),
]
docs = [
    (
        Document(
            page_content=f"{law} {cn} 正文",
            metadata={"source": f"{law}.txt", "article": cn, "doc_id": f"{law}|{cn}"},
        ),
        0.9,
    )
    for law, cn in HITS
]

out = lg.inject_related_articles(docs, retriever=None, max_inject=3)
injected = [d.metadata.get("article") for d, _ in out if d.metadata.get("injected_by") == "rule"]
print(json.dumps({"injected": injected, "total": len(out)}, ensure_ascii=False))
"""


def _run_probe(hash_seed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(BACKEND_DIR),
    )
    if proc.returncode != 0:
        pytest.fail(f"子进程失败（PYTHONHASHSEED={hash_seed}）：\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("seed_a,seed_b", [("0", "1"), ("0", "12345"), ("7", "999")])
def test_伴生法条注入不随哈希种子变化(seed_a, seed_b):
    """不同 PYTHONHASHSEED 下，注入的伴生法条必须完全一致。

    回归的是 `domain_boost.inject_related_articles` 里直接迭代 set 的写法：
    改为「列表 + 去重集合」保序后，本用例才可能通过。
    """
    a = _run_probe(seed_a)
    b = _run_probe(seed_b)
    assert a["injected"] == b["injected"], (
        f"注入顺序随哈希种子变化：\n"
        f"  PYTHONHASHSEED={seed_a} → {a['injected']}\n"
        f"  PYTHONHASHSEED={seed_b} → {b['injected']}"
    )
    assert a["total"] == b["total"]


def test_伴生法条注入确实产生了候选_防桩失效():
    """确认桩真的让 inject 生效了 —— 否则上面的用例是空对空。"""
    r = _run_probe("0")
    assert r["injected"], "没有注入任何伴生法条，说明桩未生效，上面的用例无意义"
