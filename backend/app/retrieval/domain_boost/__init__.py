"""知识图谱：伴生法条关系 + 概念映射 + 检索注入。

本包是「法条编号之外」的领域知识层。拆分为：
  - source_map.py  法律名 ↔ 文件名映射（纯数据）
  - relations.py   伴生法条关系表（纯数据，471 行数据里的 240 行）
  - concepts.py    概念→法条映射表（纯数据，213 行）
  - lookup.py      查找与注入逻辑（6 个函数）

**本 __init__ 是唯一门面**：调用方继续写
`from app.retrieval.domain_boost import inject_related_articles`，
不必知道内部怎么分文件；内部重构（再拆或再合并）不会波及调用点。

注意：私有名（下划线开头）也在此再导出，因为既有调用方与测试按名引用它们。
"""
# 纯数据
from app.retrieval.domain_boost.source_map import _LAW_SOURCE_MAP, _SOURCE_LAW_MAP
from app.retrieval.domain_boost.relations import LAW_GRAPH
from app.retrieval.domain_boost.concepts import CONCEPT_ARTICLE_MAP

# 查找与注入逻辑
from app.retrieval.domain_boost.lookup import (
    _load_parent_store,
    _exact_lookup_from_parent_store,
    concept_lookup,
    _extract_law_article_from_doc,
    get_related_articles,
    inject_related_articles,
)

# 中文数字转换：调用方与测试按 `law_graph.article_to_cn` 引用，故一并再导出
from app.utils.law_refs import article_to_cn, cn_to_int  # noqa: F401

__all__ = [
    "LAW_GRAPH", "CONCEPT_ARTICLE_MAP",
    "_LAW_SOURCE_MAP", "_SOURCE_LAW_MAP",
    "_load_parent_store", "_exact_lookup_from_parent_store",
    "concept_lookup", "_extract_law_article_from_doc",
    "get_related_articles", "inject_related_articles",
    "article_to_cn", "cn_to_int",
]
