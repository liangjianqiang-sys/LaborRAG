"""检索层门面的构造契约。

背景：`rag_engine.py` 里「VectorRetriever + HybridRetriever(权重取自 settings)」
这段构造此前**重复了 3 次**（hybrid 分支、reranked 分支、eval_retriever 属性）。
任何一次改权重来源都可能只改到其中一处 —— 那样「生产检索」与「评估检索」就会用
不同权重，评估结论对不上生产行为，而且不会有任何报错。

收敛到门面后，本文件负责锁住两件事：

1. `settings.RETRIEVER_TYPE` → 具体实现的映射
2. 主链路与评估路径拿到**同一份**权重配置

不需要向量库：vector_store_manager / bm25_retriever 传替身即可。
"""
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.retrieval import build_reranked_retriever, build_retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.vector import VectorRetriever


@pytest.fixture
def deps():
    return MagicMock(name="vector_store_manager"), MagicMock(name="bm25_retriever")


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("vector", VectorRetriever),
        ("hybrid", HybridRetriever),
        ("reranked", RerankedRetriever),
    ],
)
def test_按配置选对实现(monkeypatch, deps, kind, expected):
    monkeypatch.setattr(settings, "RETRIEVER_TYPE", kind)
    assert isinstance(build_retriever(*deps), expected)


def test_未知配置回落向量检索(monkeypatch, deps):
    """配置写错时不能抛异常，应回落最保守的实现。"""
    monkeypatch.setattr(settings, "RETRIEVER_TYPE", "no-such-retriever")
    assert isinstance(build_retriever(*deps), VectorRetriever)


def test_评估路径强制叠加重排(monkeypatch, deps):
    """评估路径必须始终带重排，不受 RETRIEVER_TYPE 影响。"""
    monkeypatch.setattr(settings, "RETRIEVER_TYPE", "hybrid")
    retriever = build_reranked_retriever(*deps)
    assert isinstance(retriever, RerankedRetriever)
    assert isinstance(retriever.hybrid_retriever, HybridRetriever)


def test_两条路径共用同一份权重配置(monkeypatch, deps):
    """主链路 hybrid 与评估路径的内层 hybrid，权重必须一致。

    这是当初重复 3 次构造埋下的隐患所在：改一处忘一处，
    生产与评估就会用不同权重，且无任何报错。
    """
    monkeypatch.setattr(settings, "RETRIEVER_TYPE", "hybrid")
    monkeypatch.setattr(settings, "VECTOR_WEIGHT", 0.31)
    monkeypatch.setattr(settings, "BM25_WEIGHT", 0.69)
    monkeypatch.setattr(settings, "RRF_K", 42)

    main = build_retriever(*deps)
    eval_inner = build_reranked_retriever(*deps).hybrid_retriever

    for retriever in (main, eval_inner):
        assert retriever.vector_weight == 0.31
        assert retriever.bm25_weight == 0.69
        assert retriever.rrf_k == 42


def test_重排构造不加载模型(monkeypatch, deps):
    """构造 RerankedRetriever 必须不触发 CrossEncoder 加载。

    模型走模块级单例懒加载，只有首次 retrieve() 才真正加载。
    若有人把加载挪进 __init__，启动就会白等一次模型加载（数百 MB / 数秒）。
    """
    import app.retrieval.reranked as reranked_module

    monkeypatch.setattr(settings, "RETRIEVER_TYPE", "hybrid")
    monkeypatch.setattr(
        reranked_module, "_reranker_instance", None, raising=False
    )
    build_reranked_retriever(*deps)
    assert reranked_module._reranker_instance is None, "构造阶段不应加载重排模型"


# ── domain_boost 包门面（原 757 行单文件拆为 数据/逻辑 多模块）──────────


def test_domain_boost_门面导出完整名字集():
    """拆包后门面必须导出全部既有名字。

    调用方与测试按名引用（含下划线开头的私有名），少一个就是 ImportError。
    这条用例把名字集固定住，防止以后调整内部文件时漏掉某个再导出。
    """
    from app.retrieval import domain_boost as lg

    expected = {
        "LAW_GRAPH", "CONCEPT_ARTICLE_MAP", "_LAW_SOURCE_MAP", "_SOURCE_LAW_MAP",
        "_load_parent_store", "_exact_lookup_from_parent_store",
        "concept_lookup", "_extract_law_article_from_doc",
        "get_related_articles", "inject_related_articles",
        "article_to_cn", "cn_to_int",
    }
    missing = expected - set(dir(lg))
    assert not missing, f"domain_boost 门面缺少导出：{sorted(missing)}"


def test_domain_boost_数据与逻辑分文件存放():
    """数据表与查找逻辑必须分处不同模块。

    拆分的目的就是「新增一条伴生关系不必在一千行里翻找」。
    若有人把它们又合并回一个文件，这条会变红。
    """
    import ast
    import inspect
    import pathlib

    import app.retrieval.domain_boost.concepts as concepts
    import app.retrieval.domain_boost.lookup as lookup
    import app.retrieval.domain_boost.relations as relations

    # 数据模块里不应有函数定义
    for module in (concepts, relations):
        defined_here = [
            name
            for name, obj in vars(module).items()
            if inspect.isfunction(obj) and obj.__module__ == module.__name__
        ]
        assert not defined_here, f"{module.__name__} 是纯数据模块，却含函数：{defined_here}"

    # 逻辑模块里不应**定义**数据表（import 进来不算 —— lookup 本就需要读它们）
    tree = ast.parse(pathlib.Path(lookup.__file__).read_text(encoding="utf-8"))
    assigned = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    for table in ("LAW_GRAPH", "CONCEPT_ARTICLE_MAP", "_LAW_SOURCE_MAP"):
        assert table not in assigned, f"lookup 是逻辑模块，不该定义数据表 {table}"


def test_domain_boost_再导出的名字与源头同一对象():
    """门面里的名字必须是同一对象，不能是副本。"""
    from app.utils.law_refs import article_to_cn as canonical_article_to_cn
    from app.utils.law_refs import cn_to_int as canonical_cn_to_int
    from app.retrieval import domain_boost as lg
    from app.retrieval.domain_boost import concepts, relations, source_map

    assert lg.LAW_GRAPH is relations.LAW_GRAPH
    assert lg.CONCEPT_ARTICLE_MAP is concepts.CONCEPT_ARTICLE_MAP
    assert lg._LAW_SOURCE_MAP is source_map._LAW_SOURCE_MAP
    assert lg.article_to_cn is canonical_article_to_cn
    assert lg.cn_to_int is canonical_cn_to_int
