"""法律名称 ↔ 向量库 source 文件名的双向映射 —— 再导出。

数据表的唯一真相源已下沉到 `app/utils/law_refs.py`（最底层，零依赖）。

为什么下沉：本包（domain_boost）与 `app/utils/text.py` 都需要这张表，
若表留在这里，就成了 `utils → retrieval`（上层）的反向依赖，构成包级循环。
放在最底层后，两侧都向下引用，依赖方向一致。

本模块仅作再导出，保留既有 `from app.retrieval.domain_boost.source_map import ...`
的引用路径。**不要在别处重新定义这张表。**
"""
from app.utils.law_refs import _LAW_SOURCE_MAP, _SOURCE_LAW_MAP  # noqa: F401

__all__ = ["_LAW_SOURCE_MAP", "_SOURCE_LAW_MAP"]
