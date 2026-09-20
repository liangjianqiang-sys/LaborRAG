"""HuggingFace 模型缓存位置：必须在 huggingface_hub 被导入之前固化。

为什么需要这条规则
------------------
`huggingface_hub` 在 **import 时** 就把 HF_HOME 读成模块级常量
（`constants.HF_HUB_CACHE`）。之后再写 `os.environ["HF_HOME"]` 是**无效的**：
不报错、不抛异常、功能照常，只是模型照旧下到
`C:\\Users\\<用户>\\.cache\\huggingface`。

这正是本项目真实踩过的坑：C 盘的 `.cache` 被清理软件整体清空，约 8GB
模型缓存（bge-m3 + bge-reranker-v2-m3 + bert-base-chinese）消失，
表现为启动时静默重新下载、离线环境下直接起不来。

修法是把 HF_HOME 提前到 `app/core/config.py` 的**模块体**里设置。
而「提前」这件事极易在后续重构中被打回原形 —— 有人把设置挪回
`get_embeddings()` 或 `_get_reranker_singleton()` 里，测试依然全绿。
本文件把这个时机约束变成会变红的规则。

模块级先 import config：这保证了「config 早于 huggingface_hub」的顺序，
否则本测试自己就可能因为导入顺序而误报。
"""
import os
from pathlib import Path

import pytest

from app.core import config  # noqa: F401  # 必须早于 huggingface_hub 导入，见模块 docstring


def test_导入_config_后_HF_HOME_已写入环境():
    """config 导入即应把 HF_HOME 落到 os.environ。"""
    assert config.HF_HOME, "config.HF_HOME 为空 —— 顶部那段 HF 设置被删了？"
    assert os.environ.get("HF_HOME") == config.HF_HOME, (
        f"os.environ['HF_HOME']={os.environ.get('HF_HOME')!r} 与 "
        f"config.HF_HOME={config.HF_HOME!r} 不一致"
    )


def test_huggingface_hub_实际使用的缓存目录落在_HF_HOME_下():
    """核心断言：直接问 huggingface_hub 它自己用哪个目录。

    只断言「我们写进了 os.environ」是不够的 —— **设晚了同样能满足那条**，
    但 huggingface_hub 会继续用它在 import 时固化的旧值。
    所以必须问它的 `constants.HF_HUB_CACHE`。
    """
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    actual = Path(constants.HF_HUB_CACHE).resolve()
    expected = Path(config.HF_HOME).resolve()

    assert actual == expected or expected in actual.parents, (
        f"huggingface_hub 的缓存目录是 {actual}，不在 HF_HOME({expected}) 之下。\n"
        f"说明 HF_HOME 设置得太晚 —— 它必须早于 huggingface_hub 的导入，"
        f"即写在 app/core/config.py 的模块体里，不能挪进任何模型加载函数。"
    )


def test_默认缓存目录避开用户级_cache():
    """默认值不能落在 `C:\\Users\\<用户>\\.cache` 下。

    该目录是清理软件（360 / 电脑管家 / Windows 存储感知）的常见清理目标，
    本项目实测被整体清空过一次。这条会拦住「有人把默认值改回去」的改动。
    只校验默认常量，不校验实际生效值 —— 用户显式配置到哪是他的自由。
    """
    default = Path(config._DEFAULT_HF_HOME).resolve()
    cleanup_prone = (Path.home() / ".cache").resolve()

    assert cleanup_prone not in default.parents and default != cleanup_prone, (
        f"默认 HF_HOME({default}) 位于 {cleanup_prone} 之下 —— "
        f"该目录会被清理软件清空，模型缓存会再次丢失。"
    )
