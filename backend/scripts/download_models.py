"""预下载运行时所需模型到 HF_HOME（默认 D:/hf_cache）。

用途
----
模型首次加载要下载约 4.5GB（bge-m3 嵌入 ~2.27GB + bge-reranker-v2-m3 ~2.2GB），
如果让它在「第一次聊天 / 第一次评估」时才下载，用户会误以为系统卡死。
本脚本提前把两个模型拉全，之后的服务启动就只是从本地缓存加载。

下载位置由 `app/core/config.py` 模块体里的 HF_HOME 决定（见该文件顶部注释），
必须早于 huggingface_hub 导入。本脚本先 import config，再触发下载，
保证顺序正确。

用法：
    cd backend
    python scripts/download_models.py            # 拉两个模型
    python scripts/download_models.py --skip-reranker   # 只要嵌入模型（纯聊天够用）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Windows 控制台默认 GBK，打印 ✅ 会 UnicodeEncodeError 中断脚本
# （本项目 README「日志编码」一节记录过这个坑）。与其它脚本一致：
# 重配为 UTF-8 + errors=replace，GBK 终端下不可编码字符替换为 ?，不崩。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 先 import config —— 它会在模块体里把 HF_HOME / HF_ENDPOINT 写进 os.environ，
# 且必须早于任何 huggingface_hub 导入（否则缓存位置会回落到 C 盘）。
from app.core.config import settings, HF_HOME  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_embeddings() -> None:
    _log(f"下载嵌入模型 {settings.EMBEDDING_MODEL_NAME} → {HF_HOME}")
    t = time.time()
    from app.knowledge.embeddings import get_embeddings

    emb = get_embeddings()
    emb.embed_query("warmup")  # 触发真正的权重加载
    _log(f"嵌入模型就绪，耗时 {time.time() - t:.1f}s")


def download_reranker() -> None:
    _log(f"下载重排模型 {settings.RERANKER_MODEL_NAME} → {HF_HOME}")
    t = time.time()
    from app.retrieval.reranked import _get_reranker_singleton

    _get_reranker_singleton()
    _log(f"重排模型就绪，耗时 {time.time() - t:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="预下载运行时模型到 HF_HOME")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="只下载嵌入模型（纯聊天、不跑评估时够用）")
    args = parser.parse_args()

    _log(f"HF_HOME = {HF_HOME}")

    download_embeddings()
    if not args.skip_reranker:
        download_reranker()

    _log("全部模型就绪 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
