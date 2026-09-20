"""LLM 槽位连通性预检：跑长任务前先确认三个槽位哪些还活着。

为什么需要它
------------
`EvalRunner` 的 20 题完整评估要跑 10 分钟以上（阶段 1 生成 + 阶段 3 RAGAS），
而它消耗的是**外部额度**。额度耗尽 / 密钥失效 / 网络不通时，如果不在启动前拦一道，
结果是：等 10 分钟，然后在某个题目上崩掉，还分不清是「代码坏了」还是「额度没了」。

本项目实测撞到过 `Free quota exhausted`（见 `app/core/errors.py` 的设计说明）。
本脚本把「额度是否可用」从**运行中途**提前到**启动前**，并区分四类故障：

    OK            —— 该槽位可用
    QUOTA/403     —— 额度耗尽或权限不足
    AUTH/401      —— API Key 无效
    NETWORK       —— 连不上（网络 / 代理 / DNS）
    NOT_CONFIGURED—— 未配置（空的槽位会 fallback，见 config.py 的 _apply_fallbacks）

为什么用 requests 而不是 langchain
----------------------------------
探测要回答的是「这个 endpoint + key + model 组合还能不能用」。
langchain 的 ChatOpenAI 会把原始 HTTP 错误包好几层，反而看不清根因。
这里直接打 OpenAI 兼容协议的 `/chat/completions`，只发 1 个 token，
拿到的是**未经包装的上游错误**。

用法：
    cd backend
    python scripts/check_llm_slots.py              # 检查三个槽位
    python scripts/check_llm_slots.py --slot ragas # 只查一个
    python scripts/check_llm_slots.py --json       # 机器可读输出

退出码：全部可用 0；存在不可用槽位 1（便于接进 CI / 评估前置检查）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import requests  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

# 槽位名 → (密钥, base_url, 模型名) 的取用三元组。
# 与 app/agent/graph.py、app/evaluation/runner.py、app/main.py、app/services/rag_engine.py
# 的 5 个构造点一一对应，改这里必须同步那 5 处（或改为全部走同一个工厂）。
SLOTS = {
    "generate": lambda: (settings.LLM_API_KEY, settings.LLM_API_BASE, settings.LLM_MODEL_NAME),
    "aux": lambda: (settings.EVAL_LLM_API_KEY, settings.EVAL_LLM_API_BASE, settings.EVAL_LLM_MODEL_NAME),
    "ragas": lambda: (settings.RAGAS_API_KEY, settings.RAGAS_API_BASE, settings.RAGAS_MODEL_NAME),
}

# 上游 HTTP 状态码 → 故障分类。403 是百炼额度耗尽的常见表现。
_STATUS_KIND = {
    401: "AUTH",
    403: "QUOTA_OR_FORBIDDEN",
    404: "MODEL_NOT_FOUND",
    429: "RATE_LIMIT",
}


def _classify(status: int, body: str) -> tuple[str, str]:
    """把上游响应归类，并抽取一句可读的根因。"""
    kind = _STATUS_KIND.get(status)
    if kind is None:
        kind = "UPSTREAM_5XX" if status >= 500 else f"HTTP_{status}"
    # 上游错误体通常很长，只取最像原因的一段
    detail = ""
    try:
        payload = json.loads(body)
        err = payload.get("error") or {}
        detail = err.get("message") or err.get("code") or ""
    except Exception:
        detail = body.strip()[:160]
    return kind, str(detail).replace("\n", " ")[:160]


def probe(slot: str, timeout: int = 30) -> dict:
    """探测单个槽位。任何异常都转成结构化结果，不向外抛。"""
    api_key, api_base, model = SLOTS[slot]()

    if not api_key:
        return {"slot": slot, "model": model, "status": "NOT_CONFIGURED",
                "detail": "该槽位未填 API Key（config.py 会 fallback 到上层槽位）"}
    if not api_base:
        return {"slot": slot, "model": model, "status": "NOT_CONFIGURED",
                "detail": "该槽位未填 API_BASE"}

    url = api_base.rstrip("/") + "/chat/completions"
    # max_tokens=1：只验证「能不能用」，不消耗有意义额度。
    # 刻意**不传** extra_body.enable_thinking —— 探测只关心连通性与鉴权，
    # 多带一个参数反而可能引入与生产调用点不一致的失败原因。
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    started = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return {"slot": slot, "model": model, "status": "NETWORK",
                "detail": f"超时（>{timeout}s）", "elapsed": round(time.time() - started, 2)}
    except requests.exceptions.ProxyError as e:
        return {"slot": slot, "model": model, "status": "NETWORK",
                "detail": f"代理错误：{e}（检查 http_proxy 是否把 127.0.0.1 也代理了）",
                "elapsed": round(time.time() - started, 2)}
    except requests.exceptions.RequestException as e:
        return {"slot": slot, "model": model, "status": "NETWORK",
                "detail": f"{type(e).__name__}: {e}", "elapsed": round(time.time() - started, 2)}

    elapsed = round(time.time() - started, 2)
    if resp.status_code == 200:
        return {"slot": slot, "model": model, "status": "OK",
                "detail": "可用", "elapsed": elapsed}

    kind, detail = _classify(resp.status_code, resp.text)
    return {"slot": slot, "model": model, "status": kind,
            "detail": f"HTTP {resp.status_code}: {detail}", "elapsed": elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 槽位连通性预检")
    parser.add_argument("--slot", choices=list(SLOTS), help="只检查指定槽位")
    parser.add_argument("--timeout", type=int, default=30, help="单槽位超时秒数")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    # 复用应用统一的编码处理（UTF-8 + errors=replace）：
    # Windows 默认 GBK，下面的 ✅/❌ 会直接 UnicodeEncodeError 中断脚本。
    # 这是项目里已有的单点解决方案，不要在这里再抄一份 reconfigure。
    setup_logging()

    targets = [args.slot] if args.slot else list(SLOTS)
    results = [probe(s, timeout=args.timeout) for s in targets]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print("=" * 68)
        print("  LLM 槽位预检")
        print("=" * 68)
        for r in results:
            mark = {"OK": "✅"}.get(r["status"], "❌")
            print(f"  {mark} {r['slot']:<9} {r['model']:<22} {r['status']}")
            print(f"      {r['detail']}")
            if "elapsed" in r:
                print(f"      ({r['elapsed']}s)")
        print("=" * 68)
        dead = [r["slot"] for r in results if r["status"] != "OK"]
        if dead:
            print(f"  ⚠️ 不可用槽位：{', '.join(dead)}")
            print("     20 题完整评估会在中途失败，建议先修好这些槽位再启动。")
        else:
            print("  ✅ 全部可用，可以启动评估。")
        print("=" * 68)

    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
