"""鉴权冒烟：确认运行中的后端「鉴权到底有没有生效、行为对不对」。

为什么需要它
------------
`AUTH_SECRET` 留空时鉴权是**静默关闭**的 —— 这是有意的本地开发便利，
但也意味着「以为配好了、其实没生效」是很容易发生的状态。而它的后果是
**接口对公网裸奔**。

本脚本对**运行中的后端**做四项检查：
  1. 公开端点 `/health` 在无凭证时可达（它必须公开，否则健康检查会 401）
  2. 受保护端点无凭证 → 401，且带 `WWW-Authenticate: Bearer`
  3. 受保护端点错凭证 → 401
  4. 受保护端点对凭证 → 200

用法：
    cd backend
    python scripts/auth_smoke.py                      # 从 settings 读 AUTH_SECRET
    python scripts/auth_smoke.py --secret <你的密钥>   # 显式指定
    python scripts/auth_smoke.py --base http://127.0.0.1:8000

注意：检查 4 需要密钥；若当前未启用鉴权，脚本会明确报告「未启用」并跳过，
而不会误报为通过。
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

PROTECTED_PATH = "/api/v1/knowledge-base/status"


def request(base: str, path: str, secret: str | None) -> tuple[int, dict, object]:
    """返回 (状态码, 响应体, headers)。

    刻意返回 headers 对象本身（而非 dict）—— HTTPMessage 的大小写不敏感查找
    是它的正确用法；转成 dict 后 `d["WWW-Authenticate"]` 会因服务端发的是小写
    而查不到，从而误报「缺少响应头」。
    """
    headers = {}
    if secret is not None:
        headers["Authorization"] = f"Bearer {secret}"
    req = urllib.request.Request(base + path, headers=headers, method="GET")
    try:
        r = urllib.request.urlopen(req, timeout=15)
        return r.status, json.loads(r.read().decode()), r.headers
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body, e.headers


def main() -> int:
    parser = argparse.ArgumentParser(description="鉴权冒烟")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--secret", default=None, help="不传则从 settings.AUTH_SECRET 读")
    args = parser.parse_args()

    secret = args.secret
    if secret is None:
        try:
            from app.core.config import settings

            secret = settings.AUTH_SECRET or None
        except Exception as e:
            print(f"读取 settings 失败：{type(e).__name__}: {e}")
            return 1

    # 先看后端自己怎么说（/health 会回 auth_enabled）
    try:
        code, body, _ = request(args.base, "/health", None)
    except Exception as e:
        print(f"后端不可达：{type(e).__name__}（请先 python run.py）")
        return 1

    enabled = bool(body.get("auth_enabled"))
    print(f"后端 /health          : HTTP {code}  auth_enabled={enabled}")
    print(f"本机 AUTH_SECRET      : {'已配置' if secret else '未配置'}")
    print()

    if not enabled:
        print("⚠️  鉴权未启用 —— 所有端点无需凭证即可访问。")
        print("    本地开发无妨；若要对公网暴露，请在 .env 中设置 AUTH_SECRET 后重启。")
        # 仍然验证「公开端点可达」这一条
        ok = code == 200
        print(f"\n结果：{'1/1' if ok else '0/1'} 通过（仅检查公开端点）")
        return 0 if ok else 1

    if not secret:
        print("✗ 后端已启用鉴权，但本机没有 AUTH_SECRET —— 无法验证正例。")
        print("  请用 --secret 传入与后端一致的密钥。")
        return 1

    failed = 0
    cases = [
        ("公开端点无凭证应可达",   "/health",        None,    200),
        ("受保护端点无凭证应拒绝", PROTECTED_PATH,  None,    401),
        ("受保护端点错凭证应拒绝", PROTECTED_PATH,  "wrong", 401),
        ("受保护端点对凭证应放行", PROTECTED_PATH,  secret,  200),
    ]
    for label, path, s, expect in cases:
        code, body, _ = request(args.base, path, s)
        ok = code == expect
        failed += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} {label}  期望 {expect} 实际 {code}")

    # 401 的响应形状：RFC 7235 要求 401 必须带 WWW-Authenticate
    code, body, headers = request(args.base, PROTECTED_PATH, None)
    if code == 401:
        challenge = headers.get("WWW-Authenticate")
        has_detail = "detail" in body
        ok = bool(challenge) and has_detail
        failed += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} 401 带 WWW-Authenticate 与统一 detail"
              f"  ({challenge!r}, detail={'有' if has_detail else '无'})")

    total = len(cases) + 1
    print(f"\n结果：{total - failed}/{total} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
