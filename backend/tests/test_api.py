"""API 层契约测试。

与 test_calculators / test_text_splitter / test_retrieval_metrics 互补：
那三个保证**算法正确**，这里保证 **HTTP 契约稳定** —— 鉴权开关、状态码语义、
统一错误格式、请求追踪头、参数校验边界。改动 routes.py / main.py 时，
这些用例是回归防线。
"""
CHAT_URL = "/api/v1/chat"
STATUS_URL = "/api/v1/knowledge-base/status"
LIST_URL = "/api/v1/documents/list"
UPLOAD_URL = "/api/v1/documents/upload"


# ── 鉴权 ──────────────────────────────────────────────────────


class TestAuth:
    def test_disabled_by_default(self, client):
        """未配置 AUTH_SECRET 时不启用鉴权，本地开发零配置可跑通。"""
        assert client.get(STATUS_URL).status_code == 200

    def test_missing_token_rejected(self, client, auth_secret):
        res = client.get(STATUS_URL)
        assert res.status_code == 401
        assert "未授权" in res.json()["detail"]

    def test_wrong_token_rejected(self, client, auth_secret):
        res = client.get(STATUS_URL, headers={"Authorization": "Bearer wrong-token"})
        assert res.status_code == 401

    def test_correct_token_accepted(self, client, auth_headers):
        assert client.get(STATUS_URL, headers=auth_headers).status_code == 200

    def test_non_bearer_scheme_treated_as_missing(self, client, auth_secret):
        """用 Basic 冒充 Bearer 应视为未提供凭证，而不是放行。"""
        res = client.get(STATUS_URL, headers={"Authorization": f"Basic {auth_secret}"})
        assert res.status_code == 401

    def test_401_advertises_bearer_scheme(self, client, auth_secret):
        res = client.get(STATUS_URL)
        assert res.headers.get("WWW-Authenticate") == "Bearer"

    def test_post_endpoint_also_protected(self, client, auth_secret):
        """鉴权挂在 router 上，POST 端点同样受保护（防新增端点漏挂）。"""
        res = client.post(CHAT_URL, json={"question": "加班费怎么算"})
        assert res.status_code == 401

    def test_health_is_public(self, client, auth_secret):
        """/health 挂在 app 上而非 router 上，探活不受鉴权影响。"""
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["auth_enabled"] is True
        assert body["knowledge_base_ready"] is True


# ── /chat ─────────────────────────────────────────────────────


class TestChat:
    def test_engine_not_ready_returns_503(self, client, fake_engine):
        fake_engine.is_ready = False
        res = client.post(CHAT_URL, json={"question": "加班费怎么算"})
        assert res.status_code == 503
        assert "知识库未就绪" in res.json()["detail"]

    def test_success_returns_full_chat_response(self, client, fake_engine):
        res = client.post(CHAT_URL, json={"question": "加班费怎么算"})
        assert res.status_code == 200

        body = res.json()
        assert body["answer"] == "关于「加班费怎么算」的回答"
        assert body["confidence"] == 0.88
        assert body["sources"][0]["source"] == "劳动法.txt"
        assert body["rag_steps"] == ["rewrite_query", "route_intent", "validate"]
        assert fake_engine.chat_calls[0].question == "加班费怎么算"

    def test_conversation_id_passthrough(self, client):
        res = client.post(
            CHAT_URL, json={"question": "试用期最长多久", "conversation_id": "c-123"}
        )
        assert res.json()["conversation_id"] == "c-123"

    def test_history_accepted_for_followup(self, client, fake_engine):
        res = client.post(
            CHAT_URL,
            json={
                "question": "那赔偿金呢",
                "history": [{"role": "user", "content": "经济补偿金怎么算"}],
            },
        )
        assert res.status_code == 200
        assert fake_engine.chat_calls[0].history[0].content == "经济补偿金怎么算"

    def test_empty_question_rejected(self, client):
        """min_length=1：空问题在进入业务逻辑前就被拦掉。"""
        res = client.post(CHAT_URL, json={"question": ""})
        assert res.status_code == 422
        assert res.json()["detail"] == "请求参数校验失败"

    def test_overlong_question_rejected(self, client):
        """max_length=2000，防止超长输入打爆 prompt。"""
        res = client.post(CHAT_URL, json={"question": "劳" * 2001})
        assert res.status_code == 422

    def test_missing_question_rejected(self, client):
        assert client.post(CHAT_URL, json={}).status_code == 422

    def test_unhandled_exception_becomes_unified_500(self, client, fake_engine):
        """引擎内部异常由全局 handler 兜底，不泄漏堆栈给客户端。"""
        fake_engine.chat_error = RuntimeError("LLM 连接超时")
        res = client.post(CHAT_URL, json={"question": "加班费怎么算"})

        assert res.status_code == 500
        body = res.json()
        assert body["detail"] == "内部服务器错误"
        assert body["request_id"]
        assert "LLM 连接超时" not in res.text


# ── 请求追踪 ──────────────────────────────────────────────────


class TestRequestId:
    def test_generated_when_absent(self, client):
        res = client.get(STATUS_URL)
        assert res.headers.get("X-Request-ID")

    def test_echoed_when_provided(self, client):
        res = client.get(STATUS_URL, headers={"X-Request-ID": "trace-abc-123"})
        assert res.headers["X-Request-ID"] == "trace-abc-123"

    def test_error_body_request_id_matches_header(self, client, fake_engine):
        """500 响应体的 request_id 与响应头一致，便于按一次请求串联日志。"""
        fake_engine.chat_error = RuntimeError("boom")
        res = client.post(
            CHAT_URL, json={"question": "x"}, headers={"X-Request-ID": "trace-500"}
        )
        assert res.status_code == 500
        assert res.json()["request_id"] == "trace-500"
        assert res.headers["X-Request-ID"] == "trace-500"


# ── 知识库与文档 ──────────────────────────────────────────────


class TestKnowledgeBase:
    def test_status_response_shape(self, client):
        res = client.get(STATUS_URL)
        assert res.status_code == 200

        body = res.json()
        assert set(body) == {
            "total_documents",
            "total_chunks",
            "vector_store_exists",
            "embedding_model",
            "llm_model",
        }
        assert body["vector_store_exists"] is False
        assert body["total_documents"] > 0
        assert body["total_chunks"] == 0

    def test_document_list_total_matches_items(self, client):
        body = client.get(LIST_URL).json()
        assert body["total"] == len(body["documents"])
        for doc in body["documents"]:
            assert set(doc) == {"filename", "size", "type"}

    def test_delete_missing_document_404(self, client):
        res = client.delete("/api/v1/documents/不存在的文件.pdf")
        assert res.status_code == 404
        assert res.json()["detail"] == "文件不存在"

    def test_upload_rejects_unsupported_type(self, client):
        res = client.post(UPLOAD_URL, files={"file": ("evil.exe", b"x")})
        assert res.status_code == 400
        assert "不支持的文件类型" in res.json()["detail"]

    def test_build_reports_failure_when_no_documents(self, client, fake_engine):
        fake_engine.build_knowledge_base = lambda rebuild=False: False
        res = client.post("/api/v1/knowledge-base/build", json={"rebuild": False})
        assert res.status_code == 400
        assert "构建失败" in res.json()["detail"]
