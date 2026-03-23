"""
Gateway 전체 파이프라인 테스트 모듈

RAG-003: Gateway /ask 엔드포인트 테스트
- POST /ask: 검색 → 리랭킹 → vLLM → 답변 반환
- POST /ingest/reindex: worker /ingest 호출
- 모든 외부 서비스 호출 모킹 처리
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


class TestGatewayAskAPI:
    """Gateway /ask 엔드포인트 테스트"""

    @patch("services.gateway.app.routers.ask.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_ask_returns_answer_and_sources(
        self, mock_qdrant_cls, mock_search_httpx, mock_ask_httpx
    ):
        """POST /ask가 answer와 sources를 반환하는지 테스트"""
        from services.gateway.app.main import app

        # /search 내 worker embed 모킹
        mock_search_client = AsyncMock()
        mock_search_httpx.return_value.__aenter__.return_value = mock_search_client
        mock_embed_resp = MagicMock()
        mock_embed_resp.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_embed_resp.raise_for_status = MagicMock()
        mock_search_client.post.return_value = mock_embed_resp

        # Qdrant 검색 결과 모킹
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_point = MagicMock()
        mock_point.score = 0.9
        mock_point.payload = {"text": "Qdrant 검색 결과", "source": "doc.pdf"}
        mock_qdrant.search.return_value = [mock_point]

        # /ask 내부 httpx 모킹 (reranker + vllm)
        mock_ask_client = AsyncMock()
        mock_ask_httpx.return_value.__aenter__.return_value = mock_ask_client

        # reranker 응답 모킹
        mock_rerank_resp = MagicMock()
        mock_rerank_resp.json.return_value = {
            "results": [{"text": "리랭킹된 문서", "source": "doc.pdf", "score": 0.95}]
        }
        mock_rerank_resp.raise_for_status = MagicMock()

        # vLLM 응답 모킹 (OpenAI 호환 형식)
        mock_vllm_resp = MagicMock()
        mock_vllm_resp.json.return_value = {
            "choices": [
                {"message": {"content": "AI가 생성한 답변입니다."}}
            ]
        }
        mock_vllm_resp.raise_for_status = MagicMock()

        # post 호출 순서: reranker → vllm
        mock_ask_client.post.side_effect = [mock_rerank_resp, mock_vllm_resp]

        client = TestClient(app)
        response = client.post(
            "/ask",
            json={"query": "테스트 질문", "top_k": 5, "top_n": 3},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)

    @patch("services.gateway.app.routers.ask.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_ask_sources_have_required_fields(
        self, mock_qdrant_cls, mock_search_httpx, mock_ask_httpx
    ):
        """POST /ask 응답의 sources에 text와 source 필드가 있는지 테스트"""
        from services.gateway.app.main import app

        # embed 모킹
        mock_search_client = AsyncMock()
        mock_search_httpx.return_value.__aenter__.return_value = mock_search_client
        mock_embed_resp = MagicMock()
        mock_embed_resp.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_embed_resp.raise_for_status = MagicMock()
        mock_search_client.post.return_value = mock_embed_resp

        # Qdrant 모킹
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_point = MagicMock()
        mock_point.score = 0.85
        mock_point.payload = {"text": "텍스트", "source": "file.txt"}
        mock_qdrant.search.return_value = [mock_point]

        # ask 내부 httpx 모킹
        mock_ask_client = AsyncMock()
        mock_ask_httpx.return_value.__aenter__.return_value = mock_ask_client

        mock_rerank_resp = MagicMock()
        mock_rerank_resp.json.return_value = {
            "results": [{"text": "텍스트", "source": "file.txt", "score": 0.85}]
        }
        mock_rerank_resp.raise_for_status = MagicMock()

        mock_vllm_resp = MagicMock()
        mock_vllm_resp.json.return_value = {
            "choices": [{"message": {"content": "답변"}}]
        }
        mock_vllm_resp.raise_for_status = MagicMock()

        mock_ask_client.post.side_effect = [mock_rerank_resp, mock_vllm_resp]

        client = TestClient(app)
        response = client.post("/ask", json={"query": "질문"})

        data = response.json()
        sources = data["sources"]
        assert len(sources) > 0
        for source in sources:
            assert "text" in source
            assert "source" in source

    @patch("services.gateway.app.routers.ask.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_ask_answer_contains_llm_response(
        self, mock_qdrant_cls, mock_search_httpx, mock_ask_httpx
    ):
        """POST /ask 응답의 answer가 vLLM 응답 내용인지 테스트"""
        from services.gateway.app.main import app

        mock_search_client = AsyncMock()
        mock_search_httpx.return_value.__aenter__.return_value = mock_search_client
        mock_embed_resp = MagicMock()
        mock_embed_resp.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_embed_resp.raise_for_status = MagicMock()
        mock_search_client.post.return_value = mock_embed_resp

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_point = MagicMock()
        mock_point.score = 0.8
        mock_point.payload = {"text": "검색 결과", "source": "source.pdf"}
        mock_qdrant.search.return_value = [mock_point]

        mock_ask_client = AsyncMock()
        mock_ask_httpx.return_value.__aenter__.return_value = mock_ask_client

        expected_answer = "이것은 AI가 생성한 정확한 답변입니다."
        mock_rerank_resp = MagicMock()
        mock_rerank_resp.json.return_value = {
            "results": [{"text": "검색 결과", "source": "source.pdf", "score": 0.9}]
        }
        mock_rerank_resp.raise_for_status = MagicMock()

        mock_vllm_resp = MagicMock()
        mock_vllm_resp.json.return_value = {
            "choices": [{"message": {"content": expected_answer}}]
        }
        mock_vllm_resp.raise_for_status = MagicMock()

        mock_ask_client.post.side_effect = [mock_rerank_resp, mock_vllm_resp]

        client = TestClient(app)
        response = client.post("/ask", json={"query": "질문"})

        assert response.json()["answer"] == expected_answer

    @patch("services.gateway.app.routers.ask.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_ask_default_params(
        self, mock_qdrant_cls, mock_search_httpx, mock_ask_httpx
    ):
        """POST /ask의 top_k=10, top_n=5 기본값 적용 테스트"""
        from services.gateway.app.main import app

        mock_search_client = AsyncMock()
        mock_search_httpx.return_value.__aenter__.return_value = mock_search_client
        mock_embed_resp = MagicMock()
        mock_embed_resp.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_embed_resp.raise_for_status = MagicMock()
        mock_search_client.post.return_value = mock_embed_resp

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.search.return_value = []

        mock_ask_client = AsyncMock()
        mock_ask_httpx.return_value.__aenter__.return_value = mock_ask_client

        mock_rerank_resp = MagicMock()
        mock_rerank_resp.json.return_value = {"results": []}
        mock_rerank_resp.raise_for_status = MagicMock()

        mock_vllm_resp = MagicMock()
        mock_vllm_resp.json.return_value = {
            "choices": [{"message": {"content": "답변"}}]
        }
        mock_vllm_resp.raise_for_status = MagicMock()

        mock_ask_client.post.side_effect = [mock_rerank_resp, mock_vllm_resp]

        client = TestClient(app)
        response = client.post("/ask", json={"query": "질문만 있음"})

        assert response.status_code == 200
        # Qdrant가 top_k=10 기본값으로 호출되었는지 확인
        call_kwargs = mock_qdrant.search.call_args.kwargs
        assert call_kwargs.get("limit") == 10


class TestGatewayIngestAPI:
    """Gateway /ingest 엔드포인트 테스트"""

    @patch("services.gateway.app.routers.ingest.httpx.AsyncClient")
    def test_reindex_calls_worker_ingest(self, mock_httpx_cls):
        """POST /ingest/reindex가 worker /ingest를 호출하는지 테스트"""
        from services.gateway.app.main import app

        mock_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok", "processed": 5, "errors": []}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        client = TestClient(app)
        response = client.post("/ingest/reindex")

        assert response.status_code == 200
        mock_client.post.assert_called_once()
        call_url = mock_client.post.call_args.args[0]
        assert "/ingest" in call_url

    @patch("services.gateway.app.routers.ingest.httpx.AsyncClient")
    def test_reindex_returns_worker_response(self, mock_httpx_cls):
        """POST /ingest/reindex가 worker 응답을 그대로 반환하는지 테스트"""
        from services.gateway.app.main import app

        mock_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_client

        worker_response = {"status": "ok", "processed": 3, "errors": []}
        mock_resp = MagicMock()
        mock_resp.json.return_value = worker_response
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        client = TestClient(app)
        response = client.post("/ingest/reindex")

        assert response.status_code == 200
        assert response.json() == worker_response


class TestGatewayHealth:
    """Gateway 헬스체크 테스트"""

    def test_health_endpoint(self):
        """GET /health 엔드포인트 테스트"""
        from services.gateway.app.main import app
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
