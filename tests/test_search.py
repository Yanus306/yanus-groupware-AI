"""
검색 API 테스트 모듈

RAG-001: Gateway 검색 엔드포인트 테스트
- POST /search: 쿼리 임베딩 → Qdrant 검색 → 결과 반환
- GET /health: 헬스체크 테스트
모든 외부 서비스 호출은 httpx 모킹으로 처리
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


class TestSearchAPI:
    """검색 API 엔드포인트 테스트"""

    @patch("services.gateway.app.main.httpx.AsyncClient")
    @patch("services.gateway.app.main.create_qdrant_client")
    def test_health_endpoint(self, mock_qdrant_factory, mock_httpx_cls):
        """GET /health 엔드포인트 테스트"""
        from services.gateway.app.main import app

        mock_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_client

        worker_resp = MagicMock()
        worker_resp.raise_for_status = MagicMock()
        reranker_resp = MagicMock()
        reranker_resp.raise_for_status = MagicMock()
        vllm_resp = MagicMock()
        vllm_resp.raise_for_status = MagicMock()
        mock_client.get.side_effect = [worker_resp, reranker_resp, vllm_resp]

        mock_qdrant = MagicMock()
        mock_qdrant_factory.return_value = mock_qdrant
        mock_qdrant.get_collections.return_value = MagicMock()

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_search_returns_results(self, mock_qdrant_cls, mock_httpx_cls):
        """POST /search 엔드포인트가 결과를 반환하는지 테스트"""
        from services.gateway.app.main import app

        # worker /embed 호출 모킹
        mock_http_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_http_client
        mock_embed_response = MagicMock()
        mock_embed_response.json.return_value = {
            "embeddings": [[0.1] * 1024]
        }
        mock_embed_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_embed_response

        # Qdrant 검색 결과 모킹
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        mock_point = MagicMock()
        mock_point.score = 0.95
        mock_point.payload = {
            "text": "테스트 문서 내용",
            "source": "test.pdf",
            "chunk_index": 0,
            "file_type": "pdf",
        }
        mock_qdrant.search.return_value = [mock_point]

        client = TestClient(app)
        response = client.post("/search", json={"query": "테스트 쿼리", "top_k": 5})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0

    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_search_result_has_required_fields(self, mock_qdrant_cls, mock_httpx_cls):
        """검색 결과에 필수 필드(text, source, score)가 있는지 테스트"""
        from services.gateway.app.main import app

        # worker /embed 호출 모킹
        mock_http_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_http_client
        mock_embed_response = MagicMock()
        mock_embed_response.json.return_value = {
            "embeddings": [[0.2] * 1024]
        }
        mock_embed_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_embed_response

        # Qdrant 모킹
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        mock_point = MagicMock()
        mock_point.score = 0.88
        mock_point.payload = {
            "text": "문서 텍스트",
            "source": "doc.txt",
            "chunk_index": 1,
            "file_type": "txt",
        }
        mock_qdrant.search.return_value = [mock_point]

        client = TestClient(app)
        response = client.post("/search", json={"query": "쿼리", "top_k": 3})

        assert response.status_code == 200
        result = response.json()["results"][0]
        assert "text" in result
        assert "source" in result
        assert "score" in result

    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_search_with_default_top_k(self, mock_qdrant_cls, mock_httpx_cls):
        """top_k 기본값(10)으로 검색하는지 테스트"""
        from services.gateway.app.main import app

        mock_http_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_http_client
        mock_embed_response = MagicMock()
        mock_embed_response.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_embed_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_embed_response

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.search.return_value = []

        client = TestClient(app)
        response = client.post("/search", json={"query": "기본 쿼리"})

        assert response.status_code == 200
        # top_k 기본값이 10으로 설정되어 Qdrant 검색 호출됨
        call_kwargs = mock_qdrant.search.call_args.kwargs
        assert call_kwargs.get("limit") == 10

    @patch("services.gateway.app.routers.search.httpx.AsyncClient")
    @patch("services.gateway.app.routers.search.QdrantClient")
    def test_search_empty_results(self, mock_qdrant_cls, mock_httpx_cls):
        """검색 결과가 없을 때 빈 리스트를 반환하는지 테스트"""
        from services.gateway.app.main import app

        mock_http_client = AsyncMock()
        mock_httpx_cls.return_value.__aenter__.return_value = mock_http_client
        mock_embed_response = MagicMock()
        mock_embed_response.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_embed_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_embed_response

        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.search.return_value = []

        client = TestClient(app)
        response = client.post("/search", json={"query": "없는 내용"})

        assert response.status_code == 200
        assert response.json()["results"] == []
