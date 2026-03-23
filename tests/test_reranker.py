"""
리랭커 서비스 테스트 모듈

RAG-002: 리랭커 서비스 테스트
- POST /rerank: 문서를 쿼리와의 관련성 점수로 재정렬
- GET /health: 헬스체크
모든 모델 호출은 unittest.mock으로 모킹
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestRerankerModel:
    """리랭커 모델 테스트"""

    @patch("services.reranker.app.model.CrossEncoder")
    def test_model_loads_correct_model_name(self, mock_ce_cls):
        """올바른 모델명(dragonkue/bge-reranker-v2-m3-ko)으로 로드하는지 테스트"""
        from services.reranker.app.model import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model

        reranker = Reranker()

        # 올바른 모델명으로 CrossEncoder가 초기화되었는지 확인
        call_args = mock_ce_cls.call_args
        assert "dragonkue/bge-reranker-v2-m3-ko" in call_args.args or \
               call_args.kwargs.get("model_name") == "dragonkue/bge-reranker-v2-m3-ko"

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_returns_sorted_by_score(self, mock_ce_cls):
        """리랭킹 결과가 점수 내림차순으로 정렬되는지 테스트"""
        from services.reranker.app.model import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model

        # 낮은 점수 먼저, 높은 점수 나중에 반환
        mock_model.predict.return_value = [0.3, 0.9, 0.6]

        reranker = Reranker()
        query = "테스트 쿼리"
        documents = [
            {"text": "관련도 낮은 문서", "source": "low.txt"},
            {"text": "가장 관련 높은 문서", "source": "high.txt"},
            {"text": "중간 관련도 문서", "source": "mid.txt"},
        ]
        results = reranker.rerank(query, documents, top_n=3)

        # 점수 내림차순 정렬 확인
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0]["source"] == "high.txt"

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_top_n_limits_results(self, mock_ce_cls):
        """top_n 파라미터가 결과 수를 제한하는지 테스트"""
        from services.reranker.app.model import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        mock_model.predict.return_value = [0.3, 0.9, 0.6, 0.8, 0.1]

        reranker = Reranker()
        query = "쿼리"
        documents = [
            {"text": f"문서 {i}", "source": f"doc{i}.txt"}
            for i in range(5)
        ]
        results = reranker.rerank(query, documents, top_n=3)

        assert len(results) == 3

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_result_has_score_field(self, mock_ce_cls):
        """리랭킹 결과에 score 필드가 있는지 테스트"""
        from services.reranker.app.model import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        mock_model.predict.return_value = [0.75]

        reranker = Reranker()
        results = reranker.rerank(
            "쿼리",
            [{"text": "문서", "source": "doc.txt"}],
            top_n=1,
        )

        assert "score" in results[0]
        assert isinstance(results[0]["score"], float)

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_preserves_original_fields(self, mock_ce_cls):
        """리랭킹 결과에 원본 필드(text, source)가 보존되는지 테스트"""
        from services.reranker.app.model import Reranker

        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        mock_model.predict.return_value = [0.9]

        reranker = Reranker()
        doc = {"text": "원본 텍스트", "source": "original.pdf"}
        results = reranker.rerank("쿼리", [doc], top_n=1)

        assert results[0]["text"] == "원본 텍스트"
        assert results[0]["source"] == "original.pdf"


class TestRerankerAPI:
    """리랭커 FastAPI 엔드포인트 테스트"""

    @patch("services.reranker.app.model.CrossEncoder")
    def test_health_endpoint(self, mock_ce_cls):
        """GET /health 엔드포인트 테스트"""
        mock_ce_cls.return_value = MagicMock()

        from services.reranker.app.main import app
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "reranker"
        assert data["components"]["reranker"] == "ok"

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_endpoint_returns_results(self, mock_ce_cls):
        """POST /rerank 엔드포인트가 결과를 반환하는지 테스트"""
        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        mock_model.predict.return_value = [0.8, 0.4, 0.9]

        from services.reranker.app.main import app
        client = TestClient(app)

        payload = {
            "query": "테스트 쿼리",
            "documents": [
                {"text": "문서 1", "source": "doc1.txt"},
                {"text": "문서 2", "source": "doc2.txt"},
                {"text": "문서 3", "source": "doc3.txt"},
            ],
            "top_n": 2,
        }
        response = client.post("/rerank", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_endpoint_sorted_by_score(self, mock_ce_cls):
        """POST /rerank 결과가 점수 내림차순으로 정렬되는지 테스트"""
        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        mock_model.predict.return_value = [0.3, 0.95, 0.7]

        from services.reranker.app.main import app
        client = TestClient(app)

        payload = {
            "query": "쿼리",
            "documents": [
                {"text": "낮은 관련도", "source": "low.txt"},
                {"text": "높은 관련도", "source": "high.txt"},
                {"text": "중간 관련도", "source": "mid.txt"},
            ],
            "top_n": 3,
        }
        response = client.post("/rerank", json=payload)
        results = response.json()["results"]

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("services.reranker.app.model.CrossEncoder")
    def test_rerank_endpoint_default_top_n(self, mock_ce_cls):
        """top_n 기본값(5)이 적용되는지 테스트"""
        mock_model = MagicMock()
        mock_ce_cls.return_value = mock_model
        mock_model.predict.return_value = [0.5, 0.7, 0.3]

        from services.reranker.app.main import app
        client = TestClient(app)

        payload = {
            "query": "쿼리",
            "documents": [
                {"text": f"문서 {i}", "source": f"doc{i}.txt"}
                for i in range(3)
            ],
        }
        response = client.post("/rerank", json=payload)

        assert response.status_code == 200
        # 문서가 3개이므로 top_n=5 기본값이어도 3개 반환
        assert len(response.json()["results"]) == 3
