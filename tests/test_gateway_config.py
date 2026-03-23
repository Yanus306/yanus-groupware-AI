"""
Gateway 설정 테스트 모듈

BE-005: gateway 설정 환경변수화 테스트
- 기본 설정값 확인
- 환경변수 오버라이드 확인
"""

import importlib
import os
from unittest.mock import patch


class TestGatewayConfig:
    """Gateway 설정 테스트"""

    def test_settings_have_expected_defaults(self):
        """환경변수가 없을 때 기본값을 사용해야 한다."""
        import services.gateway.app.config as config_module

        with patch.dict(os.environ, {}, clear=True):
            reloaded = importlib.reload(config_module)

        assert reloaded.settings.WORKER_URL == "http://localhost:8001"
        assert reloaded.settings.RERANKER_URL == "http://localhost:8002"
        assert reloaded.settings.VLLM_URL == "http://localhost:8080"
        assert reloaded.settings.QDRANT_HOST == "localhost"
        assert reloaded.settings.QDRANT_PORT == 6333
        assert reloaded.settings.QDRANT_COLLECTION == "documents"
        assert reloaded.settings.VLLM_MODEL == "Qwen/Qwen2.5-7B-Instruct"

    def test_settings_read_environment_overrides(self):
        """환경변수 값으로 gateway 설정을 덮어써야 한다."""
        import services.gateway.app.config as config_module

        with patch.dict(
            os.environ,
            {
                "WORKER_URL": "http://worker:8001",
                "RERANKER_URL": "http://reranker:8002",
                "VLLM_URL": "http://vllm:8080",
                "QDRANT_HOST": "qdrant",
                "QDRANT_PORT": "7000",
                "QDRANT_COLLECTION": "custom-documents",
                "VLLM_MODEL": "Qwen/Qwen3.5-9B",
            },
            clear=True,
        ):
            reloaded = importlib.reload(config_module)

        assert reloaded.settings.WORKER_URL == "http://worker:8001"
        assert reloaded.settings.RERANKER_URL == "http://reranker:8002"
        assert reloaded.settings.VLLM_URL == "http://vllm:8080"
        assert reloaded.settings.QDRANT_HOST == "qdrant"
        assert reloaded.settings.QDRANT_PORT == 7000
        assert reloaded.settings.QDRANT_COLLECTION == "custom-documents"
        assert reloaded.settings.VLLM_MODEL == "Qwen/Qwen3.5-9B"
