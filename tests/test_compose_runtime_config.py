"""
Docker Compose 런타임 설정 테스트 모듈

BE-009, BE-010: 실제 컨테이너 기동 전제와 compose 설정을 정렬한다.
- qdrant 의존성은 service_started 수준으로 관리해야 함
- vLLM 모델명은 환경변수로 주입되어야 함
- 모델 서비스는 긴 startup을 감안한 healthcheck 설정이 필요함
- Hugging Face 캐시는 재기동 간 재사용되어야 함
"""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def _load_compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


class TestComposeRuntimeConfig:
    def test_qdrant_does_not_require_internal_healthcheck(self):
        compose = _load_compose()
        qdrant_service = compose["services"]["qdrant"]

        assert "healthcheck" not in qdrant_service

    def test_qdrant_dependencies_wait_for_service_started(self):
        compose = _load_compose()

        worker_condition = compose["services"]["worker"]["depends_on"]["qdrant"]["condition"]
        gateway_condition = compose["services"]["gateway"]["depends_on"]["qdrant"]["condition"]

        assert worker_condition == "service_started"
        assert gateway_condition == "service_started"

    def test_vllm_command_uses_environment_model(self):
        compose = _load_compose()
        vllm_command = compose["services"]["vllm"]["command"]

        assert vllm_command[0] == "--model"
        assert vllm_command[1] == "${VLLM_MODEL:-Qwen/Qwen3.5-9B}"

    def test_model_services_have_healthcheck_start_period(self):
        compose = _load_compose()

        worker_healthcheck = compose["services"]["worker"]["healthcheck"]
        reranker_healthcheck = compose["services"]["reranker"]["healthcheck"]

        assert worker_healthcheck["start_period"] == "5m"
        assert reranker_healthcheck["start_period"] == "5m"

    def test_model_services_share_huggingface_cache_volume(self):
        compose = _load_compose()

        worker_volumes = compose["services"]["worker"]["volumes"]
        reranker_volumes = compose["services"]["reranker"]["volumes"]

        assert "huggingface-cache:/root/.cache/huggingface" in worker_volumes
        assert "huggingface-cache:/root/.cache/huggingface" in reranker_volumes
