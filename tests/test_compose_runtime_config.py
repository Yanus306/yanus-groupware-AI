"""
Docker Compose 런타임 설정 테스트 모듈

BE-009: 실제 컨테이너 기동 전제와 compose 설정을 정렬한다.
- qdrant 의존성은 service_started 수준으로 관리해야 함
- vLLM 모델명은 환경변수로 주입되어야 함
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
        assert vllm_command[1] == "${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
