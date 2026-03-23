"""
Docker 빌드 설정 테스트 모듈

BE-007, BE-008: Docker 이미지 설정 테스트
- worker/reranker가 PyTorch CUDA 런타임 이미지를 사용해야 함
- Docker 빌드 시 torch 재설치를 피해야 함
- healthcheck를 쓰는 서비스는 curl을 포함해야 함
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestGpuServiceDockerfiles:
    """GPU 서비스 Dockerfile 설정 테스트"""

    def test_worker_uses_pytorch_runtime_base(self):
        dockerfile = (ROOT / "services" / "worker" / "Dockerfile").read_text(encoding="utf-8")

        assert "FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime" in dockerfile

    def test_reranker_uses_pytorch_runtime_base(self):
        dockerfile = (ROOT / "services" / "reranker" / "Dockerfile").read_text(encoding="utf-8")

        assert "FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime" in dockerfile

    def test_dockerfiles_skip_torch_reinstall(self):
        worker_dockerfile = (ROOT / "services" / "worker" / "Dockerfile").read_text(encoding="utf-8")
        reranker_dockerfile = (ROOT / "services" / "reranker" / "Dockerfile").read_text(encoding="utf-8")

        assert "grep -v '^torch==' requirements.txt" in worker_dockerfile
        assert "grep -v '^torch==' requirements.txt" in reranker_dockerfile

    def test_gateway_installs_curl_for_healthcheck(self):
        dockerfile = (ROOT / "services" / "gateway" / "Dockerfile").read_text(encoding="utf-8")

        assert "apt-get install -y --no-install-recommends curl" in dockerfile
