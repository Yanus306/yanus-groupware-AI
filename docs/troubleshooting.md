# Troubleshooting

이 문서는 실제 개발 및 로컬 검증 과정에서 확인한 운영 이슈와 대응 순서를 정리한다.

## 1. Docker Desktop은 떠 있는데 `docker` 명령이 안 되는 경우

### 증상
- Docker Desktop UI는 열리지만 `docker version`, `docker ps`가 실패한다.
- 오류 예시:
  - `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.`
  - `request returned 500 Internal Server Error ... dockerDesktopLinuxEngine`
- `Get-Service com.docker.service` 결과가 `Stopped`일 수 있다.

### 확인 명령
```powershell
Get-Service com.docker.service
docker version
docker context ls
wsl -l -v
Get-Content "$env:LOCALAPPDATA\Docker\log\host\com.docker.backend.exe.log" -Tail 120
```

### 판단 포인트
- Docker 앱 문제가 아니라 `WSL2/Linux engine`이 `starting`에서 멈춘 경우가 많다.
- 로그에 `WSL 2 cross-distro service`, `Docker Desktop is unable to start`가 보이면 WSL 백엔드 쪽을 먼저 의심한다.

### 대응 순서
1. 관리자 PowerShell에서 `wsl --shutdown`
2. 필요하면 `wsl --update`
3. Docker Desktop 완전 종료 후 다시 실행
4. 그래도 안 되면 Windows 재부팅
5. 계속 동일하면 Docker Desktop `Troubleshoot`의 `Restart Docker Desktop`
6. 마지막 수단으로 `Clean / Purge data` 또는 재설치

## 2. `worker` / `reranker` 컨테이너가 첫 기동 때 바로 `unhealthy`가 되는 경우

### 증상
- `docker compose up -d qdrant worker reranker` 후 컨테이너는 떠 있지만 healthcheck가 빠르게 실패한다.
- 로그는 아래 수준에서 오래 머문다.
  - `INFO: Started server process`
  - `INFO: Waiting for application startup.`
- 컨테이너 내부 Hugging Face 캐시에 `.incomplete` 파일이 남아 있다.

### 원인
- `worker`, `reranker`는 앱 startup 단계에서 모델을 바로 로드한다.
- 첫 실행 시 Hugging Face 모델 다운로드와 초기화가 오래 걸리면, healthcheck가 서비스 준비보다 먼저 실패한다.

### 현재 반영된 대응
- [`docker-compose.yml`](D:/yanus-groupware-AI/docker-compose.yml)에서 `worker`, `reranker` healthcheck에 `start_period: 5m` 추가
- 같은 파일에서 두 서비스가 `huggingface-cache:/root/.cache/huggingface` 볼륨을 공유하도록 설정

### 확인 명령
```powershell
docker ps
docker inspect rag-worker --format "{{json .State.Health}}"
docker inspect rag-reranker --format "{{json .State.Health}}"
docker exec rag-reranker sh -lc "du -sh /root/.cache/huggingface"
```

### 기대 결과
- 첫 기동 중에는 `unhealthy` 대신 `starting` 상태를 유지한다.
- 재기동 시 캐시 재사용으로 다운로드 비용이 줄어든다.

## 3. GPU 서비스 이미지 빌드가 매우 오래 걸리는 경우

### 증상
- `worker`, `reranker` 빌드가 몇 분 이상 걸린다.
- 큰 레이어 다운로드가 계속 보인다.
  - 예: `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`

### 배경
- GPU 런타임 베이스 이미지 자체가 수 GB 단위다.
- 첫 빌드에서는 base image pull 때문에 시간이 오래 걸리는 것이 정상이다.

### 현재 반영된 대응
- [`services/worker/Dockerfile`](D:/yanus-groupware-AI/services/worker/Dockerfile)
- [`services/reranker/Dockerfile`](D:/yanus-groupware-AI/services/reranker/Dockerfile)

위 두 파일은 PyTorch CUDA 런타임 이미지를 사용하고, Docker 빌드에서는 `torch`를 다시 설치하지 않도록 조정되어 있다.

### 확인 포인트
- 첫 빌드가 느린 것은 정상일 수 있다.
- 이후 빌드는 Docker layer cache 덕분에 빨라져야 한다.

## 4. MinIO는 외부 서버를 쓰는데 compose에서 헷갈리는 경우

### 현재 전제
- MinIO는 `docker-compose`로 띄우지 않는다.
- 외부 MinIO 서버를 `.env` 값으로 연결한다.

### 필요한 값
```env
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET=
```

### 참고
- GitHub Secrets에 올린 값은 GitHub Actions용이다.
- 로컬 실행이나 서버 실행에는 여전히 `.env` 또는 환경변수 주입이 필요하다.

## 5. 빠른 점검 순서

```powershell
docker version
docker compose config --services
docker compose build gateway
docker compose build worker
docker compose build reranker
docker compose up -d qdrant worker reranker
docker ps
python -m pytest tests/ -v --ignore=tests/test_infra.py
```
