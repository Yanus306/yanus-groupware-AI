# yanus-groupware-AI

Python 기반 RAG 실험 저장소입니다. 현재는 Spring Boot 없이 `worker`, `reranker`, `gateway`, `vLLM`, `Qdrant`로 질의응답 파이프라인을 구성합니다.

## Services
- `worker`: MinIO 문서를 읽어 텍스트 추출, 청킹, 임베딩, Qdrant 적재
- `reranker`: 검색 결과를 관련성 점수로 재정렬
- `gateway`: `/search`, `/ask`, `/ingest/reindex`, `/health` 제공
- `qdrant`: 벡터 저장소
- `vllm`: OpenAI 호환 생성 API

## Environment
- 예시 설정은 `.env.example`에 있습니다.
- MinIO는 외부 서버를 사용하고, Qdrant/worker/reranker/gateway/vLLM은 로컬 `docker-compose` 기준입니다.

## Run
```bash
docker compose up --build
```

주요 포트:
- `8000` gateway
- `8001` worker
- `8002` reranker
- `8080` vLLM
- `6333` qdrant

## Test
```bash
python -m pytest tests/ -v --ignore=tests/test_infra.py
```

`tests/test_infra.py`는 실제 MinIO/Qdrant 연결이 필요할 때만 별도로 실행합니다.

## Workflow
- 기능 작업은 이슈 생성 후 `dev` 기준 브랜치에서 진행합니다.
- 커밋은 `test -> feat/fix -> refactor` 순서를 유지합니다.
- feature/fix/docs/chore 브랜치 PR은 `dev`로 병합합니다.
- 배포 시점에는 `dev -> main` PR만 만들고, `main` 머지는 수동으로 진행합니다.

## Current Scope
- 검색, 리랭킹, 질의응답, DOCX 포함 문서 적재, 의존 서비스 헬스체크까지 구현돼 있습니다.
- 다음 우선순위는 실제 통합 시나리오 검증과 운영 문서 보강입니다.
