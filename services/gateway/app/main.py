"""
Gateway 서비스 메인 모듈

RAG-001 + RAG-003: 검색 및 질의응답 게이트웨이
- POST /search: 문서 검색
- POST /ask: 질의응답 (검색 → 리랭킹 → LLM)
- POST /ingest/reindex: worker를 통한 재인덱싱
- GET /health: 헬스체크
"""

from types import SimpleNamespace
from fastapi import FastAPI, Response
from dotenv import load_dotenv
from httpx import AsyncClient, HTTPStatusError, RequestError
from qdrant_client import QdrantClient

from .config import settings
from .routers import search as search_router
from .routers import ask as ask_router
from .routers import ingest as ingest_router

load_dotenv()

httpx = SimpleNamespace(
    AsyncClient=AsyncClient,
    HTTPStatusError=HTTPStatusError,
    RequestError=RequestError,
)

app = FastAPI(title="Gateway Service", version="1.0.0")

# 라우터 등록
app.include_router(search_router.router)
app.include_router(ask_router.router)
app.include_router(ingest_router.router)


def create_qdrant_client() -> QdrantClient:
    """헬스체크에서 사용할 Qdrant 클라이언트를 생성합니다."""
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


async def _check_http_dependency(client, name: str, url: str) -> dict:
    """HTTP 의존 서비스 헬스체크 결과를 표준 형식으로 반환합니다."""
    try:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return {"status": "ok", "url": url}
    except httpx.RequestError as exc:
        return {"status": "error", "url": url, "detail": str(exc)}
    except httpx.HTTPStatusError as exc:
        return {"status": "error", "url": url, "detail": str(exc)}


def _check_qdrant() -> dict:
    """Qdrant 연결 상태를 확인합니다."""
    try:
        create_qdrant_client().get_collections()
        return {
            "status": "ok",
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
        }
    except Exception as exc:
        return {
            "status": "error",
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
            "detail": str(exc),
        }


@app.get("/health")
async def health(response: Response):
    """게이트웨이와 의존 서비스 상태를 함께 반환하는 헬스체크 엔드포인트"""
    async with httpx.AsyncClient() as client:
        services = {
            "worker": await _check_http_dependency(client, "worker", f"{settings.WORKER_URL}/health"),
            "reranker": await _check_http_dependency(client, "reranker", f"{settings.RERANKER_URL}/health"),
            "vllm": await _check_http_dependency(client, "vllm", f"{settings.VLLM_URL}/health"),
            "qdrant": _check_qdrant(),
        }

    overall_status = "ok" if all(item["status"] == "ok" for item in services.values()) else "degraded"
    if overall_status != "ok":
        response.status_code = 503

    return {
        "status": overall_status,
        "service": "gateway",
        "services": services,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
