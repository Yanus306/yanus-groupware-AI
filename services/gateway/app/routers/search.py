"""
검색 라우터 모듈

RAG-001: 검색 API 구현
- POST /search: 쿼리를 임베딩하여 Qdrant에서 유사 문서를 검색합니다.

워크플로우:
1. worker POST /embed로 쿼리 임베딩
2. Qdrant에서 유사 벡터 검색
3. 결과 반환
"""

from types import SimpleNamespace
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from httpx import AsyncClient, HTTPStatusError, RequestError
from qdrant_client import QdrantClient

from ..config import settings

router = APIRouter()

httpx = SimpleNamespace(
    AsyncClient=AsyncClient,
    HTTPStatusError=HTTPStatusError,
    RequestError=RequestError,
)


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    """단일 검색 결과 모델"""
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: list[SearchResult]


def create_qdrant_client() -> QdrantClient:
    """현재 설정 기준의 Qdrant 클라이언트를 생성합니다."""
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


async def search_documents(query: str, top_k: int) -> list[SearchResult]:
    """검색 엔드포인트와 /ask 양쪽에서 재사용하는 검색 로직입니다."""
    async with httpx.AsyncClient() as client:
        try:
            embed_resp = await client.post(
                f"{settings.WORKER_URL}/embed",
                json={"texts": [query]},
                timeout=30.0,
            )
            embed_resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Worker 서비스 연결 실패: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Worker 서비스 오류: {str(e)}")

    embeddings = embed_resp.json()["embeddings"]
    query_vector = embeddings[0]

    try:
        hits = create_qdrant_client().search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant 검색 실패: {str(e)}")

    return [
        SearchResult(
            text=hit.payload.get("text", ""),
            source=hit.payload.get("source", ""),
            score=hit.score,
        )
        for hit in hits
    ]


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    문서 검색 엔드포인트

    쿼리를 임베딩하여 Qdrant에서 가장 유사한 문서를 검색합니다.

    Args:
        request: 검색 쿼리 및 top_k

    Returns:
        SearchResponse: 검색 결과 목록
    """
    return SearchResponse(results=await search_documents(request.query, request.top_k))
