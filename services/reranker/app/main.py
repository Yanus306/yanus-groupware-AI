"""
리랭커 서비스 메인 모듈

RAG-002: 리랭킹 서비스
- POST /rerank: 문서를 쿼리 관련성 점수로 재정렬
- GET /health: 헬스체크
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from .model import Reranker

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 인스턴스를 앱 수명주기에 맞춰 관리해 테스트 간 상태 오염을 막는다.
    app.state.reranker = Reranker()
    yield


app = FastAPI(title="Reranker Service", version="1.0.0", lifespan=lifespan)


def _is_test_env() -> bool:
    """pytest 실행 중이면 요청마다 모델을 새로 구성한다."""
    return "PYTEST_CURRENT_TEST" in os.environ


def _get_reranker(app: FastAPI) -> Reranker:
    if _is_test_env() or not hasattr(app.state, "reranker"):
        app.state.reranker = Reranker()
    return app.state.reranker


class DocumentItem(BaseModel):
    """문서 항목 모델"""
    text: str
    source: str


class RerankRequest(BaseModel):
    """리랭킹 요청 모델"""
    query: str
    documents: list[DocumentItem]
    top_n: int = 5


class RerankResult(BaseModel):
    """단일 리랭킹 결과 모델"""
    text: str
    source: str
    score: float


class RerankResponse(BaseModel):
    """리랭킹 응답 모델"""
    results: list[RerankResult]


@app.get("/health")
def health():
    """헬스체크 엔드포인트"""
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest, request: Request):
    """
    문서 리랭킹 엔드포인트

    쿼리와 문서 목록을 받아 관련성 점수로 재정렬합니다.

    Args:
        payload: 쿼리, 문서 목록, top_n

    Returns:
        RerankResponse: 점수 내림차순으로 정렬된 문서 목록
    """
    docs = [{"text": doc.text, "source": doc.source} for doc in payload.documents]
    results = _get_reranker(request.app).rerank(
        payload.query,
        docs,
        top_n=payload.top_n,
    )

    return RerankResponse(
        results=[
            RerankResult(
                text=r["text"],
                source=r["source"],
                score=r["score"],
            )
            for r in results
        ]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
