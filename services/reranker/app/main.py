"""
리랭커 서비스 메인 모듈

RAG-002: 리랭킹 서비스
- POST /rerank: 문서를 쿼리 관련성 점수로 재정렬
- GET /health: 헬스체크
"""

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from .model import Reranker

load_dotenv()

app = FastAPI(title="Reranker Service", version="1.0.0")

# 리랭커 모델 초기화 (모듈 로드 시 한 번만)
reranker = Reranker()


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
def rerank(request: RerankRequest):
    """
    문서 리랭킹 엔드포인트

    쿼리와 문서 목록을 받아 관련성 점수로 재정렬합니다.

    Args:
        request: 쿼리, 문서 목록, top_n

    Returns:
        RerankResponse: 점수 내림차순으로 정렬된 문서 목록
    """
    docs = [{"text": doc.text, "source": doc.source} for doc in request.documents]
    results = reranker.rerank(request.query, docs, top_n=request.top_n)

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
