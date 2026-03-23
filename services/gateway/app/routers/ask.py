"""
Ask 라우터 모듈

RAG-003: 질의응답 API 구현
- POST /ask: 검색 → 리랭킹 → vLLM → 답변 반환

워크플로우:
1. /search 라우터를 통해 Qdrant 검색
2. reranker 서비스로 결과 재정렬
3. 컨텍스트로 프롬프트 구성
4. vLLM으로 답변 생성
5. 답변과 출처 반환
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

from ..config import settings
from . import search as search_router

router = APIRouter()


class AskRequest(BaseModel):
    """질의응답 요청 모델"""
    query: str
    top_k: int = 10
    top_n: int = 5


class SourceItem(BaseModel):
    """출처 항목 모델"""
    text: str
    source: str


class AskResponse(BaseModel):
    """질의응답 응답 모델"""
    answer: str
    sources: list[SourceItem]


def _build_prompt(query: str, contexts: list[dict]) -> str:
    """
    LLM 프롬프트를 구성합니다.

    Args:
        query: 사용자 질문
        contexts: 검색된 문서 컨텍스트 목록

    Returns:
        str: 구성된 프롬프트
    """
    context_text = "\n\n".join(
        f"[출처: {ctx['source']}]\n{ctx['text']}"
        for ctx in contexts
    )

    prompt = f"""다음 문서들을 참고하여 질문에 답변하세요.

=== 참고 문서 ===
{context_text}

=== 질문 ===
{query}

=== 답변 ==="""
    return prompt


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    질의응답 엔드포인트

    검색 → 리랭킹 → LLM 생성의 전체 파이프라인을 실행합니다.

    Args:
        request: 질문, top_k, top_n

    Returns:
        AskResponse: 생성된 답변 및 출처 목록
    """
    # 1. 검색 (search 라우터의 로직 재사용)
    search_req = search_router.SearchRequest(
        query=request.query,
        top_k=request.top_k,
    )
    search_resp = await search_router.search(search_req)
    search_results = search_resp.results

    async with httpx.AsyncClient() as client:
        # 2. 리랭킹
        rerank_docs = [
            {"text": r.text, "source": r.source}
            for r in search_results
        ]

        try:
            rerank_resp = await client.post(
                f"{settings.RERANKER_URL}/rerank",
                json={
                    "query": request.query,
                    "documents": rerank_docs,
                    "top_n": request.top_n,
                },
                timeout=30.0,
            )
            rerank_resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Reranker 서비스 연결 실패: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Reranker 서비스 오류: {str(e)}")

        reranked = rerank_resp.json()["results"]

        # 3. 프롬프트 구성
        prompt = _build_prompt(request.query, reranked)

        # 4. vLLM으로 답변 생성 (OpenAI 호환 API)
        try:
            vllm_resp = await client.post(
                f"{settings.VLLM_URL}/v1/chat/completions",
                json={
                    "model": settings.VLLM_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                },
                timeout=60.0,
            )
            vllm_resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"vLLM 서비스 연결 실패: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"vLLM 서비스 오류: {str(e)}")

    answer = vllm_resp.json()["choices"][0]["message"]["content"]

    # 5. 응답 구성
    sources = [
        SourceItem(text=doc["text"], source=doc["source"])
        for doc in reranked
    ]

    return AskResponse(answer=answer, sources=sources)
