"""
인제스트 라우터 모듈

RAG-003: 인덱싱 트리거 API 구현
- POST /ingest/reindex: worker 서비스의 /ingest를 호출하여 재인덱싱
"""

from fastapi import APIRouter, HTTPException
import httpx

from ..config import settings

router = APIRouter()


@router.post("/ingest/reindex")
async def reindex():
    """
    재인덱싱 엔드포인트

    worker 서비스의 /ingest를 호출하여 MinIO 파일을 Qdrant에 재인덱싱합니다.

    Returns:
        dict: worker 서비스의 응답 (status, processed, errors)
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{settings.WORKER_URL}/ingest",
                timeout=300.0,  # 대용량 파일 처리를 위해 긴 타임아웃
            )
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Worker 서비스 연결 실패: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Worker 서비스 오류: {str(e)}")

    return resp.json()
