"""
Gateway 서비스 메인 모듈

RAG-001: 검색 API 게이트웨이
- POST /search: 문서 검색
- GET /health: 헬스체크
"""

from fastapi import FastAPI
from dotenv import load_dotenv

from .routers import search as search_router

load_dotenv()

app = FastAPI(title="Gateway Service", version="1.0.0")

# 라우터 등록
app.include_router(search_router.router)


@app.get("/health")
def health():
    """헬스체크 엔드포인트"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
