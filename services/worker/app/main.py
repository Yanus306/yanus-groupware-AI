"""
Worker 서비스 메인 모듈

DOC-001: 문서 수집 및 임베딩 파이프라인
- POST /ingest: MinIO에서 파일을 가져와 Qdrant에 인덱싱
- POST /embed: 텍스트 임베딩 요청
- GET /health: 헬스체크
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from .loader import MinIOLoader
from .extractor import TextExtractor
from .chunker import TextChunker
from .embedder import Embedder
from .indexer import QdrantIndexer

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 테스트마다 TestClient가 새로 생성될 때 상태를 재초기화한다.
    app.state.loader = MinIOLoader()
    app.state.extractor = TextExtractor()
    app.state.chunker = TextChunker(chunk_size=512, overlap=50)
    app.state.embedder = Embedder()
    app.state.indexer = QdrantIndexer()
    yield


app = FastAPI(title="Worker Service", version="1.0.0", lifespan=lifespan)


def _is_test_env() -> bool:
    """pytest 실행 중이면 요청마다 의존성을 새로 구성한다."""
    return "PYTEST_CURRENT_TEST" in os.environ


def _get_loader(app: FastAPI) -> MinIOLoader:
    if _is_test_env() or not hasattr(app.state, "loader"):
        app.state.loader = MinIOLoader()
    return app.state.loader


def _get_extractor(app: FastAPI) -> TextExtractor:
    if _is_test_env() or not hasattr(app.state, "extractor"):
        app.state.extractor = TextExtractor()
    return app.state.extractor


def _get_chunker(app: FastAPI) -> TextChunker:
    if _is_test_env() or not hasattr(app.state, "chunker"):
        app.state.chunker = TextChunker(chunk_size=512, overlap=50)
    return app.state.chunker


def _get_embedder(app: FastAPI) -> Embedder:
    if _is_test_env() or not hasattr(app.state, "embedder"):
        app.state.embedder = Embedder()
    return app.state.embedder


def _get_indexer(app: FastAPI) -> QdrantIndexer:
    if _is_test_env() or not hasattr(app.state, "indexer"):
        app.state.indexer = QdrantIndexer()
    return app.state.indexer


class EmbedRequest(BaseModel):
    """임베딩 요청 모델"""
    texts: list[str]


class EmbedResponse(BaseModel):
    """임베딩 응답 모델"""
    embeddings: list[list[float]]


@app.get("/health")
def health(request: Request):
    """내부 컴포넌트 준비 상태를 함께 반환하는 헬스체크 엔드포인트"""
    components = {
        "loader": "ok" if _get_loader(request.app) else "missing",
        "extractor": "ok" if _get_extractor(request.app) else "missing",
        "chunker": "ok" if _get_chunker(request.app) else "missing",
        "embedder": "ok" if _get_embedder(request.app) else "missing",
        "indexer": "ok" if _get_indexer(request.app) else "missing",
    }
    status = "ok" if all(value == "ok" for value in components.values()) else "degraded"
    return {
        "status": status,
        "service": "worker",
        "components": components,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest, request: Request):
    """
    텍스트 임베딩 엔드포인트

    Args:
        payload: 임베딩할 텍스트 목록

    Returns:
        EmbedResponse: 임베딩 벡터 목록
    """
    if not payload.texts:
        raise HTTPException(status_code=400, detail="텍스트 목록이 비어있습니다.")

    embeddings = _get_embedder(request.app).embed(payload.texts)
    return EmbedResponse(embeddings=embeddings)


@app.post("/ingest")
def ingest(request: Request):
    """
    MinIO에서 모든 파일을 수집하여 Qdrant에 인덱싱하는 엔드포인트

    워크플로우:
    1. MinIO에서 파일 목록 조회
    2. 각 파일 다운로드 → 텍스트 추출 → 청킹
    3. 청크 임베딩 → Qdrant에 업서트
    """
    loader = _get_loader(request.app)
    extractor = _get_extractor(request.app)
    chunker = _get_chunker(request.app)
    embedder = _get_embedder(request.app)
    indexer = _get_indexer(request.app)

    object_names = loader.list_objects()
    processed = 0
    errors = []

    for object_name in object_names:
        try:
            # 파일 확장자 추출
            ext = object_name.rsplit(".", 1)[-1].lower() if "." in object_name else "txt"

            # 파일 다운로드 및 텍스트 추출
            content = loader.download(object_name)
            try:
                text = extractor.extract(content, ext)
            except ValueError:
                # 지원하지 않는 파일 형식은 건너뜀
                continue

            # 텍스트 청킹
            chunks = chunker.chunk(text)
            if not chunks:
                continue

            # 임베딩 및 인덱싱
            vectors = embedder.embed(chunks)
            payloads = [
                {
                    "source": object_name,
                    "chunk_index": i,
                    "text": chunk,
                    "file_type": ext,
                }
                for i, chunk in enumerate(chunks)
            ]
            indexer.upsert(vectors, payloads)
            processed += 1

        except Exception as e:
            errors.append({"file": object_name, "error": str(e)})

    return {
        "status": "ok",
        "processed": processed,
        "errors": errors,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
