"""
Worker 서비스 메인 모듈

DOC-001: 문서 수집 및 임베딩 파이프라인
- POST /ingest: MinIO에서 파일을 가져와 Qdrant에 인덱싱
- POST /embed: 텍스트 임베딩 요청
- GET /health: 헬스체크
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from .loader import MinIOLoader
from .extractor import TextExtractor
from .chunker import TextChunker
from .embedder import Embedder
from .indexer import QdrantIndexer

load_dotenv()

app = FastAPI(title="Worker Service", version="1.0.0")

# 컴포넌트 초기화 (모듈 로드 시 한 번만)
loader = MinIOLoader()
extractor = TextExtractor()
chunker = TextChunker(chunk_size=512, overlap=50)
embedder = Embedder()
indexer = QdrantIndexer()


class EmbedRequest(BaseModel):
    """임베딩 요청 모델"""
    texts: list[str]


class EmbedResponse(BaseModel):
    """임베딩 응답 모델"""
    embeddings: list[list[float]]


@app.get("/health")
def health():
    """헬스체크 엔드포인트"""
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    """
    텍스트 임베딩 엔드포인트

    Args:
        request: 임베딩할 텍스트 목록

    Returns:
        EmbedResponse: 임베딩 벡터 목록
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="텍스트 목록이 비어있습니다.")

    embeddings = embedder.embed(request.texts)
    return EmbedResponse(embeddings=embeddings)


@app.post("/ingest")
def ingest():
    """
    MinIO에서 모든 파일을 수집하여 Qdrant에 인덱싱하는 엔드포인트

    워크플로우:
    1. MinIO에서 파일 목록 조회
    2. 각 파일 다운로드 → 텍스트 추출 → 청킹
    3. 청크 임베딩 → Qdrant에 업서트
    """
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
