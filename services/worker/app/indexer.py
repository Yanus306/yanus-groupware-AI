"""
Qdrant 인덱서 모듈

텍스트 임베딩을 Qdrant 벡터 데이터베이스에 업서트합니다.
컬렉션명: documents
벡터 차원: 1024 (BGE-m3-ko 출력)
"""

import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "documents"
VECTOR_SIZE = 1024


class QdrantIndexer:
    """Qdrant 벡터 인덱서 클래스"""

    def __init__(self):
        """Qdrant 클라이언트를 초기화합니다."""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self.client = QdrantClient(host=host, port=port)

    def _ensure_collection(self):
        """컬렉션이 없으면 생성합니다."""
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, vectors: list[list[float]], payloads: list[dict]) -> None:
        """
        벡터와 메타데이터를 Qdrant에 업서트합니다.

        Args:
            vectors: 임베딩 벡터 목록
            payloads: 메타데이터 목록 (source, chunk_index, text, file_type)
        """
        self._ensure_collection()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            for vector, payload in zip(vectors, payloads)
        ]

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
