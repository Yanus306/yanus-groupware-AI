"""
BE-001: MinIO + Qdrant 인프라 헬스체크 테스트
- MinIO: 버킷 존재 확인 및 파일 업로드 성공 여부
- Qdrant: 컬렉션 생성 성공 여부
"""

import io
import os
import pytest
from minio import Minio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from dotenv import load_dotenv
load_dotenv()

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "192.168.0.30:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "yanus")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "yanus0409")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "yanus-files")

QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "test-collection"
VECTOR_SIZE       = 4


# ── MinIO ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minio_client():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    yield client
    # teardown: 테스트 전용 오브젝트만 정리
    try:
        client.remove_object(MINIO_BUCKET, "test-infra.txt")
    except Exception:
        pass


@pytest.fixture
def qdrant_client():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    yield client
    # teardown: 테스트 컬렉션 정리
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)


# ── 테스트 ──────────────────────────────────────────────────────────────────

class TestMinIO:
    def test_minio_bucket_exists(self, minio_client):
        """MinIO 버킷이 존재하는지 확인"""
        assert minio_client.bucket_exists(MINIO_BUCKET), \
            f"버킷 '{MINIO_BUCKET}' 이 MinIO 서버에 없습니다"

    def test_minio_file_upload(self, minio_client):
        """MinIO 파일 업로드 성공"""
        data = b"hello rag"
        minio_client.put_object(
            MINIO_BUCKET,
            "test-infra.txt",
            io.BytesIO(data),
            length=len(data),
            content_type="text/plain",
        )
        objects = [o.object_name for o in minio_client.list_objects(MINIO_BUCKET)]
        assert "test-infra.txt" in objects


class TestQdrant:
    def test_qdrant_collection_create(self, qdrant_client):
        """Qdrant 컬렉션 생성 성공"""
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        existing = [c.name for c in qdrant_client.get_collections().collections]
        assert QDRANT_COLLECTION in existing

    def test_qdrant_collection_info(self, qdrant_client):
        """Qdrant 컬렉션 벡터 설정 정상 확인"""
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        assert info.config.params.vectors.size == VECTOR_SIZE
        assert info.config.params.vectors.distance == Distance.COSINE
