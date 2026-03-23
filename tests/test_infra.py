"""
BE-001: MinIO + Qdrant 인프라 헬스체크 테스트
- MinIO: 버킷 생성 및 파일 업로드 성공 여부
- Qdrant: 컬렉션 생성 성공 여부
"""

import io
import pytest
from minio import Minio
from minio.error import S3Error
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_BUCKET = "test-bucket"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "test-collection"
VECTOR_SIZE = 4


# ── MinIO ──────────────────────────────────────────────

@pytest.fixture
def minio_client():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    yield client
    # teardown: 버킷 정리
    if client.bucket_exists(MINIO_BUCKET):
        for obj in client.list_objects(MINIO_BUCKET):
            client.remove_object(MINIO_BUCKET, obj.object_name)
        client.remove_bucket(MINIO_BUCKET)


@pytest.fixture
def qdrant_client():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    yield client
    # teardown: 컬렉션 정리
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)


# ── 테스트 ────────────────────────────────────────────

class TestMinIO:
    def test_minio_bucket_create(self, minio_client):
        """MinIO 버킷 생성 성공"""
        minio_client.make_bucket(MINIO_BUCKET)
        assert minio_client.bucket_exists(MINIO_BUCKET)

    def test_minio_file_upload(self, minio_client):
        """MinIO 파일 업로드 성공"""
        minio_client.make_bucket(MINIO_BUCKET)

        data = b"hello rag"
        minio_client.put_object(
            MINIO_BUCKET,
            "test.txt",
            io.BytesIO(data),
            length=len(data),
            content_type="text/plain",
        )

        objects = list(minio_client.list_objects(MINIO_BUCKET))
        assert len(objects) == 1
        assert objects[0].object_name == "test.txt"


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
        """Qdrant 컬렉션 정보 정상 조회"""
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        assert info.config.params.vectors.size == VECTOR_SIZE
        assert info.config.params.vectors.distance == Distance.COSINE
