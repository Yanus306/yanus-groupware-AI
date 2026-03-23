"""
Worker 서비스 테스트 모듈

DOC-001: 문서 수집 및 임베딩 파이프라인 테스트
- loader: MinIO 파일 로딩 테스트
- extractor: 텍스트 추출 테스트 (PDF, MD, TXT)
- chunker: 텍스트 청킹 테스트
- embedder: BGE-m3-ko 임베딩 테스트 (GPU 모킹)
- indexer: Qdrant 업서트 테스트 (모킹)
- main: FastAPI 엔드포인트 테스트
"""

import pytest
import io
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


# ───────────────────────────────────────────────────
# loader 테스트
# ───────────────────────────────────────────────────

class TestMinIOLoader:
    """MinIO 파일 로더 테스트"""

    @patch("services.worker.app.loader.Minio")
    def test_list_objects_returns_filenames(self, mock_minio_cls):
        """MinIO 버킷에서 객체 목록을 반환하는지 테스트"""
        from services.worker.app.loader import MinIOLoader

        # MinIO 클라이언트 모킹
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        mock_obj1 = MagicMock()
        mock_obj1.object_name = "doc1.pdf"
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "doc2.txt"
        mock_client.list_objects.return_value = [mock_obj1, mock_obj2]

        loader = MinIOLoader()
        filenames = loader.list_objects()

        assert "doc1.pdf" in filenames
        assert "doc2.txt" in filenames
        mock_client.list_objects.assert_called_once()

    @patch("services.worker.app.loader.Minio")
    def test_download_file_returns_bytes(self, mock_minio_cls):
        """MinIO에서 파일을 다운로드하면 bytes를 반환하는지 테스트"""
        from services.worker.app.loader import MinIOLoader

        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client

        fake_content = b"Hello, World!"
        mock_response = MagicMock()
        mock_response.read.return_value = fake_content
        mock_client.get_object.return_value = mock_response

        loader = MinIOLoader()
        data = loader.download("test.txt")

        assert data == fake_content
        mock_client.get_object.assert_called_once()


# ───────────────────────────────────────────────────
# extractor 테스트
# ───────────────────────────────────────────────────

class TestTextExtractor:
    """텍스트 추출기 테스트"""

    def test_extract_txt_returns_text(self):
        """TXT 파일에서 텍스트를 추출하는지 테스트"""
        from services.worker.app.extractor import TextExtractor

        extractor = TextExtractor()
        content = b"Hello, World!"
        result = extractor.extract(content, "txt")

        assert result == "Hello, World!"

    def test_extract_md_returns_text(self):
        """MD 파일에서 텍스트를 추출하는지 테스트"""
        from services.worker.app.extractor import TextExtractor

        extractor = TextExtractor()
        content = b"# Title\n\nSome content here."
        result = extractor.extract(content, "md")

        assert "Title" in result
        assert "Some content here." in result

    @patch("services.worker.app.extractor.PdfReader")
    def test_extract_pdf_returns_text(self, mock_pdf_reader_cls):
        """PDF 파일에서 텍스트를 추출하는지 테스트 (pypdf 모킹)"""
        from services.worker.app.extractor import TextExtractor

        mock_reader = MagicMock()
        mock_pdf_reader_cls.return_value = mock_reader

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PDF page content"
        mock_reader.pages = [mock_page]

        extractor = TextExtractor()
        result = extractor.extract(b"%PDF fake content", "pdf")

        assert "PDF page content" in result

    def test_extract_unsupported_type_raises(self):
        """지원하지 않는 파일 타입 시 예외 발생 테스트"""
        from services.worker.app.extractor import TextExtractor

        extractor = TextExtractor()
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            extractor.extract(b"content", "docx")


# ───────────────────────────────────────────────────
# chunker 테스트
# ───────────────────────────────────────────────────

class TestTextChunker:
    """텍스트 청킹 테스트"""

    def test_chunk_short_text_returns_single_chunk(self):
        """짧은 텍스트는 단일 청크로 반환하는지 테스트"""
        from services.worker.app.chunker import TextChunker

        chunker = TextChunker(chunk_size=512, overlap=50)
        text = "짧은 텍스트입니다."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text_returns_multiple_chunks(self):
        """긴 텍스트를 여러 청크로 분할하는지 테스트"""
        from services.worker.app.chunker import TextChunker

        chunker = TextChunker(chunk_size=100, overlap=10)
        # 300자 이상의 긴 텍스트 생성
        text = "가나다라마바사아자차카타파하 " * 30
        chunks = chunker.chunk(text)

        assert len(chunks) > 1

    def test_chunk_overlap_is_applied(self):
        """청크 간 오버랩이 올바르게 적용되는지 테스트"""
        from services.worker.app.chunker import TextChunker

        chunk_size = 50
        overlap = 10
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        text = "A" * 200
        chunks = chunker.chunk(text)

        # 오버랩이 있으면 청크 크기가 조정됨
        assert len(chunks) > 1
        # 모든 청크가 비어있지 않아야 함
        for chunk in chunks:
            assert len(chunk) > 0

    def test_chunk_empty_text_returns_empty_list(self):
        """빈 텍스트는 빈 리스트를 반환하는지 테스트"""
        from services.worker.app.chunker import TextChunker

        chunker = TextChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_chunk_default_params(self):
        """기본 파라미터 (512/50) 확인 테스트"""
        from services.worker.app.chunker import TextChunker

        chunker = TextChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50


# ───────────────────────────────────────────────────
# embedder 테스트
# ───────────────────────────────────────────────────

class TestEmbedder:
    """임베더 테스트 (GPU 모킹)"""

    @patch("services.worker.app.embedder.SentenceTransformer")
    def test_embed_returns_list_of_vectors(self, mock_st_cls):
        """텍스트 임베딩 시 벡터 리스트를 반환하는지 테스트"""
        from services.worker.app.embedder import Embedder

        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model

        import numpy as np
        mock_model.encode.return_value = np.zeros((2, 1024))

        embedder = Embedder()
        texts = ["텍스트 1", "텍스트 2"]
        result = embedder.embed(texts)

        assert len(result) == 2
        assert len(result[0]) == 1024

    @patch("services.worker.app.embedder.SentenceTransformer")
    def test_embed_single_text(self, mock_st_cls):
        """단일 텍스트 임베딩 테스트"""
        from services.worker.app.embedder import Embedder

        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model

        import numpy as np
        mock_model.encode.return_value = np.zeros((1, 1024))

        embedder = Embedder()
        result = embedder.embed(["단일 텍스트"])

        assert len(result) == 1

    @patch("services.worker.app.embedder.SentenceTransformer")
    def test_embed_output_dimension_is_1024(self, mock_st_cls):
        """임베딩 출력 차원이 1024인지 테스트"""
        from services.worker.app.embedder import Embedder

        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model

        import numpy as np
        mock_model.encode.return_value = np.zeros((3, 1024))

        embedder = Embedder()
        result = embedder.embed(["a", "b", "c"])

        for vec in result:
            assert len(vec) == 1024


# ───────────────────────────────────────────────────
# indexer 테스트
# ───────────────────────────────────────────────────

class TestQdrantIndexer:
    """Qdrant 인덱서 테스트"""

    @patch("services.worker.app.indexer.QdrantClient")
    def test_upsert_creates_collection_if_not_exists(self, mock_qdrant_cls):
        """컬렉션이 없을 때 생성하는지 테스트"""
        from services.worker.app.indexer import QdrantIndexer

        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.collection_exists.return_value = False

        indexer = QdrantIndexer()
        vectors = [[0.1] * 1024]
        payloads = [{"source": "test.pdf", "chunk_index": 0, "text": "hello", "file_type": "pdf"}]
        indexer.upsert(vectors, payloads)

        mock_client.create_collection.assert_called_once()

    @patch("services.worker.app.indexer.QdrantClient")
    def test_upsert_skips_create_if_collection_exists(self, mock_qdrant_cls):
        """컬렉션이 이미 있으면 생성을 건너뛰는지 테스트"""
        from services.worker.app.indexer import QdrantIndexer

        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.collection_exists.return_value = True

        indexer = QdrantIndexer()
        vectors = [[0.1] * 1024]
        payloads = [{"source": "test.pdf", "chunk_index": 0, "text": "hello", "file_type": "pdf"}]
        indexer.upsert(vectors, payloads)

        mock_client.create_collection.assert_not_called()

    @patch("services.worker.app.indexer.QdrantClient")
    def test_upsert_calls_upsert_with_correct_payload(self, mock_qdrant_cls):
        """upsert 호출 시 올바른 페이로드를 전달하는지 테스트"""
        from services.worker.app.indexer import QdrantIndexer

        mock_client = MagicMock()
        mock_qdrant_cls.return_value = mock_client
        mock_client.collection_exists.return_value = True

        indexer = QdrantIndexer()
        vectors = [[0.1] * 1024]
        payloads = [{"source": "doc.txt", "chunk_index": 0, "text": "내용", "file_type": "txt"}]
        indexer.upsert(vectors, payloads)

        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "documents"


# ───────────────────────────────────────────────────
# FastAPI 엔드포인트 테스트
# ───────────────────────────────────────────────────

class TestWorkerAPI:
    """Worker FastAPI 엔드포인트 테스트"""

    @patch("services.worker.app.embedder.SentenceTransformer")
    @patch("services.worker.app.indexer.QdrantClient")
    @patch("services.worker.app.loader.Minio")
    def test_health_endpoint(self, mock_minio, mock_qdrant, mock_st):
        """GET /health 엔드포인트 테스트"""
        mock_st.return_value = MagicMock()
        mock_qdrant.return_value = MagicMock()
        mock_minio.return_value = MagicMock()

        from services.worker.app.main import app
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("services.worker.app.embedder.SentenceTransformer")
    @patch("services.worker.app.indexer.QdrantClient")
    @patch("services.worker.app.loader.Minio")
    def test_embed_endpoint(self, mock_minio, mock_qdrant, mock_st):
        """POST /embed 엔드포인트 테스트"""
        import numpy as np

        mock_model = MagicMock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.zeros((2, 1024))
        mock_qdrant.return_value = MagicMock()
        mock_minio.return_value = MagicMock()

        from services.worker.app.main import app
        client = TestClient(app)
        response = client.post("/embed", json={"texts": ["텍스트1", "텍스트2"]})

        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 2

    @patch("services.worker.app.embedder.SentenceTransformer")
    @patch("services.worker.app.indexer.QdrantClient")
    @patch("services.worker.app.loader.Minio")
    def test_ingest_endpoint(self, mock_minio_cls, mock_qdrant_cls, mock_st_cls):
        """POST /ingest 엔드포인트 테스트"""
        import numpy as np

        # MinIO 모킹
        mock_minio = MagicMock()
        mock_minio_cls.return_value = mock_minio

        mock_obj = MagicMock()
        mock_obj.object_name = "test.txt"
        mock_minio.list_objects.return_value = [mock_obj]

        mock_response = MagicMock()
        mock_response.read.return_value = b"Hello, this is a test document."
        mock_minio.get_object.return_value = mock_response

        # 임베더 모킹
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model
        mock_model.encode.return_value = np.zeros((1, 1024))

        # Qdrant 모킹
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant
        mock_qdrant.collection_exists.return_value = True

        from services.worker.app.main import app
        client = TestClient(app)
        response = client.post("/ingest")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
