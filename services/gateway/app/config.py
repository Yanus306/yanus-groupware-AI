"""
Gateway 서비스 설정 모듈

환경 변수에서 서비스 연결 정보를 읽어옵니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """서비스 설정 클래스"""

    # Worker 서비스 URL
    WORKER_URL: str = os.getenv("WORKER_URL", "http://localhost:8001")

    # Reranker 서비스 URL
    RERANKER_URL: str = os.getenv("RERANKER_URL", "http://localhost:8002")

    # vLLM 서비스 URL
    VLLM_URL: str = os.getenv("VLLM_URL", "http://localhost:8080")

    # Qdrant 연결 정보
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

    # Qdrant 컬렉션
    QDRANT_COLLECTION: str = "documents"

    # vLLM 모델명
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"


settings = Settings()
