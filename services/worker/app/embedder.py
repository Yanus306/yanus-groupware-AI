"""
임베딩 모듈

dragonkue/BGE-m3-ko 모델을 사용하여 텍스트를 임베딩합니다.
GPU 1 (CUDA_VISIBLE_DEVICES=1)을 사용합니다.
출력 차원: 1024
"""

import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# GPU 할당: GPU 1 사용
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

MODEL_NAME = "dragonkue/BGE-m3-ko"


class Embedder:
    """BGE-m3-ko 텍스트 임베더 클래스"""

    def __init__(self):
        """모델을 로드합니다."""
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        텍스트 목록을 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트 목록

        Returns:
            list[list[float]]: 임베딩 벡터 목록 (각 벡터 차원: 1024)
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
