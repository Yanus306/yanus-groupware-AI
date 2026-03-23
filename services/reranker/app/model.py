"""
리랭커 모델 모듈

dragonkue/bge-reranker-v2-m3-ko 모델을 사용하여 문서를 재정렬합니다.
GPU 1 (CUDA_VISIBLE_DEVICES=1)을 사용합니다.
"""

import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# GPU 할당: GPU 1 사용
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"


class Reranker:
    """BGE 리랭커 클래스"""

    def __init__(self):
        """리랭커 모델을 로드합니다."""
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        self.model = CrossEncoder(MODEL_NAME, device=device)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        문서를 쿼리와의 관련성 점수로 재정렬합니다.

        Args:
            query: 검색 쿼리
            documents: 재정렬할 문서 목록 (각 문서는 text, source 필드 포함)
            top_n: 반환할 최대 문서 수

        Returns:
            list[dict]: 점수 내림차순으로 정렬된 문서 목록 (score 필드 추가)
        """
        if not documents:
            return []

        # CrossEncoder 입력 형식: (query, doc) 쌍
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs)

        # 점수를 문서에 추가
        scored_docs = [
            {**doc, "score": float(score)}
            for doc, score in zip(documents, scores)
        ]

        # 점수 내림차순 정렬 후 top_n 반환
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_n]
