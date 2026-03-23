"""
텍스트 청킹 모듈

긴 텍스트를 지정된 크기의 청크로 분할합니다.
오버랩을 지원하여 청크 경계에서 문맥이 끊기는 것을 방지합니다.
"""


class TextChunker:
    """텍스트 청킹 클래스"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        청커를 초기화합니다.

        Args:
            chunk_size: 각 청크의 최대 문자 수 (기본값: 512)
            overlap: 인접 청크 간 중복 문자 수 (기본값: 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """
        텍스트를 청크로 분할합니다.

        Args:
            text: 분할할 텍스트

        Returns:
            list[str]: 텍스트 청크 목록
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        stride = self.chunk_size - self.overlap
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= len(text):
                break
            start += stride

        return chunks
