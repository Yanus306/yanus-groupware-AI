"""
텍스트 추출 모듈

PDF, Markdown, TXT 파일에서 텍스트를 추출합니다.
- PDF: pypdf 라이브러리 사용
- MD: 일반 텍스트로 처리
- TXT: 일반 텍스트로 처리
"""

import io
from pypdf import PdfReader


class TextExtractor:
    """파일 형식별 텍스트 추출 클래스"""

    SUPPORTED_TYPES = {"pdf", "md", "txt"}

    def extract(self, content: bytes, file_type: str) -> str:
        """
        파일 내용에서 텍스트를 추출합니다.

        Args:
            content: 파일 바이너리 데이터
            file_type: 파일 확장자 (pdf, md, txt)

        Returns:
            str: 추출된 텍스트

        Raises:
            ValueError: 지원하지 않는 파일 형식인 경우
        """
        file_type = file_type.lower().lstrip(".")

        if file_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"지원하지 않는 파일 형식: {file_type}")

        if file_type == "pdf":
            return self._extract_pdf(content)
        elif file_type in ("md", "txt"):
            return self._extract_text(content)

    def _extract_pdf(self, content: bytes) -> str:
        """PDF 파일에서 텍스트를 추출합니다."""
        reader = PdfReader(io.BytesIO(content))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texts.append(page_text)
        return "\n".join(texts)

    def _extract_text(self, content: bytes) -> str:
        """텍스트/마크다운 파일에서 텍스트를 추출합니다."""
        return content.decode("utf-8", errors="replace")
