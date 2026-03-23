"""
MinIO 파일 로더 모듈

MinIO 오브젝트 스토리지에서 파일을 다운로드하는 기능을 제공합니다.
환경 변수에서 MinIO 연결 정보를 읽어옵니다.
"""

import os
from minio import Minio
from dotenv import load_dotenv

load_dotenv()


class MinIOLoader:
    """MinIO 파일 로더 클래스"""

    def __init__(self):
        """MinIO 클라이언트를 초기화합니다."""
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.getenv("MINIO_BUCKET", "yanus-files")

        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,
        )

    def list_objects(self) -> list[str]:
        """
        MinIO 버킷의 모든 객체 이름을 반환합니다.

        Returns:
            list[str]: 객체 이름 목록
        """
        objects = self.client.list_objects(self.bucket)
        return [obj.object_name for obj in objects]

    def download(self, object_name: str) -> bytes:
        """
        MinIO에서 특정 객체를 다운로드합니다.

        Args:
            object_name: 다운로드할 객체 이름

        Returns:
            bytes: 파일 바이너리 데이터
        """
        response = self.client.get_object(self.bucket, object_name)
        return response.read()
