FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 의존성 설치
COPY requirements.txt .
RUN grep -v '^torch==' requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt \
    && rm requirements.docker.txt

# 소스 코드 복사
COPY app/ ./app/

# 포트 노출
EXPOSE 8001

# 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
