# 기반 이미지 선택
FROM python:3.10.12

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    bzip2 \
    ca-certificates \
    libffi-dev \
    tk \ 
    libsqlite3-dev \
    liblzma-dev \
    zlib1g-dev \
    libgl1-mesa-glx

# 작업 디렉토리 생성
WORKDIR /app

# 호스트의 현재 디렉토리의 모든 파일 복사
COPY . /app

# 필요한 종속성 설치
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN sudo apt-get update
RUN sudo apt-get install postgresql
RUN sudo apt install redis-server

RUN pip install -r requirements.txt

FROM postgres:latest
FROM redis:latest


# PostgreSQL 서버 초기화 SQL 파일 복사
COPY init.sql /docker-entrypoint-initdb.d/

# 환경 변수 설정 (사용자 이름 및 패스워드)

ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postres

# PostgreSQL 및 Redis 서버를 백그라운드 모드로 실행
CMD ["bash", "-c", "service postgresql start && service redis-server start && gunicorn app:app"]