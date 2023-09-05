-- 사용자 생성 및 패스워드 설정
CREATE USER postgres WITH PASSWORD 'postgres';
-- 원하는 데이터베이스와 권한 설정 추가
CREATE DATABASE postgres;
GRANT ALL PRIVILEGES ON DATABASE postgres TO postgres;