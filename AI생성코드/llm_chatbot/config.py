import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수 불러오기
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
