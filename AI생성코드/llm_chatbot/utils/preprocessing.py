# preprocessing.py
import re
from konlpy.tag import Mecab

# 형태소 분석기(Mecab) 초기화
mecab = Mecab()

# 한국어 전처리 함수 정의
def preprocess_korean_text(text):
    text = re.sub(r"[^가-힣\s]", "", text)  # 특수 문자 및 숫자 제거
    tokens = mecab.morphs(text)  # 형태소 분석을 통한 토큰화
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    tokens = [token for token in tokens if token not in stopwords]  # 불용어 제거
    return " ".join(tokens)
