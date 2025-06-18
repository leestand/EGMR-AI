import re
from konlpy.tag import Mecab

mecab = Mecab(r'/home/letskt90/dev/mecab-0.996-ko-0.9.2/mecab-ko-dic-2.1.1-20180720')

def preprocess_korean_text(text):
    text = re.sub(r"[^가-힣\s]", "", text)
    tokens = mecab.morphs(text)
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)
