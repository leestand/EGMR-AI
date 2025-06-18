import json
import os
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# 데이터 로드
# JSON 파일 읽어오기
with open('/content/drive/MyDrive/NaverCrawlingData/final_final_final_dataset.json', 'r') as f:
    data = json.load(f)

# 빈 리스트 생성 (추출된 데이터를 저장하기 위한)
df_data = []

# 각 store에 대해 반복
for store in data:
    store_id = store['store_id']
    store_name = store['store_info']['store_name']

    # 각 review에 대해 반복
    for review in store['reviews']:
        text = review['text']
        df_data.append([store_id, store_name, text])

# DataFrame 생성
df = pd.DataFrame(df_data, columns=['store_id', 'store_name', 'text'])

print(df[df['text'].str.isspace()])
print(df.isnull().sum())

# 결측치 제거
df = df[df['text'] != '']
df.reset_index(drop=True, inplace=True)


# store_id의 고유한 개수 세기
unique_store_count = df['store_id'].nunique()

# 결과 출력
print(f"unique한 store_id의 개수: {unique_store_count}")


!pip install -Uqq torch transformers datasets pandas
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("WhitePeak/bert-base-cased-Korean-sentiment")

