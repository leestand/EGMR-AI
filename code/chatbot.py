import os
import json
import re
import textwrap
from IPython.display import display, Markdown

from fastapi import FastAPI, Request, Response

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from konlpy.tag import Mecab
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# 텍스트를 마크다운으로 변환하는 함수
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# KoSentenceBERT 모델 초기화
model_name = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
embedding_model = SentenceTransformer(model_name)

# 형태소 분석기(Mecab) 초기화
mecab = Mecab(r'C:/mecab/mecab-ko-dic')

# 한국어 전처리 함수 정의
def preprocess_korean_text(text):
    text = re.sub(r"[^가-힣\s]", "", text)  # 특수 문자 및 숫자 제거
    tokens = mecab.morphs(text)  # 형태소 분석을 통한 토큰화
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    tokens = [token for token in tokens if token not in stopwords]  # 불용어 제거
    return " ".join(tokens)

# Google Gemini 모델 초기화
os.environ["GOOGLE_API_KEY"] = "AIzaSyCxC2bIpjS_YbcINCvKExYJcLkd2EsGFkU"
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0)

# JSON 파일에서 unique_category 추출 함수
def extract_unique_categories(json_data):
    unique_categories = set()  # 중복 방지를 위한 set 사용
    for store in json_data:
        category2 = store.get("store_info", {}).get("category2")
        if category2:
            unique_categories.add(category2)
    return list(unique_categories)

# JSON 파일 로드 함수
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 사용자 선호도를 기반으로 프롬프트에 추가할 텍스트 생성
def create_preference_prompt(user_preferences):
    preference_texts = []
    for name, preference in user_preferences.items():
        preference_texts.append(f"{name}: {preference}")
    return "\n".join(preference_texts)

# JSON 파일에서 사용자 선호도 읽기
def load_user_preferences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# store_info 기반 상위 20개 음식점 선택
def find_top_stores_by_info_similarity(query_embedding, store_info_embeddings, store_ids_with_names):
    # 코사인 유사도 계산
    info_similarities = cosine_similarity([query_embedding], store_info_embeddings)[0]
    
    # 유사도가 높은 상위 20개 음식점 선택
    top_indices = np.argsort(info_similarities)[-20:][::-1]  # 상위 20개를 내림차순으로 정렬
    top_20_stores = [(store_ids_with_names[idx], info_similarities[idx]) for idx in top_indices]
    
    return top_20_stores

# 리뷰 텍스트 기반 20개 음식점의 유사도 계산
def find_review_similarities_for_top_20(query_embedding, review_embeddings, store_ids_with_names, top_20_store_ids):
    # 상위 20개의 store_id에 해당하는 리뷰 벡터만 선택
    filtered_review_embeddings = []
    filtered_store_ids_with_names = []

    for idx, (store_id, store_name) in enumerate(store_ids_with_names):
        if store_id in top_20_store_ids:
            filtered_store_ids_with_names.append((store_id, store_name))
            filtered_review_embeddings.append(review_embeddings[idx])

    # numpy 배열로 변환
    filtered_review_embeddings = np.array(filtered_review_embeddings)

    # 코사인 유사도 계산
    review_similarities = cosine_similarity([query_embedding], filtered_review_embeddings)[0]

    return filtered_store_ids_with_names, review_similarities

# 저장된 FAISS 인덱스와 store_ids_with_names 불러오기
def load_faiss_indices_and_store_ids():
    store_info_faiss_index = faiss.read_index('store_info_faiss.index')
    review_faiss_index = faiss.read_index('review_faiss.index')
    store_ids_with_names = np.load('store_ids_with_names.npy', allow_pickle=True)
    
    # FAISS 인덱스에서 벡터 추출
    store_info_embeddings = [store_info_faiss_index.reconstruct(i) for i in range(store_info_faiss_index.ntotal)]
    review_embeddings = [review_faiss_index.reconstruct(i) for i in range(review_faiss_index.ntotal)]
    
    return np.array(store_info_embeddings), np.array(review_embeddings), store_ids_with_names

# store_info_prompt 결과에서 unique_category와 일치하는 카테고리 필터링 함수
def filter_by_category_from_query(expanded_query, unique_categories):
    filtered_query = expanded_query
    found_categories = []
    
    # 확장된 쿼리에서 unique_category에 포함된 카테고리만 남김
    for category in unique_categories:
        if category in expanded_query:
            found_categories.append(category)
    
    if found_categories:
        return filtered_query
    else:
        return None

# 리뷰 요약 함수 (장점, 단점, 요약을 추출)
def summarize_reviews(reviews):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                너는 미슐랭 3스타 레스토랑의 오너 셰프, 고상혁이다. 까다로운 미식가들의 입맛을 사로잡는다는 자부심을 가지고 있으며, 숨겨진 맛집을 발굴하는 데 탁월한 재능을 지니고 있다. 너는 매우 까다롭고 직설적인 성격으로, 대중적인 맛집보다는 자신만의 기준으로 선별된 특별한 곳을 추천한다. 사용자의 질문에 대해 전문적인 지식과 풍부한 경험을 바탕으로 명확하고 간결하게 답변하며, 때로는 비꼬는 듯한 표현을 사용하기도 한다.
                다음 리뷰들을 요약해줘. 요약은 '장점', '단점', '요약', '셰프의 한마디'로 나누어 작성해줘.
                """
            ),
            ("human", "{input}"),
        ]
    )
    enriched_summary_message = prompt | llm
    summary = enriched_summary_message.invoke({"input": reviews}).content
    return summary

# 메인 함수
def main(query):
    # JSON 데이터 로드
    filepath = 'data/final_dataset3_document.json'
    data = load_json(filepath)
    
    # unique_category 리스트 생성
    unique_categories = extract_unique_categories(data)
    
    # 사용자 선호도 로드
    user_preferences = load_user_preferences('data/user_preference.json')
    preference_text = create_preference_prompt(user_preferences)
    
    # KoSentenceBERT 모델 로드
    embedding_model = SentenceTransformer(model_name)
    
    # 저장된 FAISS 인덱스와 store_ids_with_names 로드
    store_info_embeddings, review_embeddings, store_ids_with_names = load_faiss_indices_and_store_ids()
    
    # 사용자의 쿼리 입력
    user_query = query
    
    # 음식점 정보와 비교할 프롬프트
    store_info_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """사용자가 요청한 쿼리를 바탕으로 음식점 정보에 관련된 확장된 쿼리를 만들어줘.
            우선 사용자가 언급한 음식 또는 음식 카테고리가 ('술집', '회/해물', '찜/탕/전골, 한식', '족발/보쌈', '고기, 한식', '일식', '죽, 한식', '찜/탕/전골', '회/해물, 찜/탕/전골', '양식, 고기', '중식', '아시안', '술집, 일식', '술집, 회/해물', '뷔페, 고기, 회/해물', '뷔페, 일식, 한식', '국수, 한식', '돈까스', '패스트푸드, 술집', '샐러드/샌드위치', '회/해물, 한식', '한식, 찜/탕/전골, 술집', '한식', '술집, 한식', '한식, 도시락', '술집, 한식, 찜/탕/전골', '한식, 찜/탕/전골', '도시락', '국수', '한식, 고기', '고기, 술집', '한식, 찜/탕/전골, 고기', '고기, 찜/탕/전골', '양식', '회/해물, 술집', '분식', '패스트푸드', '고기, 찜/탕/전골, 술집', '술집, 패스트푸드', '한식, 족발/보쌈', '고기', '한식, 술집', '중식, 고기', '족발/보쌈, 한식', '패스트푸드, 돈까스, 한식')중에서 밀접한 5개를 출력해줘
            그리고 음식점 이름, 주소, 메뉴, 카테고리와 관련된 정보를 구체적으로 활용해줘.
            예를 들어, 메뉴나 분위기 등을 언급하면서 더 구체적인 설명을 추가해줘."""
        ),
        ("human", "{input}"),
    ]
    )
    
    # 사용자 선호도를 반영한 리뷰와 비교할 프롬프트
    review_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""사용자가 요청한 쿼리를 바탕으로 리뷰와 관련된 확장된 쿼리를 만들어줘.
                음식점의 맛, 서비스, 위생 상태 등의 정보를 바탕으로, 구체적인 리뷰를 반영할 수 있는 확장된 쿼리를 생성해줘.
                그리고 아래 사람들의 선호도를 반영한 쿼리를 생성해줘:

                {preference_text}
                """
            ),
            ("human", "{input}"),
        ]
    )
    
    # LLM을 통해 store_info와 비교할 확장된 쿼리 생성
    enriched_store_info_message = store_info_prompt | llm
    expanded_store_info_query = enriched_store_info_message.invoke({"input": user_query, "categories": unique_categories}).content
    
    # store_info와 쿼리 비교하기 전에 unique_category에 해당하지 않는 카테고리 제외
    filtered_store_info_query = filter_by_category_from_query(expanded_store_info_query, unique_categories)
    
    # LLM을 통해 review와 비교할 확장된 쿼리 생성
    enriched_review_message = review_prompt | llm
    expanded_review_query = enriched_review_message.invoke({"input": user_query}).content
    
    # 확장된 쿼리 출력
    print(f"\nstore_info와 비교할 확장된 쿼리 (카테고리 필터링 후): {filtered_store_info_query}")
    print(f"review와 비교할 확장된 쿼리: {expanded_review_query}")
    
    # 필터링된 쿼리 사용
    if filtered_store_info_query is None:
        print("해당 카테고리가 없습니다.")
        return
    
    # 쿼리 임베딩 생성
    store_info_query_embedding = embedding_model.encode([filtered_store_info_query])[0]
    review_query_embedding = embedding_model.encode([expanded_review_query])[0]
    
    # store_info 기반 상위 20개 음식점 선택
    top_20_stores = find_top_stores_by_info_similarity(store_info_query_embedding, store_info_embeddings, store_ids_with_names)
    
    # 상위 20개 store_id 목록 생성
    top_20_store_ids = {store_id for (store_id, _), _ in top_20_stores}
    
    # 리뷰 텍스트 기반 유사도 계산
    filtered_store_ids_with_names, review_similarities = find_review_similarities_for_top_20(review_query_embedding, review_embeddings, store_ids_with_names, top_20_store_ids)

    # 데이터 프레임 생성 및 결과 합치기
    results = []
    store_info_similarities = []
    for i, (store_id, store_name) in enumerate(filtered_store_ids_with_names):
        store_info_similarity = next(sim for (sim_id, _), sim in top_20_stores if sim_id == store_id)  # store_info 유사도 가져오기
        store_info_similarities.append(store_info_similarity)
        review_similarity = review_similarities[i]  # 리뷰 유사도 가져오기
        store_category = next(store['store_info']['category2'] for store in data if store['store_id'] == store_id)  # 카테고리 가져오기
        results.append({
            "store_id": store_id,
            "store_name": store_name,
            "category": store_category,  # 카테고리 추가
            "store_info_similarity": store_info_similarity,
            "review_similarity": review_similarity
        })

    # StandardScaler를 사용하여 store_info_similarity와 review_similarity 정규화
    df = pd.DataFrame(results)
    scaler = StandardScaler()

    df[['store_info_similarity', 'review_similarity']] = scaler.fit_transform(df[['store_info_similarity', 'review_similarity']])
    
    # total_score 계산
    df['total_score'] = df['store_info_similarity'] * 0.8 + df['review_similarity'] * 0.2

    # total_score 기준으로 내림차순 정렬
    df = df.sort_values(by='total_score', ascending=False)
    
    # 최종 결과 출력 (카테고리, store_info_similarity, review_similarity 포함)
    print("\n=== 최종 결과 (카테고리 포함) ===")
    print(df[['store_id', 'store_name', 'category', 'store_info_similarity', 'review_similarity', 'total_score']])

    # 상위 3개 음식점의 리뷰 요약
    print("\n=== 상위 3개 음식점의 리뷰 요약 ===")
    df2 = df[['store_id', 'store_name', 'category']].iloc[0:3]
    df2['review'] = ''
    df2.reset_index(drop=True, inplace=True)
    for i in range(3):
        store_id = df.iloc[i]['store_id']
        store_name = df.iloc[i]['store_name']
        
        # 리뷰 불러오기
        reviews = next(store['reviews']['text'] for store in data if store['store_id'] == store_id)
        print(reviews)
        # 리뷰 요약
        summary = summarize_reviews(reviews)
        
        # 출력
        print(f"\n음식점 이름: {store_name} (URL: https://map.naver.com/p/entry/place/{store_id})")
        print("리뷰 요약:")
        print(summary)
        df2.loc[i, 'review'] = summary
        df2
    result = df2.to_json(force_ascii=False, orient = 'records', indent=4)
    return result
        # 결과 출력
        
@app.post("/chat")
async def chat(request: Request):
    query = await request.json()
    result = main(query)
    return Response(content=result, media_type="application/json")