import pandas as pd
from models.llm import llm, ChatPromptTemplate

def summarize_reviews(stores, review_similarities, top_20_stores, data):
    results = []
    
    # 각 음식점의 요약 정보 생성
    for i, (store_id, store_name) in enumerate(stores):
        # 상위 20개 음식점 중 현재 음식점의 정보 유사도 가져오기
        store_info_similarity = next(sim for (sim_id, _), sim in top_20_stores if sim_id == store_id)
        review_similarity = review_similarities[i]
        
        # 카테고리 가져오기
        store_category = next(store['store_info']['category2'] for store in data if store['store_id'] == store_id)
        
        # 리뷰 텍스트 불러오기
        reviews = next(store['reviews']['text'] for store in data if store['store_id'] == store_id)
        
        # LLM을 통한 리뷰 요약 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
                다음 리뷰들을 요약해줘. 요약은 '장점', '단점' 으로 나누어 작성해주고, '장점', '단점'에 대한 묘사만 간결하게 해주고 
                '장점: (내용들) /n 단점: (내용들)' 형식으로 출력해줘
            """),
            ("human", "{input}")
        ])
        
        enriched_summary_message = prompt | llm
        summary = enriched_summary_message.invoke({"input": reviews}).content
        
        # 요약에서 장점과 단점을 구분하여 추출
        pros_keyword = "장점:"
        cons_keyword = "단점:"
        
        pros_index = summary.find(pros_keyword) + len(pros_keyword)
        cons_index = summary.find(cons_keyword)
        pros = summary[pros_index:cons_index].strip()
        cons = summary[cons_index + len(cons_keyword):].strip()
        
        # 결과 딕셔너리에 저장
        results.append({
            "store_id": store_id,
            "store_name": store_name,
            "category": store_category,
            "store_info_similarity": store_info_similarity,
            "review_similarity": review_similarity,
            "pros": pros,
            "cons": cons
        })
    
    # 결과 데이터프레임 생성 및 total_score 계산
    df = pd.DataFrame(results)
    df['total_score'] = df['store_info_similarity'] * 0.8 + df['review_similarity'] * 0.2
    df = df.sort_values(by='total_score', ascending=False)
    
    # JSON으로 변환하여 반환
    result_json = df[['store_id', 'store_name', 'category', 'pros', 'cons', 'total_score']].to_json(force_ascii=False, orient='records', indent=4)
    return result_json
