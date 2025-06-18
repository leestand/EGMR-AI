import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_top_stores_by_info_similarity(query_embedding, store_info_embeddings, store_ids_with_names):
    """
    쿼리 임베딩과 음식점 정보 임베딩 간의 유사도를 계산하고 상위 20개의 유사도 높은 음식점을 반환합니다.
    """
    # 코사인 유사도를 계산하여 상위 20개 음식점 선택
    info_similarities = cosine_similarity([query_embedding], store_info_embeddings)[0]
    top_indices = np.argsort(info_similarities)[-20:][::-1]  # 상위 20개를 내림차순으로 정렬
    top_20_stores = [(store_ids_with_names[idx], info_similarities[idx]) for idx in top_indices]
    
    return top_20_stores

def find_review_similarities_for_top_20(query_embedding, review_embeddings, store_ids_with_names, top_20_store_ids):
    """
    상위 20개의 음식점에 해당하는 리뷰 임베딩과 쿼리 임베딩 간의 유사도를 계산하여 반환합니다.
    """
    # 상위 20개 음식점의 리뷰 임베딩을 선택
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
