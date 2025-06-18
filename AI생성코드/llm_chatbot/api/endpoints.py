from models.llm import generate_expanded_queries
from models.embedding import create_embeddings
from models.faiss_index import load_faiss_indices
from utils.summarization import summarize_reviews
from utils.json_utils import load_json, load_user_preferences
from utils.similarity import find_top_stores_by_info_similarity, find_review_similarities_for_top_20

def handle_chat_request(query):
    # 데이터 및 사용자 선호도 로드
    data = load_json('data/final_dataset3_document.json')
    user_preferences = load_user_preferences('data/user_preference.json')
    
    # 확장된 쿼리 생성
    store_info_query, review_query = generate_expanded_queries(query, user_preferences, data)
    
    # 임베딩 생성
    store_info_embedding, review_embedding = create_embeddings(store_info_query, review_query)
    
    # 상위 20개 음식점 및 유사도 계산
    store_info_embeddings, review_embeddings, store_ids_with_names = load_faiss_indices()
    top_20_stores = find_top_stores_by_info_similarity(store_info_embedding, store_info_embeddings, store_ids_with_names)
    top_20_store_ids = {store_id for (store_id, _), _ in top_20_stores}
    filtered_store_ids_with_names, review_similarities = find_review_similarities_for_top_20(review_embedding, review_embeddings, store_ids_with_names, top_20_store_ids)

    # 요약 생성 및 결과 반환
    result = summarize_reviews(filtered_store_ids_with_names, review_similarities, top_20_stores, data)
    return result

