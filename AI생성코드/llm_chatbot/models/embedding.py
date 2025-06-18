from sentence_transformers import SentenceTransformer

# KoSentenceBERT 모델 초기화
model_name = "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
embedding_model = SentenceTransformer(model_name)

def create_embeddings(store_info_query, review_query):
    store_info_embedding = embedding_model.encode([store_info_query])[0]
    review_embedding = embedding_model.encode([review_query])[0]
    return store_info_embedding, review_embedding
