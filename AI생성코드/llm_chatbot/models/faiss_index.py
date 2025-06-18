import faiss
import numpy as np

def load_faiss_indices():
    store_info_faiss_index = faiss.read_index('store_info_faiss.index')
    review_faiss_index = faiss.read_index('review_faiss.index')
    store_ids_with_names = np.load('store_ids_with_names.npy', allow_pickle=True)
    
    store_info_embeddings = [store_info_faiss_index.reconstruct(i) for i in range(store_info_faiss_index.ntotal)]
    review_embeddings = [review_faiss_index.reconstruct(i) for i in range(review_faiss_index.ntotal)]
    
    return np.array(store_info_embeddings), np.array(review_embeddings), store_ids_with_names
