import numpy as np


def cosine_similarity(store_embeddings, query_embedding, top_k):
    dot_product = np.dot(store_embeddings, query_embedding)
    magnitude_a = np.linalg.norm(store_embeddings, axis=1)
    magnitude_b = np.linalg.norm(query_embedding)

    similarity = dot_product / magnitude_a * magnitude_b

    sim = np.argsort(similarity)
    top_k_indices = sim[::-1][:top_k]

    return top_k_indices
