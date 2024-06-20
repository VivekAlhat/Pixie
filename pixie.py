import numpy as np
from sentence_transformers import SentenceTransformer

from helpers import cosine_similarity


class Pixie:
    def __init__(self, embedder) -> None:
        self.store: np.ndarray = None
        self.embedder: SentenceTransformer = embedder

    def from_docs(self, docs):
        self.docs = np.array(docs)
        self.store = self.embedder.encode(self.docs)
        return f"Ingested {len(docs)} documents"

    def similarity_search(self, query, top_k=3):
        matches = list()
        q_embedding = self.embedder.encode(query)
        top_k_indices = cosine_similarity(self.store, q_embedding, top_k)
        for i in top_k_indices:
            matches.append(self.docs[i])
        return matches

    def __cosine__similarity():
        print()
