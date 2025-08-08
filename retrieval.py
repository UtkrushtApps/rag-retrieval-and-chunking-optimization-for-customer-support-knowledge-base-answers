# retrieval.py
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SupportChunkRetriever:
    def __init__(self, collection_name='support_chunks', embedder_model='all-mpnet-base-v2'):
        self.client = chromadb.Client()
        self.collection = self.client.get_collection(collection_name)
        self.model = SentenceTransformer(embedder_model)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=['documents', 'embeddings', 'metadatas', 'ids']
        )
        # Format the output: list of {chunk, score, metadata}
        # Chroma will return 'distances' (smaller is closer for L2), but for cosine, larger is closer
        formatted = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'chunk_id': results['ids'][0][i],
                'chunk_text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': results['distances'][0][i] if 'distances' in results else None
            }
            formatted.append(chunk)
        return formatted

# Sample usage & spot-check
if __name__ == '__main__':
    retriever = SupportChunkRetriever()
    sample_queries = [
        "How do I reset my password?",
        "Urgent payment failed issue yesterday",
        "Change subscription tier"
    ]
    for q in sample_queries:
        print(f"Query: {q}")
        top_chunks = retriever.retrieve(q, top_k=5)
        for i, c in enumerate(top_chunks):
            print(f"Rank {i+1}: ({c['metadata']['category']}, priority: {c['metadata']['priority']}, date: {c['metadata']['date']})\n---\n{c['chunk_text']}\n---\n")
        print("="*40)
