# rag_chunking.py
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import chromadb
from typing import List, Dict
import tqdm
import datetime

nltk.download('punkt')

# Assumptions:
# - support_documents: list of dict, each with keys: 'id', 'content', 'category', 'priority', 'date'
# - ChromaDB is already initialized; collection is already created

# 0. Example loader for support_documents (replace with actual document source)
# support_documents = load_support_documents()  # Each doc: {id, content, category, priority, date}

CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 200  # tokens


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks of chunk_size tokens with chunk_overlap."""
    words = word_tokenize(text)
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunk_text = ' '.join(chunk)
        chunks.append(chunk_text)
        if i + chunk_size >= len(words):
            break
        i += chunk_size - chunk_overlap
    return chunks

def process_and_chunk_documents(support_documents: List[Dict]) -> List[Dict]:
    """Chunk each support document and format chunks with accurate metadata."""
    processed_chunks = []
    for doc in tqdm.tqdm(support_documents, desc='Chunking documents'):
        doc_id = doc['id']
        content = doc['content']
        category = doc['category']
        priority = doc['priority']
        date = doc['date']
        
        # Split into chunks
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{idx}"
            chunk_metadata = {
                'doc_id': doc_id,
                'chunk_index': idx,
                'category': category,
                'priority': priority,
                'date': date
            }
            processed_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'metadata': chunk_metadata
            })
    return processed_chunks

def embed_chunks(chunks: List[Dict], model_name='all-mpnet-base-v2') -> List[Dict]:
    """Run embedding model on each chunk."""
    model = SentenceTransformer(model_name)
    chunk_texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    for i, emb in enumerate(embeddings):
        chunks[i]['embedding'] = emb
    return chunks

def batch_upsert_chunks(chunks: List[Dict], collection_name='support_chunks', batch_size=512):
    # Chroma client (assume configuration elsewhere)
    client = chromadb.Client()
    collection = client.get_collection(collection_name)
    total = len(chunks)
    for i in tqdm.tqdm(range(0, total, batch_size), desc='Upserting into Chroma'):
        batch = chunks[i:i+batch_size]
        ids = [c['id'] for c in batch]
        docs = [c['text'] for c in batch]
        embeddings = [c['embedding'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metadatas
        )

def main():
    from support_docs_loader import load_support_documents  # You must implement this
    support_documents = load_support_documents()
    chunks = process_and_chunk_documents(support_documents)
    chunks = embed_chunks(chunks)
    batch_upsert_chunks(chunks)

if __name__ == '__main__':
    main()
