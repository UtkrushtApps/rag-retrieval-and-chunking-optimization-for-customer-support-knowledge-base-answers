# eval_recall5.py
from retrieval import SupportChunkRetriever
from support_docs_loader import load_support_documents
import tqdm

# For recall@5 evaluation,
# We'll pick queries for which we know the ground truth (e.g., from original docs),
# and check whether a chunk from the right document is in the top-5 retrieved.

def recall_at_5(documents, sample_queries):
    retriever = SupportChunkRetriever()
    hits = 0
    for item in tqdm.tqdm(sample_queries):
        query = item['query']
        gt_doc_id = item['doc_id']
        predictions = retriever.retrieve(query, top_k=5)
        pred_doc_ids = set(chunk['metadata']['doc_id'] for chunk in predictions)
        if gt_doc_id in pred_doc_ids:
            hits += 1
    return hits / len(sample_queries)

def main():
    # Example: Build a test set
    documents = load_support_documents()
    # Let's use the first 100 docs for spot-check evaluation
    sample_queries = []
    for doc in documents[:100]:
        query = doc['content'][:250]  # Simulate realistic customer query from start of doc
        sample_queries.append({
            'query': query,
            'doc_id': doc['id']
        })
    recall5 = recall_at_5(documents, sample_queries)
    print(f'Recall@5: {recall5:.2%}')

if __name__ == '__main__':
    main()
