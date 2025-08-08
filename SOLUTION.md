# Solution Steps

1. Implement a function to load all support documents, each with accurate metadata (id, content, category, priority, date).

2. Implement a text chunker that splits each document's content into overlapping chunks (~500 tokens per chunk, 200 tokens overlap).

3. For each chunk, create a unique chunk ID and attach the corresponding source document's metadata, including category, priority, and date.

4. Use a sentence-transformers model (e.g., 'all-mpnet-base-v2') to compute embeddings for each chunk's text.

5. Batch upsert the chunks into the Chroma collection, including chunk text, embeddings, chunk ID, and metadata.

6. Implement a retrieval module: for a user query, embed the query and perform a cosine similarity search to retrieve the top-5 most relevant chunks from Chroma, returning each chunk's text and metadata.

7. Validate improvements by manually issuing spot-check queries (comparing retrieval results to expectations).

8. Automate retrieval quality evaluation using recall@5 on a set of queries with known source document IDs, confirming that relevant chunks are present in the retrieved set.

