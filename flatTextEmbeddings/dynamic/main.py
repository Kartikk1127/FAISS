from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def chunk_document(text, chunk_size=500, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def search_document(index, model, query_text, chunks, k=5, threshold=0.3):
    # Encode and normalize query
    query_embedding = model.encode([query_text]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # Search
    distances, indices = index.search(query_embedding, k=k)

    # Filter and format results
    results = []
    for i in range(len(distances[0])):
        if distances[0][i] > threshold:
            chunk_idx = indices[0][i]
            results.append({
                'similarity': distances[0][i],
                'chunk_index': chunk_idx,
                'content': chunks[chunk_idx]
            })

    return results


# Read and chunk document
print("Loading document...")
with open('nimbbl_reference.txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

chunks = chunk_document(document_text, chunk_size=500, overlap=20)
print(f"Created {len(chunks)} chunks")
print(f"First chunk preview: {chunks[0][:150]}...")

# Generate embeddings
print("\nGenerating embeddings...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Create FAISS index
print("Creating FAISS index...")
embeddings_np = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings_np)

index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

print(f"Index created with {index.ntotal} chunks")

# Interactive search
print("\n" + "=" * 50)
print("DOCUMENT SEARCH READY")
print("=" * 50)

while True:
    query = input("\nEnter your search query (or 'quit' to exit): ")

    if query.lower() == 'quit':
        break

    # Search document
    results = search_document(index, model, query, chunks, k=5, threshold=0.3)

    if results:
        print(f"\nFound {len(results)} relevant chunks:")
        print("-" * 50)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.3f}")
            print(f"Content: {result['content'][:200]}...")
            if len(result['content']) > 200:
                print("    (truncated - full chunk available)")
    else:
        print("No relevant chunks found. Try a different query.")

print("\nSearch session ended.")