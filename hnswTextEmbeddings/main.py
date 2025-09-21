from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

k=4


def clean_chunk(text):
    # Remove non-printable characters and keep only ASCII
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove very short or empty chunks
    text = text.strip()

    return text

def chunk_and_clean_document(text, chunk_size=200, overlap=10):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Clean the chunk
        clean_chunk_text = clean_chunk(chunk)

        # Only keep chunks that are meaningful after cleaning
        if len(clean_chunk_text) > 50:  # At least 50 characters
            chunks.append(clean_chunk_text)

        start += chunk_size - overlap

    return chunks

# Read and chunk document
print("Loading and cleaning document...")
with open('Designing Data-Intensive Applications The Big Ideas Behind Reliable, Scalable, and Maintainable Systems by Martin Kleppmann (z-lib.org).txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

print("✅ File loaded successfully")

chunks = chunk_and_clean_document(document_text, chunk_size=200, overlap=10)
print(f"✅ Created {len(chunks)} chunks")
# Generate embeddings
print("\nGenerating embeddings...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model loaded")

# Test progressive scaling to find breaking point
test_sizes = [1, 2, 5, 10, 25, 50, 100, 125]
max_working_size = 1

for size in test_sizes:
    print(f"\n=== Testing with {size} chunks ===")

    test_chunks = chunks[:size]
    test_embeddings = model.encode(test_chunks)
    test_embeddings_np = np.array(test_embeddings).astype('float32')
    faiss.normalize_L2(test_embeddings_np)

    try:
        index = faiss.IndexHNSWFlat(384, 4)
        index.hnsw.efConstruction = 8
        index.hnsw.efSearch = 8

        index.add(test_embeddings_np)
        print(f"SUCCESS: {size} chunks work fine")
        max_working_size = size  # Update the max working size

    except Exception as e:
        print(f"FAILED at {size} chunks: {e}")
        print(f"Maximum working size is {max_working_size}")
        break

# Use the maximum working size for the final search
print(f"\n=== Using {max_working_size} chunks for final search ===")
final_chunks = chunks[:max_working_size]
embeddings = model.encode(final_chunks)
embeddings_np = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings_np)

d = embeddings_np.shape[1]

# Use conservative parameters since we found the scaling limit
M = 4
ef_search = 8
ef_construction = 8

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = ef_search
print("✅ HNSW index created")

index.add(embeddings_np)
print(f"✅ HNSW index created with {index.ntotal} vectors")

# Search
k = min(4, max_working_size)  # Don't search for more results than we have chunks
query = model.encode(["different types of replication strategies in distributed systems"]).astype('float32')
faiss.normalize_L2(query.reshape(1, -1))
D, I = index.search(query.reshape(1, -1), k)

print("✅ Search completed")

for idx, dist in zip(I[0], D[0]):
    if idx < len(final_chunks):  # Safety check
        print(f"{dist:.3f}: {final_chunks[idx][:200]}...")