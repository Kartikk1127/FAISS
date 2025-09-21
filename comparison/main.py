from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import time


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


def benchmark_index(index, embeddings_np, query_embeddings, index_name, k=5):
    """Benchmark index performance"""
    print(f"\n=== {index_name} INDEX ===")

    # Single query benchmark
    start_time = time.time()
    distances, indices = index.search(query_embeddings, k)
    single_query_time = time.time() - start_time

    # Multiple queries benchmark (simulate real usage)
    num_queries = 10
    start_time = time.time()
    for i in range(num_queries):
        # Use different chunks as queries
        query_idx = i % len(embeddings_np)
        single_query = embeddings_np[query_idx:query_idx + 1]
        index.search(single_query, k)
    batch_time = time.time() - start_time

    print(f"Single query time: {single_query_time * 1000:.2f} ms")
    print(f"Average query time ({num_queries} queries): {(batch_time / num_queries) * 1000:.2f} ms")
    print(f"Index size in memory: ~{index.ntotal * embeddings_np.shape[1] * 4 / 1024 / 1024:.2f} MB")

    return distances, indices, single_query_time


def compare_results(flat_results, hnsw_results, k=5):
    """Compare result quality between indexes"""
    flat_distances, flat_indices,_ = flat_results
    hnsw_distances, hnsw_indices,_ = hnsw_results

    # Calculate recall@k (how many results match)
    matches = len(set(flat_indices[0][:k]) & set(hnsw_indices[0][:k]))
    recall = matches / k

    print(f"\n=== RESULT COMPARISON ===")
    print(f"Recall@{k}: {recall:.2f} ({matches}/{k} results match)")
    print(f"Flat top result similarity: {flat_distances[0][0]:.3f}")
    print(f"HNSW top result similarity: {hnsw_distances[0][0]:.3f}")

    return recall


# Load and prepare data
print("Loading document...")
with open('nimbbl_reference.txt', 'r', encoding='utf-8') as file:
    document_text = file.read()

chunks = chunk_document(document_text, chunk_size=500, overlap=20)
print(f"Created {len(chunks)} chunks")

print("\nGenerating embeddings...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)
embeddings_np = np.array(embeddings).astype('float32')

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings_np)

d = embeddings_np.shape[1]
k = 5

print(f"Dataset: {len(chunks)} chunks, {d} dimensions")

# === CREATE INDEXES ===

print("\n" + "=" * 50)
print("CREATING INDEXES")
print("=" * 50)

# 1. Flat Index (Exact)
start_time = time.time()
flat_index = faiss.IndexFlatIP(d)
flat_index.add(embeddings_np)
flat_build_time = time.time() - start_time

print(f"Flat index built in {flat_build_time * 1000:.2f} ms")

# 2. HNSW Index (Approximate)
start_time = time.time()
M = 64
ef_construction = 64
ef_search = 32

hnsw_index = faiss.IndexHNSWFlat(d, M)
hnsw_index.hnsw.efConstruction = ef_construction
hnsw_index.hnsw.efSearch = ef_search
hnsw_index.add(embeddings_np)
hnsw_build_time = time.time() - start_time

print(f"HNSW index built in {hnsw_build_time * 1000:.2f} ms")

# === BENCHMARK SEARCHES ===

print("\n" + "=" * 50)
print("SEARCH BENCHMARKS")
print("=" * 50)

# Create a test query
query_text = "payment processing"
query_embedding = model.encode([query_text]).astype('float32')
faiss.normalize_L2(query_embedding)

# Benchmark both indexes
flat_results = benchmark_index(flat_index, embeddings_np, query_embedding, "FLAT", k)
hnsw_results = benchmark_index(hnsw_index, embeddings_np, query_embedding, "HNSW", k)

# Compare result quality
recall = compare_results(flat_results, hnsw_results, k)

# === SUMMARY ===
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Dataset size: {len(chunks)} chunks")
print(f"Build time - Flat: {flat_build_time * 1000:.2f} ms, HNSW: {hnsw_build_time * 1000:.2f} ms")
print(f"Search speed - Flat: {flat_results[2] * 1000:.2f} ms, HNSW: {hnsw_results[2] * 1000:.2f} ms")
print(f"Result quality - Recall@{k}: {recall:.2f}")

print(f"\n📊 For {len(chunks)} chunks:")
if len(chunks) < 1000:
    print("   → Flat index is fine (small dataset)")
    print("   → HNSW overhead not worth it yet")
else:
    print("   → HNSW shows its benefits with larger datasets")

print(f"\n🔍 Speed difference: {(flat_results[2] / hnsw_results[2]):.1f}x")

# === INTERACTIVE SEARCH ===
print("\n" + "=" * 30)
print("INTERACTIVE SEARCH TEST")
print("=" * 30)

while True:
    query = input("\nEnter search query (or 'quit'): ")
    if query.lower() == 'quit':
        break

    query_emb = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_emb)

    # Search both indexes
    print(f"\nQuery: '{query}'")

    flat_dist, flat_idx = flat_index.search(query_emb, 3)
    hnsw_dist, hnsw_idx = hnsw_index.search(query_emb, 3)

    print(f"\nFlat results:")
    for i in range(min(3, len(flat_idx[0]))):
        if flat_dist[0][i] > 0.3:
            print(f"  {flat_dist[0][i]:.3f}: {chunks[flat_idx[0][i]][:100]}...")

    print(f"\nHNSW results:")
    for i in range(min(3, len(hnsw_idx[0]))):
        if hnsw_dist[0][i] > 0.3:
            print(f"  {hnsw_dist[0][i]:.3f}: {chunks[hnsw_idx[0][i]][:100]}...")

print("\nComparison complete!")