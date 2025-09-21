from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import faiss

sentences = ["pet", "wild animal", "feline", "canine", "domestic animal"]
#
# sentences = ["car", "building", "computer"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings) #this generates 384d embeddings for each sentence

# similarity validations

# cosine similarity
cosine_similarity_matrix = cosine_similarity(embeddings)
print(cosine_similarity_matrix)
print("Cosine Similarity Matrix: ")
for i, sentence1 in enumerate(sentences):
    for j, sentence2 in enumerate(sentences):
        print(f"{sentence1} <-> {sentence2}: {cosine_similarity_matrix[i][j]:.3f}")

# Euclidean distance
euclidean_distance_matrix = euclidean_distances(embeddings)
print("Euclidean Distance Matrix: ")
print(euclidean_distance_matrix)
for i, sentence1 in enumerate(sentences):
    for j, sentence2 in enumerate(sentences):
        print(f"{sentence1} <-> {sentence2}: {euclidean_distance_matrix[i][j]:.3f}")

# Faiss implementation
print("Faiss Index Matrix: ")
embeddings_np = np.array(embeddings).astype('float32')
print(embeddings_np.shape)
dimensions = embeddings_np.shape[1]

print("Cosine Similarity using Faiss Index: ")
faiss.normalize_L2(embeddings_np)
cosine_index = faiss.IndexFlatIP(dimensions)
cosine_index.add(embeddings_np)
query1 = model.encode(["animal"]).astype('float32')
faiss.normalize_L2(query1)

distances, indices = cosine_index.search(query1, k=4)
distance_threshold = 0.6
print("Raw distances:", distances[0])
print("Raw indices:", indices[0])

filtered_results = []
for i in range(len(distances[0])):
    sentence_idx = indices[0][i]
    if distances[0][i] > distance_threshold:
        filtered_results.append((distances[0][i], sentence_idx, sentences[sentence_idx]))

    else :
        print("Skipping sentence since the similarity is lesser with an angle of:", sentences[sentence_idx],
              distances[0][i])

print("Filtered results using cosine similarity (distance, index, sentence):", filtered_results)

print("Euclidean distance metrics using Faiss Index: ")
euclidean_index = faiss.IndexFlatL2(dimensions)
euclidean_index.add(embeddings_np)
query2 = model.encode(["animal"]).astype('float32')
faiss.normalize_L2(query2)

distances, indices = euclidean_index.search(query2, k=4)
euclidean_distance_threshold = 0.7
print("Raw distances:", distances[0])
print("Raw indices:", indices[0])

filtered_results.clear()
for j in range(len(distances[0])):
    sentence_idx = indices[0][j]
    if distances[0][j] < euclidean_distance_threshold:
        filtered_results.append((distances[0][j], sentence_idx, sentences[sentence_idx]))
    else:
        print("Skipping sentence since the distance is higher with a value of:", sentences[sentence_idx], distances[0][j])

print("Filtered results using euclidean metric (distance, index, sentence):", filtered_results)