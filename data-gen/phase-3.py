import os
import time
import numpy as np
import h5py
import faiss  # Ensure you have installed faiss-cpu or faiss-gpu as needed
from concurrent.futures import ThreadPoolExecutor

# -------------------------------
# Step 1: Load Embeddings from HDF5
# -------------------------------
def load_embeddings(hdf5_file="transaction_embeddings.h5"):
    try:
        with h5py.File(hdf5_file, "r") as hf:
            embeddings = hf["embeddings"][:]
        print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        return embeddings
    except Exception as e:
        print("Error loading embeddings:", e)
        return None

embeddings = load_embeddings()
if embeddings is None:
    raise ValueError("Failed to load embeddings.")

# -------------------------------
# Step 2: Build FAISS Indexes
# -------------------------------
d = embeddings.shape[1]  # dimensionality

# 2.A: For cosine similarity, normalize embeddings and use IndexFlatIP.
# Cosine similarity is equivalent to inner product on normalized vectors.
embeddings_norm = embeddings.astype('float32')
norms = np.linalg.norm(embeddings_norm, axis=1, keepdims=True)
embeddings_norm = embeddings_norm / (norms + 1e-10)  # avoid division by zero

index_cosine = faiss.IndexFlatIP(d)  # Inner product index
index_cosine.add(embeddings_norm)
print("Cosine (normalized) index built with {} vectors.".format(index_cosine.ntotal))

# 2.B: For L2 distance, use the raw embeddings with IndexFlatL2.
embeddings_l2 = embeddings.astype('float32')
index_l2 = faiss.IndexFlatL2(d)
index_l2.add(embeddings_l2)
print("L2 index built with {} vectors.".format(index_l2.ntotal))

# 2.C: Build a quantized index to simulate fast ANN search with quantization.
# We'll only train the quantized index if we have enough training points.
n_train = embeddings_norm.shape[0]
min_required = 195  # e.g., if nlist=5, need at least 5*39=195 training points
if n_train < min_required:
    print(f"Not enough training points ({n_train}) for quantized index training. Skipping quantized index.")
    quant_index = None
else:
    nlist = n_train if n_train < 256 else 256  # use 256 clusters if possible, else use n_train
    m = 8  # number of subquantizers
    bits = 8  # bits per subquantizer
    quant_index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, m, bits)
    if not quant_index.is_trained:
        print(f"Training quantized index with nlist={nlist} on {n_train} points...")
        quant_index.train(embeddings_norm)
    quant_index.add(embeddings_norm)
    print("Quantized index built with {} vectors.".format(quant_index.ntotal))

# -------------------------------
# Step 3: Implement High-Speed Similarity Search
# -------------------------------
def search_index(query_embedding, index, k=3):
    # query_embedding should be a 2D float32 array.
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def real_time_detection(query_embedding):
    # For cosine similarity (vectors must be normalized)
    d_cos, i_cos = search_index(query_embedding, index_cosine, k=3)
    # For L2 distance (using raw embeddings)
    d_l2, i_l2 = search_index(query_embedding, index_l2, k=3)
    return d_cos, i_cos, d_l2, i_l2

# -------------------------------
# Step 4: Multi-Threaded Search Queries Simulation
# -------------------------------
def process_query(query_embedding, query_id):
    start = time.time()
    d_cos, i_cos, d_l2, i_l2 = real_time_detection(query_embedding)
    elapsed = (time.time() - start) * 1000  # in milliseconds
    print(f"Query {query_id} processed in {elapsed:.2f} ms")
    return (d_cos, i_cos, d_l2, i_l2)

# Simulate multiple concurrent queries (using first 5 embeddings as queries)
queries = [embeddings_norm[i] for i in range(min(5, embeddings_norm.shape[0]))]

print("Running multi-threaded search queries:")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_query, q, idx) for idx, q in enumerate(queries)]
    for future in futures:
        res = future.result()
        print("Cosine distances and indices:", res[0], res[1])
        print("L2 distances and indices:", res[2], res[3])

# -------------------------------
# Step 5: Quantized Index Search Example (if available)
# -------------------------------
if quant_index is not None:
    query = embeddings_norm[0]  # using first normalized embedding as query
    d_q, i_q = quant_index.search(np.array([query]).astype('float32'), k=3)
    print("Quantized index results (inner product):", d_q, i_q)
else:
    print("Quantized index not built due to insufficient training data.")

# -------------------------------
# Step 6: Simulate Distributed Index Updates (Kafka Streams & NoSQL DB Integration)
# -------------------------------
def update_index(new_embeddings, index):
    """
    Simulate an index update.
    In production, new embeddings would be streamed (e.g., via Kafka),
    and stored in a sharded NoSQL DB (like Cassandra or DynamoDB) before being added to the index.
    """
    new_embeddings = new_embeddings.astype('float32')
    # If updating the cosine index, new embeddings must be normalized.
    norms_new = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    new_embeddings_norm = new_embeddings / (norms_new + 1e-10)
    index.add(new_embeddings_norm)
    print(f"Added {new_embeddings.shape[0]} new embeddings to the index. Total now: {index.ntotal}")

# Simulate receiving new embeddings (here 2 new random vectors)
new_emb = np.random.rand(2, d).astype('float32')
update_index(new_emb, index_cosine)

print("Phase 3: Distributed Vector Search & Real-Time Detection complete.")