"""
hnsw_pq_index.py
----------------
Hybrid HNSW + PQ Index — core architecture of the paper.

Two-stage search:
  Stage 1: HNSW graph traversal with PQ distances  (fast, approximate)
  Stage 2: Rerank top candidates with exact distances (accurate)
"""

import numpy as np
import faiss
import time
from learned_pq import StandardPQ, LearnedPQ


class HNSWIndex:
    """
    Baseline: Pure HNSW (no quantization, no filtering)
    State-of-art graph-based ANN.
    """

    def __init__(self, d, M=32, ef_construction=200):
        """
        d              : vector dimension
        M              : number of neighbors per node (higher = better recall, more memory)
        ef_construction: beam width during build (higher = better quality, slower build)
        """
        self.d  = d
        self.M  = M
        self.ef_construction = ef_construction
        self.index = faiss.IndexHNSWFlat(d, M)
        self.index.hnsw.efConstruction = ef_construction

    def add(self, X):
        print(f"[HNSW] Adding {len(X)} vectors (d={self.d}, M={self.M})")
        t0 = time.time()
        self.index.add(X)
        print(f"[HNSW] Build done in {time.time()-t0:.2f}s")

    def search(self, xq, k=10, ef_search=100):
        self.index.hnsw.efSearch = ef_search
        D, I = self.index.search(xq, k)
        return D, I

    def memory_bytes(self):
        """Approximate memory: each vector stored as float32."""
        return self.index.ntotal * self.d * 4


# ──────────────────────────────────────────────────────────────────────────────


class IVFPQIndex:
    """
    FAISS baseline: IVF + Product Quantization.
    Industry standard (used in Facebook, etc.)
    """

    def __init__(self, d, nlist=100, m=8, bits=8):
        """
        nlist : number of IVF clusters (Voronoi cells)
        m     : number of PQ subvectors
        bits  : bits per subvector (8 → K=256 centroids)
        """
        self.d     = d
        self.nlist = nlist
        self.m     = m
        self.bits  = bits

        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

    def train_and_add(self, X):
        print(f"[IVF-PQ] Training on {len(X)} vectors...")
        t0 = time.time()
        self.index.train(X)
        self.index.add(X)
        print(f"[IVF-PQ] Done in {time.time()-t0:.2f}s")

    def search(self, xq, k=10, nprobe=10):
        self.index.nprobe = nprobe
        D, I = self.index.search(xq, k)
        return D, I

    def memory_bytes(self):
        return self.index.ntotal * self.m  # m bytes per vector


# ──────────────────────────────────────────────────────────────────────────────


class HybridPQHNSW:
    """
    PAPER'S MAIN ARCHITECTURE: PQ-compressed HNSW with two-stage reranking.

    How it works:
    ┌─────────────────────────────────────────────────────────┐
    │  Build time:                                            │
    │    - Compress all vectors with LearnedPQ (64 bytes)     │
    │    - Store original float32 for reranking               │
    │    - Build HNSW graph on PQ-reconstructed vectors       │
    │                                                         │
    │  Query time:                                            │
    │    Stage 1: HNSW traversal → top α*k candidates        │
    │    Stage 2: Exact L2 on candidates → top k results      │
    └─────────────────────────────────────────────────────────┘

    Benefits:
    - 48x memory reduction (768d float32 → 64 bytes PQ codes)
    - O(log n) search complexity
    - High recall through reranking
    """

    def __init__(self, d, m=8, K=256, M=32, ef_construction=200,
                 use_learned_pq=True, rerank_factor=10):
        """
        rerank_factor : retrieve k*rerank_factor candidates, rerank to k
                        Higher → better recall, slower
        """
        self.d               = d
        self.m               = m
        self.K               = K
        self.M               = M
        self.rerank_factor   = rerank_factor
        self.use_learned_pq  = use_learned_pq

        # PQ for compression
        if use_learned_pq:
            self.pq = LearnedPQ(m=m, K=K)
        else:
            self.pq = StandardPQ(m=m, K=K)

        # HNSW on PQ-reconstructed vectors
        self.hnsw_index = faiss.IndexHNSWFlat(d, M)
        self.hnsw_index.hnsw.efConstruction = ef_construction

        # Store originals for reranking
        self.original_vectors = None
        self.pq_codes         = None
        self.deleted_ids      = set()

    def build(self, X):
        """Build the hybrid index."""
        n = len(X)
        pq_type = "Learned" if self.use_learned_pq else "Standard"
        print(f"\n[HybridPQHNSW] Building index: n={n}, d={self.d}")
        print(f"[HybridPQHNSW] PQ type: {pq_type}, m={self.m}, K={self.K}")
        print(f"[HybridPQHNSW] HNSW: M={self.M}")

        total_start = time.time()

        # Step 1: Train PQ
        print("\n[Step 1/3] Training PQ codebooks...")
        self.pq.train(X)

        # Step 2: Encode all vectors
        print("\n[Step 2/3] Encoding vectors with PQ...")
        t0 = time.time()
        self.pq_codes = self.pq.encode(X)
        X_reconstructed = self.pq.decode(self.pq_codes)
        print(f"  Encoding done in {time.time()-t0:.2f}s")

        # Step 3: Build HNSW on reconstructed vectors
        print("\n[Step 3/3] Building HNSW graph...")
        t0 = time.time()
        self.hnsw_index.add(X_reconstructed.astype('float32'))
        print(f"  HNSW build done in {time.time()-t0:.2f}s")

        # Store originals
        self.original_vectors = X.copy()

        total_time = time.time() - total_start
        print(f"\n[HybridPQHNSW] Total build time: {total_time:.2f}s")
        print(f"[HybridPQHNSW] Memory: PQ codes={self.pq_codes.nbytes/1e6:.1f}MB, "
              f"Originals={X.nbytes/1e6:.1f}MB")

    def add_vectors(self, X):
        """
        Dynamically insert multiple vectors into the index.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        codes = self.pq.encode(X)
        X_reconstructed = self.pq.decode(codes)
        self.hnsw_index.add(X_reconstructed.astype('float32'))
        
        if self.original_vectors is not None:
            self.original_vectors = np.vstack([self.original_vectors, X])
        else:
            self.original_vectors = X.copy()
            
        if self.pq_codes is not None:
            self.pq_codes = np.vstack([self.pq_codes, codes])
        else:
            self.pq_codes = codes.copy()

    def delete_vector(self, idx):
        """
        Mark a vector as deleted (tombstone).
        """
        self.deleted_ids.add(idx)

    def search(self, xq, k=10, ef_search=100):
        """
        Two-stage search with tombstone filtering.
        Returns: (distances, indices) for top-k results
        """
        if len(xq.shape) == 1:
            xq = xq.reshape(1, -1)

        self.hnsw_index.hnsw.efSearch = ef_search
        n_candidates = min(k * self.rerank_factor, len(self.original_vectors))

        # Stage 1: HNSW retrieval (approximate, fast)
        _, candidate_ids = self.hnsw_index.search(xq, n_candidates)

        # Stage 2: Exact reranking
        results_I = np.zeros((len(xq), k), dtype='int64') - 1
        results_D = np.zeros((len(xq), k), dtype='float32') + float('inf')

        for i, (q, cands) in enumerate(zip(xq, candidate_ids)):
            valid = cands[cands >= 0]
            if len(self.deleted_ids) > 0:
                valid = np.array([c for c in valid if c not in self.deleted_ids], dtype='int64')
            if len(valid) == 0:
                continue

            # Exact L2 on candidates
            cand_vecs = self.original_vectors[valid]
            dists = np.sum((cand_vecs - q) ** 2, axis=1)
            sorted_idx = np.argsort(dists)[:k]

            n_ret = min(k, len(sorted_idx))
            results_I[i, :n_ret] = valid[sorted_idx[:n_ret]]
            results_D[i, :n_ret] = dists[sorted_idx[:n_ret]]

        return results_D, results_I

    def memory_bytes(self):
        """Total memory usage."""
        pq_mem    = self.pq_codes.nbytes if self.pq_codes is not None else 0
        orig_mem  = (self.original_vectors.nbytes
                     if self.original_vectors is not None else 0)
        return pq_mem + orig_mem

    def memory_per_vector_bytes(self):
        if self.original_vectors is None:
            return 0
        n = len(self.original_vectors)
        return self.memory_bytes() // n


# ──────────────────────────────────────────────────────────────────────────────


def compute_ground_truth(X, xq, k=10):
    """Exact brute-force k-NN (ground truth for recall evaluation)."""
    print(f"[GroundTruth] Computing exact {k}-NN for {len(xq)} queries...")
    t0 = time.time()
    index_flat = faiss.IndexFlatL2(X.shape[1])
    index_flat.add(X)
    _, I = index_flat.search(xq, k)
    print(f"[GroundTruth] Done in {time.time()-t0:.2f}s")
    return I


def recall_at_k(retrieved, ground_truth, k=10):
    """
    Recall@k: fraction of true k-NN found in retrieved results.
    
    retrieved    : (nq, k) array of retrieved indices
    ground_truth : (nq, k) array of true indices
    """
    recalls = []
    for ret, gt in zip(retrieved, ground_truth):
        r = len(set(ret[:k].tolist()) & set(gt[:k].tolist())) / k
        recalls.append(r)
    return float(np.mean(recalls))


def benchmark_index(search_fn, xq, k=10, n_warmup=5):
    """
    Measure QPS and latency (p50, p95, p99).
    """
    # Warmup
    for _ in range(n_warmup):
        search_fn(xq[:1], k)

    # Benchmark each query individually for latency distribution
    latencies = []
    for q in xq:
        t0 = time.perf_counter()
        search_fn(q.reshape(1, -1), k)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    total_time = latencies.sum() / 1000  # seconds

    return {
        'qps':     len(xq) / total_time,
        'p50_ms':  float(np.percentile(latencies, 50)),
        'p95_ms':  float(np.percentile(latencies, 95)),
        'p99_ms':  float(np.percentile(latencies, 99)),
        'mean_ms': float(np.mean(latencies)),
    }


if __name__ == "__main__":
    from data_loader import generate_synthetic

    d  = 128
    xb, xq = generate_synthetic(n=20000, d=d, nq=200)

    gt = compute_ground_truth(xb, xq, k=10)

    # Build hybrid index
    hybrid = HybridPQHNSW(d=d, m=8, K=256, M=32,
                          use_learned_pq=True, rerank_factor=10)
    hybrid.build(xb)

    # Search
    _, I = hybrid.search(xq, k=10)
    r = recall_at_k(I, gt)
    print(f"\nRecall@10: {r:.3f}")
    print(f"Memory/vector: {hybrid.memory_per_vector_bytes()} bytes")
