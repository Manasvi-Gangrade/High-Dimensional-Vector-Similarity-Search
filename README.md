# High-Dimensional Approximate Nearest Neighbor (ANN) Search
**Department of AI & Machine Learning, Indore Institute of Science and Technology (IIST)**

This repository contains the official implementation of the hybrid index framework combining **Product Quantization (PQ)**, **Random Bit Quantization (RaBitQ)**, and **Hierarchical Navigable Small World (HNSW)** graphs. The codebase serves as an experimental environment for evaluating similarity search recall, query throughput, memory compression, and metadata attribute-filtering on high-dimensional vectors.

---

## Architectural Components

The framework integrates four distinct algorithmic components targeting the memory-accuracy trade-off in vector search:

| Component | Modules | Description |
| :--- | :--- | :--- |
| **Learned PQ** | `src/learned_pq.py` | Embedding-aware quantization applying a learned PCA rotation to de-correlate subspaces, paired with adaptive variance-based subspace allocation. |
| **RaBitQ** | `src/learned_pq.py` | Random Bit Quantization (SIGMOD 2024 / TKDE) that projects normalized vectors onto a random orthogonal basis and compresses each coordinate to a single bit ($\text{sign}(Px)$), facilitating popcount distance computation. |
| **Hybrid HNSW+PQ** | `src/hnsw_pq_index.py` | A two-stage index that performs navigable search on an HNSW graph using reconstructed PQ distances, followed by exact Euclidean distance reranking on the top candidates. |
| **Attribute Filtering** | `src/attribute_filter.py` | A partitioned sub-indexing architecture (inspired by the *iRangeGraph* paradigm) designed to resolve hybrid predicate-vector queries without the recall collapse of post-filtering. |

---

## Repository Structure

```
Similarity Search Code/
├── main.py                  # Entry point for the full evaluation pipeline
├── requirements.txt         # Package dependencies
├── .gitignore               # Excludes virtual environments, datasets, and local outputs
├── src/
│   ├── data_loader.py       # Generators for synthetic and correlated datasets
│   ├── learned_pq.py        # Quantization models (StandardPQ, LearnedPQ, RaBitQ)
│   ├── hnsw_pq_index.py     # Graph-hybrid index classes and baseline wrappers
│   ├── attribute_filter.py  # Range and categorical attribute filtering index
│   ├── evaluate.py          # Benchmark scripts for all 6 experiments
│   └── plot_results.py      # Plotting scripts for publication figures
└── results/                 # Destination for plots, transcripts, and raw results
```

---

## Setup and Installation

### 1. Requirements Installation
To install the dependencies, execute:
```bash
pip install -r requirements.txt
```

### 2. Execution Modes

* **Quick Execution Mode** (Runs on small sample sizes for verification in under 2 minutes):
  ```bash
  python main.py --quick
  ```

* **Full Evaluation Mode** (Runs on full benchmark sizes to generate final publication metrics):
  ```bash
  python main.py
  ```

---

## Experimental Suite

The framework automatically evaluates the following six parameters:

### Experiment 1: Quantization Reconstruction Error
Compares the average $L_2$ reconstruction error of Standard PQ against Learned PQ across codebook sizes $m \in \{4, 8, 16\}$. Learned PQ leverages the PCA-rotated, non-isotropic variance distribution of the dataset to achieve lower distortion.

### Experiment 2: Recall@10 vs. QPS Trade-off
Benchmarks queries-per-second (QPS) against accuracy (Recall@10). The baseline methods (Exact Search, Pure HNSW, and IVF-PQ) are evaluated alongside Standard and Learned Hybrid HNSW-PQ configurations.

### Experiment 3: Predicate Attribute-Filtering
Evaluates routing performance on queries containing scalar metadata constraints (e.g., categorical match and range filters). Compares partitioned sub-indexing against naive post-filtering.

### Experiment 4: High-Dimensional Scalability
Evaluates indexing efficiency and search recall on high-dimensional vectors ($d=768$) to simulate workloads involving dense transformer-based sentence embeddings (e.g., BERT/RoBERTa).

### Experiment 5: Ablation Study on Reranking Factor ($\alpha$)
Measures the effect of the reranking candidate pool size ($\alpha$) on recall and query throughput. Setting $\alpha \in [1, 50]$ identifies the optimal transition point between stage-1 graph routing and stage-2 full-precision verification.

### Experiment 6: Dynamic Index Maintenance
Evaluates streaming write/delete workloads on the index:
* **Dynamic Additions**: Encodes and appends new vectors into the HNSW graph using the pre-trained PQ codebooks.
* **Tombstone Deletions**: Flags removed vectors in a hash table to guarantee deleted indices are immediately filtered out from query search results.

---

## Automated Output and Metrics Logging

All runs generate formatted metrics inside the `./results/` directory:
1. **`results/benchmark_log.txt`**: Complete textual transcript of the execution, containing formatted ASCII tables, build times, and evaluation logs.
2. **`results/raw_results.json`**: Structured JSON format containing the raw numerical points of the runs for external analysis.
3. **Publication-Ready Figures** (Matplotlib PNG outputs):
   * `fig1_recall_qps.png`: Throughput vs. Recall curves.
   * `fig2_reconstruction.png`: Reconstruction distortion comparisons.
   * `fig3_filtering.png`: Sub-index routing vs. post-filtering query latency.
   * `fig4_ablation.png`: Reranking factor ($\alpha$) ablation curves.
   * `fig5_memory.png`: RAM footprint comparison.

---

## API Reference and Examples

### Quantizing Vectors with Learned PQ
```python
from src.learned_pq import LearnedPQ

# Initialize Learned Product Quantization with 8 subvectors
lpq = LearnedPQ(m=8, K=256, use_rotation=True, adaptive_split=True)
lpq.train(X_train)

# Encode to codes (n, m) and decode back to original space
codes = lpq.encode(X_test)
X_reconstructed = lpq.decode(codes)
```

### Creating and Searching the Hybrid Index
```python
from src.hnsw_pq_index import HybridPQHNSW

# Initialize HNSW index built on PQ-reconstructed vectors
index = HybridPQHNSW(d=128, m=8, K=256, M=32, rerank_factor=10, use_learned_pq=True)
index.build(X_database)

# Perform 2-stage search
distances, indices = index.search(X_queries, k=10)
```

### Partitioned Attribute-Filtering Search
```python
from src.attribute_filter import AttributeFilteredIndex

index = AttributeFilteredIndex(d=128)
index.add(vectors, attributes)  # attributes = [{'category': 'books', 'price': 300}, ...]

# Routes query directly to category sub-index
results, latency = index.search(query_vec, k=10, category='books', max_price=500)
```

---

## Academic References

* Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for Nearest Neighbor Search. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.
* Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE TPAMI*.
* Gao, J., & Long, C. (2024). RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. *ACM SIGMOD*.
