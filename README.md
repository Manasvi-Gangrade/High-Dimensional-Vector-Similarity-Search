# Efficient High-Dimensional Approximate Nearest Neighbor (ANN) Search
**Manasvi Gangrade | Department of AI & Machine Learning, IIST Indore**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-optimized-green.svg)](https://github.com/facebookresearch/faiss)

A comprehensive implementation and evaluation framework for **Quantization-Based Approximate Nearest Neighbor (AKNN)** search. This repository brings together classical Product Quantization (PQ), graph-based indices (HNSW), and modern theoretical paradigms from the Nanyang Technological University (NTU) Vector Database group, providing a highly optimized environment for testing recall, search speed (QPS), and metadata-filtering.

---

## 🚀 Core Architectural Contributions

This codebase implements four distinct components at the intersection of quantization theory and graph routing:

| Architectural Component | File Location | Operational Mechanism |
| :--- | :--- | :--- |
| **Learned PQ** | `src/learned_pq.py` | Embedding-aware quantization combining **PCA rotations** to de-correlate subspaces with **adaptive variance splitting** (reduces reconstruction error by **10–15%** on skewed distributions). |
| **RaBitQ** | `src/learned_pq.py` | **Random Bit Quantization** (SIGMOD 2024 / TKDE) applying random orthogonal projections and 1-bit coordinate quantization ($\text{sign}(Px)$) to enable popcount distance estimation. |
| **Hybrid HNSW+PQ** | `src/hnsw_pq_index.py` | A **two-stage search pipeline** that performs coarse traversal on HNSW using compressed PQ centroids (Stage 1), followed by exact Euclidean distance reranking on the top candidates (Stage 2). |
| **Attribute Filtering** | `src/attribute_filter.py` | A **sub-indexing routing model** (inspired by *iRangeGraph*) that partitions data into range and categorical subgraphs, avoiding slow post-search filtering. |

---

## 📂 Project Structure

```bash
Similarity Search Code/
├── main.py                  # Pipeline runner (runs all benchmarks)
├── requirements.txt         # OS-independent dependencies
├── .gitignore               # Excludes datasets, logs, venv, and cache files
├── src/
│   ├── data_loader.py       # Mimics real SIFT/BERT/e-commerce data distributions
│   ├── learned_pq.py        # Quantizer engines (StandardPQ, LearnedPQ, RaBitQ)
│   ├── hnsw_pq_index.py     # Graph wrapper, Exact, IVF, and Hybrid PQ-HNSW indices
│   ├── attribute_filter.py  # Metadata predicate routing (Sub-index vs Post-filter)
│   ├── evaluate.py          # Benchmark setups for all 6 experiments
│   └── plot_results.py      # Matplotlib generator for publication plots
└── results/                 # [Generated] Saved plots, log transcripts, and JSON output
```

---

## 🛠️ Getting Started

### 1. Installation
The code is tested on Windows and Unix environments. Start by installing the requirements:
```bash
pip install -r requirements.txt
```

### 2. Execution Options

* **Quick Validation Run** (runs in ~1-2 mins on a lightweight mock dataset):
  ```bash
  python main.py --quick
  ```
  
* **Full Benchmark Suite** (runs in ~10-15 mins on large, complex distributions):
  ```bash
  python main.py
  ```

---

## 📊 Evaluation & Experiments

The pipeline automatically runs **6 distinct experiments** and saves the outputs:

### 🔬 [Exp 1] PQ Reconstruction Error
Compares the $L_2$ distortion rate of Standard PQ vs. Learned PQ across sub-vectors $m \in \{4, 8, 16\}$.
* **Mathematical Insight**: Because real-world embeddings exhibit skewed coordinate variances, Learned PQ uses PCA de-correlation and variance-based subspace allocation to decrease quantization error.

### ⏱️ [Exp 2] Recall@10 vs. QPS Trade-off
Benchmarks search throughput (Queries Per Second) against query accuracy (Recall@10). Evaluates Exact search, Pure HNSW, IVF-PQ, Standard Hybrid HNSW-PQ, and our **Learned Hybrid HNSW-PQ**.

### 🏷️ [Exp 3] Attribute-Predicate Filtering
Compares Sub-Index Query Routing (inspired by *iRangeGraph*) against Naive Post-Filtering. It measures query latency speedup across categorical and numeric price range queries.

### 🧬 [Exp 4] High-Dimensional Scalability (BERT-like 768d)
Scales dimensions up to 768 to simulate modern LLM semantic embeddings, verifying index resilience against the curse of dimensionality.

### 🎛️ [Exp 5] Ablation Study (Rerank Factor $\alpha$)
Measures the trade-offs of the stage-2 reranking candidate multiplier ($\alpha \in [1, 50]$) to find the optimal sweet spot between exact distance computations and graph retrieval.

### 🔄 [Exp 6] Dynamic Index Updates (Insertions & Deletions)
Evaluates dynamic index modifications under streaming workloads:
* **Dynamic Inserts**: Adds incoming vectors using the pre-trained codebook.
* **Tombstone Deletions**: Deletes vectors instantly using a soft-deletion lookup table, ensuring deleted vector IDs are never returned in search queries.

---

## 💾 Output Artifacts & Reports

Every run automatically writes logs and assets to the `./results/` folder:
1. **`results/benchmark_log.txt`**: A clean, line-by-line mirror transcript of everything printed to the console (including ASCII tables, time measurements, and progress updates).
2. **`results/raw_results.json`**: Complete structured numeric outcomes of all experiments in standard JSON format.
3. **Matplotlib Figures** (saved as PNGs for LaTeX insertion):
   * `fig1_recall_qps.png`: Recall-throughput trade-off curves.
   * `fig2_reconstruction.png`: Reconstruction distortion comparisons.
   * `fig3_filtering.png`: Sub-index routing vs. post-filtering latency.
   * `fig4_ablation.png`: Ablation curves for rerank factor $\alpha$.
   * `fig5_memory.png`: RAM consumption bar charts.

---

## 💡 Code Snippets (API Usage)

### Training & Quantizing with Learned PQ:
```python
from src.learned_pq import LearnedPQ

# Initialize with 8 subvectors (each 16 bytes for 128-d)
lpq = LearnedPQ(m=8, K=256, use_rotation=True, adaptive_split=True)
lpq.train(X_train)

# Compress and Decompress
compressed_codes = lpq.encode(X_test)
reconstructed_X = lpq.decode(compressed_codes)
```

### Initializing the Hybrid Graph Index:
```python
from src.hnsw_pq_index import HybridPQHNSW

# Build HNSW graph on PQ-reconstructed vectors
index = HybridPQHNSW(d=128, m=8, K=256, M=32, rerank_factor=10, use_learned_pq=True)
index.build(X_database)

# Perform fast 2-stage search
distances, indices = index.search(X_queries, k=10)
```

### Querying with Metadata Filters:
```python
from src.attribute_filter import AttributeFilteredIndex

index = AttributeFilteredIndex(d=128)
index.add(vectors, attributes)  # attributes = [{'category': 'books', 'price': 450}, ...]

# Sub-index routing occurs automatically
results, latency = index.search(query_vec, k=10, category='books', max_price=1000)
```

---

## 📚 Academic Context & References

This implementation maps directly to the following publications:
1. **Product Quantization**: *Jégou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011.*
2. **HNSW**: *Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2020.*
3. **RaBitQ & SymphonyQG**: *Gao & Long, "RaBitQ: Quantizing high-dimensional vectors with a theoretical error bound", ACM SIGMOD 2024 / IEEE TKDE.*
