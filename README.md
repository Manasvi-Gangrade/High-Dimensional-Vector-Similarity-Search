# Efficient ANN Search — Paper Implementation
**Manasvi Gangrade | IIST Indore**

> Implementation of: *"Efficient Approximate Nearest Neighbor Search in High-Dimensional Spaces: A Quantization-Based Approach"*

---

## What This Code Does

This implements all 3 core contributions of the paper:

| Contribution | File | What it does |
|---|---|---|
| **Learned PQ** | `src/learned_pq.py` | Embedding-aware codebook training (15-20% better than standard PQ) |
| **Hybrid HNSW+PQ** | `src/hnsw_pq_index.py` | Two-stage search: fast graph traversal + exact reranking |
| **Attribute Filtering** | `src/attribute_filter.py` | Multi-index for hybrid vector+metadata queries |

---

## Project Structure

```
ann_paper/
├── main.py                  ← Run this for full pipeline
├── requirements.txt
├── src/
│   ├── data_loader.py       ← Dataset generation & loading
│   ├── learned_pq.py        ← StandardPQ + LearnedPQ (contribution 1)
│   ├── hnsw_pq_index.py     ← HybridPQHNSW + evaluation utils
│   ├── attribute_filter.py  ← Multi-index attribute filtering (contribution 2)
│   ├── evaluate.py          ← All 5 experiments
│   └── plot_results.py      ← Paper figures (matplotlib)
├── data/                    ← Datasets go here
└── results/                 ← Figures saved here
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test run (~2-3 minutes)
python main.py --quick

# Full run (~15-20 minutes, paper-quality results)
python main.py
```

---

## Experiments

### Exp 1 — PQ Reconstruction Error
Compares Standard PQ vs Learned PQ across m ∈ {4, 8, 16}.
**Expected**: Learned PQ shows 15-20% lower reconstruction error.

### Exp 2 — Recall@10 vs QPS
Compares all methods: Exact, HNSW, IVF-PQ, Hybrid-StdPQ, Hybrid-LearnedPQ.
**Expected**: Our method achieves >95% recall at highest QPS among guaranteed methods.

### Exp 3 — Attribute Filtering
Compares sub-index strategy vs naive post-filter.
**Expected**: 5-10x speedup on selective (category + price) queries.

### Exp 4 — High-Dimensional (768d)
Tests scalability on BERT-like embeddings.

### Exp 5 — Ablation: Rerank Factor
Shows recall-speed trade-off as rerank_factor α varies from 1 to 50.

---

## Using Real SIFT1M Dataset

```python
from src.data_loader import load_sift1m

# Downloads automatically (~160MB)
result = load_sift1m('./data/sift1m')
if result:
    xb, xq, gt = result
    # Use xb, xq, gt in your experiments
```

---

## Key Classes

```python
# Learned Product Quantization
from src.learned_pq import LearnedPQ
lpq = LearnedPQ(m=8, K=256, use_rotation=True, adaptive_split=True)
lpq.train(X_train)
codes = lpq.encode(X_test)      # compress
X_hat = lpq.decode(codes)       # reconstruct

# Hybrid Index
from src.hnsw_pq_index import HybridPQHNSW
index = HybridPQHNSW(d=128, m=8, K=256, M=32, rerank_factor=10)
index.build(X_database)
D, I = index.search(X_queries, k=10)

# Attribute Filtered Index
from src.attribute_filter import AttributeFilteredIndex
idx = AttributeFilteredIndex(d=128)
idx.add(vectors, attributes)    # attributes = [{'category': 'electronics', 'price': 999}, ...]
results, ms = idx.search(query, k=10, category='electronics', max_price=5000)
```

---

## Paper Tables → Code Mapping

| Paper Table | Experiment | Code |
|---|---|---|
| Table 1 (PQ error) | `exp1_reconstruction()` | `evaluate.py` |
| Table 2 (main results) | `exp2_recall_qps()` | `evaluate.py` |
| Table 3 (high-dim) | `exp4_high_dimensional()` | `evaluate.py` |
| Table 4 (ablation) | `exp5_ablation_rerank()` | `evaluate.py` |

---

## Paper Figures → Plot Functions

| Figure | Function | File |
|---|---|---|
| Fig 1: Recall vs QPS | `plot_recall_qps()` | `plot_results.py` |
| Fig 2: Reconstruction error | `plot_reconstruction_error()` | `plot_results.py` |
| Fig 3: Filter speedup | `plot_filtering_speedup()` | `plot_results.py` |
| Fig 4: Ablation | `plot_ablation_rerank()` | `plot_results.py` |
| Fig 5: Memory | `plot_memory_comparison()` | `plot_results.py` |

---

## References

- Jégou et al. (2011) — Product Quantization for Nearest Neighbor Search
- Malkov & Yashunin (2016/2020) — HNSW
- Johnson et al. (2019) — FAISS library
