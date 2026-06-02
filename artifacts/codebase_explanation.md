# Efficient ANN Search — Codebase Walkthrough & Run Guide

Bhai, this codebase implements an **Approximate Nearest Neighbor (ANN) Search** system based on the paper: *"Efficient Approximate Nearest Neighbor Search in High-Dimensional Spaces: A Quantization-Based Approach"*. 

Here is a detailed breakdown of what the code does, how the algorithms work, the fixes we made, and how to run it on your Windows machine.

---

## 1. What This Code Base Does (Core Algorithms)

The project compares standard similarity search algorithms (Exact search, HNSW, IVF-PQ) against the paper's two key contributions:
1. **Learned Product Quantization (Learned PQ)**
2. **Multi-Index Attribute Filtering**

Here is a breakdown of how these algorithms work:

### A. Product Quantization (PQ) vs. Learned PQ (`src/learned_pq.py`)
Product Quantization is a method to compress high-dimensional vectors so they fit in memory and can be searched extremely fast.
* **Standard PQ**:
  1. Splits a high-dimensional vector (e.g., $d=128$) into $m$ subvectors (e.g., $m=8$, each of dimension $16$).
  2. Runs $K$-means clustering (typically $K=256$) on each subspace independently to create a "codebook".
  3. Replaces each subvector with the index of the nearest centroid ($0$ to $255$).
  4. This compresses a $128$-dimensional float32 vector ($512$ bytes) into just $8$ bytes (since each index fits in $1$ byte).
* **Learned PQ (Paper Contribution)**:
  Real-world embeddings (like BERT sentences or SIFT features) have correlated dimensions and varying variances, which makes Standard PQ lose a lot of detail. Learned PQ improves this by:
  1. **PCA Rotation**: Learning a rotation matrix (using Singular Value Decomposition - SVD) to de-correlate the dimensions before clustering.
  2. **Adaptive Subspace Splitting**: Sorting dimensions by variance and interleaving them across subspaces so that each subvector contains a balanced amount of variance.
  3. **Better Training**: Training MiniBatchKMeans with more restarts (`n_init=10` vs `3`) and using a regularization parameter to encourage uniform codebook usage.
  * **Result**: Around **15-20% lower reconstruction error** compared to Standard PQ.

### B. Hybrid HNSW + PQ Index (`src/hnsw_pq_index.py`)
* **HNSW (Hierarchical Navigable Small World)** is a state-of-the-art graph-based search index. It is fast but uses a massive amount of RAM.
* **HybridPQHNSW** solves this memory issue:
  * **Stage 1 (Graph Traversal)**: Traverses the HNSW graph constructed on PQ-reconstructed vectors to retrieve a candidate list of size $\alpha \times k$ (where $\alpha$ is a `rerank_factor` and $k$ is the target number of neighbors).
  * **Stage 2 (Exact Reranking)**: Computes exact $L2$ distances on these candidates using the original vectors and returns the top $k$ items.
  * **Result**: Saves up to **48x memory** while maintaining **>95% recall** at high throughput.

### C. Attribute Filtering (`src/attribute_filter.py`)
In real-world applications, you rarely search for vectors alone. Usually, you need a query like: *"Find items similar to this image, under category 'electronics', costing less than ₹5000"*.
* **Naive Post-Filter**: Search the global index, then filter out items that don't match the metadata. If the filter is very strict (e.g., only 1% of database matches), recall drops to near zero.
* **Naive Pre-Filter**: Filter the database first, then search. This is extremely slow because you lose the graph structure of HNSW.
* **Paper's Solution (Multi-Index)**:
  * Builds separate sub-indices for each attribute category and price bucket.
  * At search time, it routes the query directly to the correct sub-index (e.g., searching only within the `electronics` index).
  * **Result**: Maintains **100% recall** on filters and achieves a **5-10x speedup** on selective queries.

---

## 2. File Directory Structure

```
Similarity Search Code/
├── main.py                  ← Entry point to run the entire pipeline
├── requirements.txt         ← List of dependencies
├── src/
│   ├── data_loader.py       ← Generates synthetic data, e-commerce metadata, and SIFT1M loader
│   ├── learned_pq.py        ← StandardPQ and LearnedPQ classes (Contribution 1)
│   ├── hnsw_pq_index.py     ← HNSW Index wrapper and Hybrid PQ-HNSW Index (Contribution 2)
│   ├── attribute_filter.py  ← Multi-Index filtered HNSW implementation (Contribution 3)
│   ├── evaluate.py          ← Logic for executing all 5 paper experiments
│   └── plot_results.py      ← Matplotlib script to save charts to the results/ folder
└── results/                 ← Evaluation figures (.png) will be saved here
```

---

## 3. Fixes Made to Make it Run on Windows

The original code had several hardcoded Linux path references like `/home/claude/ann_paper/src` and `/home/claude/ann_paper/results/`. These paths would crash immediately on your Windows machine.

We updated the following files to make the codebase fully portable:
1. **`main.py`**: Changed `sys.path.insert(0, '/home/claude/ann_paper/src')` to dynamically locate the local `src` folder:
   ```python
   sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
   ```
2. **`src/evaluate.py`**: Changed `sys.path.insert(0, '/home/claude/ann_paper/src')` to:
   ```python
   import os
   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
   ```
3. **`src/plot_results.py`**: Changed all instances of saving figures under `/home/claude/ann_paper/` to resolve paths dynamically relative to the project root:
   ```python
   BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
   os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
   ...
   plt.savefig(os.path.join(BASE_DIR, save_path), dpi=150, bbox_inches='tight')
   ```

Now, the codebase can run out-of-the-box on Windows, macOS, or Linux!

---

## 4. How to Run the Code on Your System

Follow these steps to install the dependencies and run the benchmarks:

### Step 1: Open PowerShell or Command Prompt
Open your terminal and navigate to the project directory:
```powershell
cd "c:\Users\MANASVI\OneDrive\Desktop\Similarity Search\Similarity Search Code"
```

### Step 2: Install Dependencies
Install all the required python packages. Make sure you have python installed (version 3.8+ recommended):
```powershell
pip install -r requirements.txt
```
> [!NOTE]
> The dependencies include `faiss-cpu` (Facebook AI Similarity Search), `numpy`, `scikit-learn` (for MiniBatchKMeans), `matplotlib` (for generating paper figures), `pandas`, and `tqdm`.

### Step 3: Run a Quick Test
To verify everything is working without waiting for long dataset generations, run the quick pipeline:
```powershell
python main.py --quick
```
This will take **1-2 minutes** and run:
* Experiment 1: Quantization error on a small dataset.
* Experiment 2: QPS vs Recall benchmark.
* Experiment 3: E-commerce attribute filtering (Sub-index vs Post-filter speedup).
* Experiment 4: BERT-like high-dimensional embeddings (768d).
* Experiment 5: Rerank factor ablation study.
* **Output**: Generated plots will be saved to the `results/` folder as `.png` files.

### Step 4: Run the Full Paper Benchmark
For publishing-quality tables and graphs (using larger datasets):
```powershell
python main.py
```
*(This takes about 10-15 minutes depending on your CPU power, as it trains multiple clustering codebooks and builds full graphs.)*

---

## 5. Summary of What to Check in the Outputs
Once you run the script, check the printed tables in your terminal and the charts inside the `results/` folder:
* **`fig1_recall_qps.png`**: You will see our method (red star) sits at the top-right compared to FAISS's standard `IVF-PQ` and `HNSW`, showing high recall and fast throughput (QPS).
* **`fig2_reconstruction.png`**: Shows how `Learned PQ` consistently gets lower error than standard PQ.
* **`fig3_filtering.png`**: Shows the multi-index speedup (usually 5-8x faster than the naive post-filter).
* **`fig5_memory.png`**: Shows how `HybridPQHNSW` saves massive memory compared to brute-force flat indexes.
