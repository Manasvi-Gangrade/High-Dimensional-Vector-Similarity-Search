"""
data_loader.py
--------------
Dataset loading and generation for ANN experiments.
Supports: SIFT1M (real), GloVe-like (real), Synthetic (always available)
"""

import numpy as np
import os
import urllib.request
import struct
import time

# ─────────────────────────────────────────────
# Synthetic Dataset (always works, no download)
# ─────────────────────────────────────────────

def generate_synthetic(n=100_000, d=128, nq=1000, seed=42):
    """
    Generate synthetic float32 vectors.
    Good for quick testing when real data not available.
    """
    np.random.seed(seed)
    print(f"[DataLoader] Generating synthetic dataset: n={n}, d={d}, nq={nq}")

    # Clustered data (more realistic than pure random)
    n_clusters = 100
    centers = np.random.randn(n_clusters, d).astype('float32')

    # Assign each vector to a cluster
    cluster_ids = np.random.randint(0, n_clusters, n)
    xb = centers[cluster_ids] + 0.3 * np.random.randn(n, d).astype('float32')

    # Queries from same distribution
    q_cluster_ids = np.random.randint(0, n_clusters, nq)
    xq = centers[q_cluster_ids] + 0.3 * np.random.randn(nq, d).astype('float32')

    # Introduce dimension variance decay (simulate real-world skewed dimensions)
    decay = np.exp(-np.linspace(0, 2.5, d)).astype('float32')
    xb = xb * decay
    xq = xq * decay

    print(f"[DataLoader] Done. xb={xb.shape}, xq={xq.shape}")
    return xb.astype('float32'), xq.astype('float32')


# ─────────────────────────────────────────────
# SIFT1M Dataset (real benchmark)
# ─────────────────────────────────────────────

def _read_fvecs(fname):
    """Read .fvecs file format (used by SIFT1M)."""
    with open(fname, 'rb') as f:
        data = f.read()

    d = struct.unpack('i', data[:4])[0]
    record_size = 4 + d * 4  # int32 dim + d float32s
    n = len(data) // record_size

    vecs = np.zeros((n, d), dtype='float32')
    for i in range(n):
        offset = i * record_size + 4  # skip dim field
        vecs[i] = struct.unpack(f'{d}f', data[offset:offset + d * 4])

    return vecs


def _read_ivecs(fname):
    """Read .ivecs file format (used for ground truth)."""
    with open(fname, 'rb') as f:
        data = f.read()

    d = struct.unpack('i', data[:4])[0]
    record_size = 4 + d * 4
    n = len(data) // record_size

    vecs = np.zeros((n, d), dtype='int32')
    for i in range(n):
        offset = i * record_size + 4
        vecs[i] = struct.unpack(f'{d}i', data[offset:offset + d * 4])

    return vecs


def load_sift1m(data_dir='./data/sift1m'):
    """
    Load SIFT1M dataset.
    Downloads automatically if not present (~160MB).
    """
    os.makedirs(data_dir, exist_ok=True)

    base_url = "ftp://ftp.irisa.fr/local/texmex/corpus/"
    files = {
        'sift_base.fvecs':  'xb',
        'sift_query.fvecs': 'xq',
        'sift_groundtruth.ivecs': 'gt'
    }

    local_paths = {}
    for fname, key in files.items():
        local_path = os.path.join(data_dir, fname)
        local_paths[key] = local_path

        if not os.path.exists(local_path):
            url = base_url + fname
            print(f"[DataLoader] Downloading {fname} ...")
            try:
                urllib.request.urlretrieve(url, local_path)
                print(f"[DataLoader] Downloaded {fname}")
            except Exception as e:
                print(f"[DataLoader] Could not download {fname}: {e}")
                print("[DataLoader] Falling back to synthetic data.")
                return None

    print("[DataLoader] Loading SIFT1M from disk...")
    xb = _read_fvecs(local_paths['xb'])
    xq = _read_fvecs(local_paths['xq'])
    gt = _read_ivecs(local_paths['gt'])

    print(f"[DataLoader] SIFT1M loaded. xb={xb.shape}, xq={xq.shape}, gt={gt.shape}")
    return xb, xq, gt


# ─────────────────────────────────────────────
# E-commerce Attribute Dataset
# ─────────────────────────────────────────────

def generate_ecommerce(n=50_000, d=128, nq=500, seed=42):
    """
    Synthetic e-commerce dataset with product embeddings + metadata.
    Used for attribute-filtered ANN experiments.
    """
    np.random.seed(seed)

    CATEGORIES = ['electronics', 'clothing', 'books', 'sports', 'home']
    BRANDS     = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    # Category-based cluster centers (products in same category are similar)
    cat_centers = {
        cat: np.random.randn(d).astype('float32')
        for cat in CATEGORIES
    }

    vectors    = []
    attributes = []

    for i in range(n):
        cat   = np.random.choice(CATEGORIES)
        brand = np.random.choice(BRANDS)
        price = float(np.random.choice([
            np.random.uniform(100,  500),   # budget
            np.random.uniform(500,  2000),  # mid-range
            np.random.uniform(2000, 10000)  # premium
        ]))
        rating = round(np.random.uniform(1.0, 5.0), 1)

        vec = cat_centers[cat] + 0.4 * np.random.randn(d).astype('float32')
        vectors.append(vec)
        attributes.append({
            'id':       i,
            'category': cat,
            'brand':    brand,
            'price':    price,
            'rating':   rating
        })

    xb = np.array(vectors, dtype='float32')

    # Queries
    xq   = []
    qattrs = []
    for _ in range(nq):
        cat   = np.random.choice(CATEGORIES)
        vec   = cat_centers[cat] + 0.4 * np.random.randn(d).astype('float32')
        max_p = np.random.choice([500, 2000, 5000, 10000])
        xq.append(vec)
        qattrs.append({'category': cat, 'max_price': float(max_p)})

    xq = np.array(xq, dtype='float32')

    print(f"[DataLoader] E-commerce dataset: xb={xb.shape}, xq={xq.shape}")
    return xb, attributes, xq, qattrs


# ─────────────────────────────────────────────
# BERT-like High-Dimensional Dataset
# ─────────────────────────────────────────────

def generate_bert_like(n=50_000, d=768, nq=500, seed=42):
    """
    Simulate BERT sentence embeddings (768-d).
    Uses realistic clustering structure.
    """
    np.random.seed(seed)
    print(f"[DataLoader] Generating BERT-like dataset: n={n}, d={d}")

    n_topics = 200
    topic_centers = np.random.randn(n_topics, d).astype('float32')
    # Normalize like real BERT embeddings
    norms = np.linalg.norm(topic_centers, axis=1, keepdims=True)
    topic_centers /= norms

    topic_ids = np.random.randint(0, n_topics, n)
    xb = topic_centers[topic_ids] + 0.1 * np.random.randn(n, d).astype('float32')
    xb = xb.astype('float32')

    q_topic_ids = np.random.randint(0, n_topics, nq)
    xq = topic_centers[q_topic_ids] + 0.1 * np.random.randn(nq, d).astype('float32')
    xq = xq.astype('float32')

    # Introduce dimension variance decay (simulate real BERT embeddings)
    decay = np.exp(-np.linspace(0, 3.0, d)).astype('float32')
    xb = xb * decay
    xq = xq * decay

    print(f"[DataLoader] Done. xb={xb.shape}, xq={xq.shape}")
    return xb, xq


if __name__ == "__main__":
    # Quick test
    xb, xq = generate_synthetic(n=10000, d=128, nq=100)
    print(f"Synthetic OK: xb={xb.shape}, xq={xq.shape}")

    xb2, attrs, xq2, qattrs = generate_ecommerce(n=5000, d=128, nq=50)
    print(f"Ecommerce OK: xb={xb2.shape}, attrs={len(attrs)}")

    xb3, xq3 = generate_bert_like(n=5000, d=768, nq=50)
    print(f"BERT-like OK: xb={xb3.shape}, xq={xq3.shape}")
