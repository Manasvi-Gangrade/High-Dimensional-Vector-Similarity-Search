"""
attribute_filter.py
-------------------
Multi-Index Attribute Filtering — second major contribution of the paper.

Problem: Real-world queries need BOTH vector similarity AND metadata filters.
  e.g. "Find products similar to this image, in Electronics, under ₹5000"

Naive approaches fail:
  Post-filter: search → filter → low recall when filter is strict
  Pre-filter:  filter → search over tiny subgraph → slow, poor quality

Our Solution: Build separate HNSW sub-index per attribute combination.
  - Route query to correct sub-index
  - Search within (never filters, full recall maintained)
  - 5-10x faster than post-filtering on selective queries
"""

import numpy as np
import faiss
import time
from collections import defaultdict


class AttributeFilteredIndex:
    """
    Multi-Index ANN with attribute filtering.

    Supports:
      - Categorical filters  (category == 'electronics')
      - Range filters        (price <= 5000)
      - Combined filters     (category == 'X' AND price <= Y)

    Architecture:
      ┌──────────────────────────────────────────────────┐
      │  Main Index (all vectors)                        │
      │  ┌──────────────┐  ┌──────────────┐             │
      │  │ Sub-Index:   │  │ Sub-Index:   │  ...        │
      │  │ electronics  │  │ clothing     │             │
      │  └──────────────┘  └──────────────┘             │
      │  ┌──────────────┐  ┌──────────────┐             │
      │  │ Sub-Index:   │  │ Sub-Index:   │  ...        │
      │  │ price: 0-500 │  │ price:500-2k │             │
      │  └──────────────┘  └──────────────┘             │
      └──────────────────────────────────────────────────┘
    """

    PRICE_BUCKETS = [
        (0,     500,   'budget'),
        (500,   2000,  'mid'),
        (2000,  5000,  'premium'),
        (5000,  float('inf'), 'luxury')
    ]

    def __init__(self, d, M=16, ef_construction=100):
        self.d  = d
        self.M  = M
        self.ef_construction = ef_construction

        # Global index (all vectors, no filter)
        self.global_index  = faiss.IndexHNSWFlat(d, M)
        self.global_index.hnsw.efConstruction = ef_construction

        # Sub-indices: key → IndexHNSWFlat
        self.cat_indices   = {}   # category → index
        self.price_indices = {}   # price_bucket → index

        # Metadata storage
        self.all_vectors = []     # list of float32 arrays
        self.all_attrs   = []     # list of dicts

        # Mapping: sub-index local id → global id
        self.cat_id_map   = defaultdict(list)   # cat → [global_ids]
        self.price_id_map = defaultdict(list)   # bucket → [global_ids]
        self.deleted_ids  = set()

    def _get_price_bucket(self, price):
        for lo, hi, name in self.PRICE_BUCKETS:
            if lo <= price < hi:
                return name
        return 'luxury'

    def add(self, vectors, attributes):
        """
        Add vectors with attributes to all relevant indices.

        vectors    : np.array (n, d) float32
        attributes : list of dicts with keys: category, price, [brand, rating, ...]
        """
        assert len(vectors) == len(attributes)
        n = len(vectors)
        print(f"\n[AttrIndex] Adding {n} vectors with attributes...")
        t0 = time.time()

        # Global index
        self.global_index.add(vectors)

        start_global_id = len(self.all_vectors)

        for i, (vec, attr) in enumerate(zip(vectors, attributes)):
            global_id = start_global_id + i
            self.all_vectors.append(vec)
            self.all_attrs.append(attr)

            cat    = attr.get('category', 'unknown')
            price  = attr.get('price', 0)
            bucket = self._get_price_bucket(price)

            # Add to category sub-index
            if cat not in self.cat_indices:
                idx = faiss.IndexHNSWFlat(self.d, self.M)
                idx.hnsw.efConstruction = self.ef_construction
                self.cat_indices[cat] = idx
            self.cat_indices[cat].add(vec.reshape(1, -1))
            self.cat_id_map[cat].append(global_id)

            # Add to price bucket sub-index
            if bucket not in self.price_indices:
                idx = faiss.IndexHNSWFlat(self.d, self.M)
                idx.hnsw.efConstruction = self.ef_construction
                self.price_indices[bucket] = idx
            self.price_indices[bucket].add(vec.reshape(1, -1))
            self.price_id_map[bucket].append(global_id)

        build_time = time.time() - t0
        n_cats = len(self.cat_indices)
        print(f"[AttrIndex] Done in {build_time:.2f}s")
        print(f"[AttrIndex] Categories: {list(self.cat_indices.keys())}")
        print(f"[AttrIndex] Price buckets: {list(self.price_indices.keys())}")
        print(f"[AttrIndex] Total sub-indices: {n_cats + len(self.price_indices)}")

    def delete(self, global_id):
        """
        Mark a vector as deleted (tombstone).
        """
        self.deleted_ids.add(global_id)

    def search(self, query, k=10, category=None, max_price=None,
               ef_search=50, strategy='auto'):
        """
        Hybrid filtered search.

        strategy options:
          'sub_index'  : use category sub-index directly (fast)
          'post_filter': search global → filter results (naive baseline)
          'auto'       : choose best strategy automatically
        """
        if len(query.shape) == 1:
            query = query.reshape(1, -1)

        if strategy == 'auto':
            strategy = 'sub_index' if category else 'post_filter'

        t0 = time.time()

        if strategy == 'sub_index' and category and category in self.cat_indices:
            results = self._search_sub_index(
                query, k, category, max_price, ef_search
            )
        else:
            results = self._search_post_filter(
                query, k, category, max_price, ef_search
            )

        search_time = (time.time() - t0) * 1000
        return results, search_time

    def _search_sub_index(self, query, k, category, max_price, ef_search):
        """Fast path: search within category sub-index."""
        cat_index   = self.cat_indices[category]
        global_ids  = self.cat_id_map[category]

        # Fetch more candidates to account for price filtering
        n_fetch = min(k * 5, cat_index.ntotal)
        if n_fetch == 0:
            return []

        cat_index.hnsw.efSearch = ef_search
        D, I = cat_index.search(query, n_fetch)

        results = []
        for dist, local_id in zip(D[0], I[0]):
            if local_id < 0 or local_id >= len(global_ids):
                continue
            gid  = global_ids[local_id]
            if gid in self.deleted_ids:
                continue
            attr = self.all_attrs[gid]

            # Apply price filter if requested
            if max_price is not None and attr.get('price', 0) > max_price:
                continue

            results.append({
                'global_id': gid,
                'distance':  float(dist),
                'attr':      attr
            })
            if len(results) >= k:
                break

        return results

    def _search_post_filter(self, query, k, category, max_price, ef_search):
        """Slow path: search globally then filter (naive baseline)."""
        n_fetch = min(k * 20, self.global_index.ntotal)
        self.global_index.hnsw.efSearch = ef_search
        D, I = self.global_index.search(query, n_fetch)

        results = []
        for dist, gid in zip(D[0], I[0]):
            if gid < 0 or gid >= len(self.all_attrs):
                continue
            if gid in self.deleted_ids:
                continue
            attr = self.all_attrs[gid]

            if category and attr.get('category') != category:
                continue
            if max_price is not None and attr.get('price', 0) > max_price:
                continue

            results.append({
                'global_id': int(gid),
                'distance':  float(dist),
                'attr':      attr
            })
            if len(results) >= k:
                break

        return results

    def stats(self):
        """Print index statistics."""
        n_total = len(self.all_vectors)
        cat_counts = {cat: idx.ntotal for cat, idx in self.cat_indices.items()}
        price_counts = {b: idx.ntotal for b, idx in self.price_indices.items()}

        print(f"\n[AttrIndex] Statistics:")
        print(f"  Total vectors: {n_total}")
        print(f"  Category distribution: {cat_counts}")
        print(f"  Price bucket distribution: {price_counts}")
        return {'total': n_total, 'categories': cat_counts, 'prices': price_counts}


# ──────────────────────────────────────────────────────────────────────────────


def compare_filter_strategies(attr_index, xq_queries, query_attrs, k=10):
    """
    Compare sub-index strategy vs post-filter strategy.
    Shows speedup of paper's approach.
    """
    print("\n" + "="*60)
    print("FILTER STRATEGY COMPARISON")
    print("="*60)

    sub_times  = []
    post_times = []

    for q, qa in zip(xq_queries, query_attrs):
        cat = qa.get('category')
        mp  = qa.get('max_price')

        _, t_sub  = attr_index.search(q, k=k, category=cat,
                                      max_price=mp, strategy='sub_index')
        _, t_post = attr_index.search(q, k=k, category=cat,
                                      max_price=mp, strategy='post_filter')

        sub_times.append(t_sub)
        post_times.append(t_post)

    mean_sub  = np.mean(sub_times)
    mean_post = np.mean(post_times)
    speedup   = mean_post / mean_sub if mean_sub > 0 else float('inf')

    print(f"Sub-Index Strategy:  {mean_sub:.2f} ms / query")
    print(f"Post-Filter Strategy:{mean_post:.2f} ms / query")
    print(f"Speedup:             {speedup:.1f}x")
    print(f"(Paper claims 5-10x speedup on selective queries)")

    return {
        'sub_index_ms':   mean_sub,
        'post_filter_ms': mean_post,
        'speedup':        speedup
    }


if __name__ == "__main__":
    from data_loader import generate_ecommerce

    d = 128
    xb, attrs, xq, qattrs = generate_ecommerce(n=10000, d=d, nq=100)

    idx = AttributeFilteredIndex(d=d, M=16)
    idx.add(xb, attrs)
    idx.stats()

    # Test search
    results, ms = idx.search(xq[0], k=5,
                             category='electronics',
                             max_price=2000)
    print(f"\nSample query results ({ms:.2f}ms):")
    for r in results[:3]:
        print(f"  ID={r['global_id']}, dist={r['distance']:.3f}, "
              f"cat={r['attr']['category']}, price={r['attr']['price']:.0f}")

    compare_filter_strategies(idx, xq[:50], qattrs[:50], k=10)
