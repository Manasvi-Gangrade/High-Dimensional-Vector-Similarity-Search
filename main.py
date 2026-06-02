"""
main.py
-------
Entry point for the ANN paper implementation.

Run this file to execute the complete pipeline:
  1. Generate/load datasets
  2. Train all models
  3. Run all experiments
  4. Generate all figures
  5. Print summary

Usage:
  python main.py              # full run
  python main.py --quick      # small n, fast run for testing
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from evaluate    import (exp1_reconstruction, exp2_recall_qps,
                          exp3_attribute_filtering, exp4_high_dimensional,
                          exp5_ablation_rerank, exp6_dynamic_updates, print_banner)
from plot_results import generate_all_plots


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(quick=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, 'results', 'benchmark_log.txt')
    logger = Logger(log_path)
    sys.stdout = logger
    print("\n" + "█"*60)
    print("  EFFICIENT ANN SEARCH — PAPER IMPLEMENTATION")
    print("  Manasvi Gangrade | IIST Indore")
    print("  Product Quantization + HNSW + Attribute Filtering")
    print("█"*60)

    if quick:
        print("\n[Mode] QUICK RUN (small dataset)")
        cfg = dict(n_basic=5000, n_ecom=3000, n_bert=2000, nq=100)
    else:
        print("\n[Mode] FULL RUN")
        cfg = dict(n_basic=25000, n_ecom=12000, n_bert=8000, nq=300)

    all_results = {}

    # ── Experiment 1: PQ Reconstruction ──
    all_results['reconstruction'] = exp1_reconstruction(
        n=cfg['n_basic'], d=128
    )

    # ── Experiment 2: Recall vs QPS ──
    all_results['recall_qps'] = exp2_recall_qps(
        n=cfg['n_basic'], d=128, k=10
    )

    # ── Experiment 3: Attribute Filtering ──
    all_results['filtering'] = exp3_attribute_filtering(
        n=cfg['n_ecom'], d=128
    )

    # ── Experiment 4: High-Dim BERT ──
    all_results['high_dim'] = exp4_high_dimensional(
        n=cfg['n_bert'], d=768
    )

    # ── Experiment 5: Ablation ──
    all_results['ablation'] = exp5_ablation_rerank(
        n=cfg['n_basic'], d=128
    )

    # ── Experiment 6: Dynamic Updates ──
    all_results['dynamic_updates'] = exp6_dynamic_updates(
        n=cfg['n_basic'], d=128
    )

    # ── Generate Plots ──
    print_banner("GENERATING FIGURES")
    generate_all_plots(all_results)

    # ── Final Summary ──
    print_banner("SUMMARY — KEY RESULTS FOR PAPER")

    rq = all_results['recall_qps']
    for r in rq:
        if 'OURS' in r['name']:
            print(f"\n  Our method (Hybrid Learned-PQ + HNSW):")
            print(f"    Recall@10 : {r['recall']:.3f}")
            print(f"    QPS       : {r['qps']:.0f}")
            print(f"    Memory    : {r['mem_bytes']//1024//1024:.1f} MB")

    if 'filtering' in all_results:
        f = all_results['filtering']
        print(f"\n  Attribute Filtering Speedup: {f['speedup']:.1f}x")

    print("\n  Figures saved to: ann_paper/results/")
    print("  Use these numbers directly in your paper tables!\n")

    # ── Save Raw Results to JSON ──
    import json
    import numpy as np

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    raw_results_path = os.path.join(base_dir, 'results', 'raw_results.json')
    try:
        with open(raw_results_path, 'w') as f:
            json.dump(convert_numpy(all_results), f, indent=4)
        print(f"  Raw JSON results saved to: {raw_results_path}\n")
    except Exception as e:
        print(f"  Error saving raw results to JSON: {e}\n")

    # Restore stdout
    sys.stdout = sys.__stdout__
    logger.log.close()

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick run with small dataset')
    args = parser.parse_args()

    main(quick=args.quick)
