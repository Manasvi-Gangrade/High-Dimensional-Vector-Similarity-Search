"""
plot_results.py
---------------
Generate publication-quality figures for the paper.
Saves all plots to results/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

STYLE = {
    'Exact (Brute Force)':         ('gray',   'o',  '-',  2.0),
    'HNSW':                        ('blue',   's',  '--', 2.0),
    'IVF-PQ (FAISS)':              ('orange', '^',  '--', 2.0),
    'Hybrid Std-PQ + HNSW':        ('green',  'D',  '-',  2.0),
    'Hybrid Learned-PQ+HNSW [OURS]': ('red', '*',  '-',  3.0),
}


def plot_recall_qps(results, save_path='results/fig1_recall_qps.png'):
    """Figure 1: Recall@10 vs QPS (main result)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        name = r['name']
        color, marker, ls, lw = STYLE.get(name, ('black', 'o', '-', 1.5))
        ax.scatter(r['qps'], r['recall'],
                   c=color, marker=marker, s=150, zorder=5,
                   label=name, edgecolors='white', linewidths=0.5)

    ax.axhline(0.95, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 10,
            0.951, '95% recall target', fontsize=8, color='gray')

    ax.set_xlabel('QPS (Queries Per Second)', fontsize=13)
    ax.set_ylabel('Recall@10', fontsize=13)
    ax.set_title('Recall@10 vs Throughput\n(Higher-right is better)', fontsize=14)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_reconstruction_error(m_values, std_errors, lpq_errors,
                              save_path='results/fig2_reconstruction.png'):
    """Figure 2: Reconstruction error comparison."""
    x      = np.arange(len(m_values))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, std_errors, width,
                   label='Standard PQ', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, lpq_errors, width,
                   label='Learned PQ (Ours)', color='tomato', alpha=0.8)

    # Improvement annotations
    for i, (se, le) in enumerate(zip(std_errors, lpq_errors)):
        imp = (se - le) / se * 100
        ax.annotate(f'−{imp:.1f}%',
                    xy=(x[i] + width/2, le),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, color='darkred', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'm={m}' for m in m_values], fontsize=11)
    ax.set_xlabel('Number of Subvectors (m)', fontsize=12)
    ax.set_ylabel('Mean Reconstruction Error (L2)', fontsize=12)
    ax.set_title('PQ Reconstruction Error: Standard vs Learned\n'
                 '(Lower is better)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_filtering_speedup(results, save_path='results/fig3_filtering.png'):
    """Figure 3: Attribute filtering strategy comparison."""
    labels   = ['Post-Filter\n(Naive)', 'Sub-Index\n(Ours)']
    times    = [results['post_filter_ms'], results['sub_index_ms']]
    colors   = ['steelblue', 'tomato']

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, times, color=colors, alpha=0.85, width=0.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{t:.2f} ms', ha='center', fontsize=12, fontweight='bold')

    ax.annotate(
        f"Speedup: {results['speedup']:.1f}×",
        xy=(0.5, 0.85), xycoords='axes fraction',
        ha='center', fontsize=14, fontweight='bold', color='darkgreen',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4)
    )

    ax.set_ylabel('Mean Query Latency (ms)', fontsize=12)
    ax.set_title('Attribute Filtering: Strategy Comparison\n'
                 '(Lower is better)', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_ablation_rerank(rf_values, recalls, qps_values,
                         save_path='results/fig4_ablation.png'):
    """Figure 4: Rerank factor ablation."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1, color2 = 'tomato', 'steelblue'

    ax1.plot(rf_values, recalls, 'o-', color=color1, linewidth=2.5,
             markersize=8, label='Recall@10')
    ax1.axhline(0.95, color=color1, linestyle=':', alpha=0.5)
    ax1.set_xlabel('Rerank Factor (α)', fontsize=12)
    ax1.set_ylabel('Recall@10', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(rf_values, qps_values, 's--', color=color2, linewidth=2.5,
             markersize=8, label='QPS')
    ax2.set_ylabel('QPS', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center right', fontsize=11)

    plt.title('Ablation: Rerank Factor vs Recall–Speed Trade-off', fontsize=13)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def plot_memory_comparison(results, save_path='results/fig5_memory.png'):
    """Figure 5: Memory usage comparison."""
    names  = [r['name'].replace(' [OURS]', '\n[OURS]') for r in results]
    mems   = [r['mem_bytes'] / 1e6 for r in results]
    colors = ['gray', 'steelblue', 'orange', 'green', 'tomato'][:len(names)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, mems, color=colors, alpha=0.85)

    for bar, m in zip(bars, mems):
        ax.text(m + 0.5, bar.get_y() + bar.get_height()/2,
                f'{m:.1f} MB', va='center', fontsize=10)

    ax.set_xlabel('Total Memory (MB)', fontsize=12)
    ax.set_title('Memory Consumption Comparison\n(Lower is better)', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved: {save_path}")


def generate_all_plots(eval_results):
    """Generate all paper figures from evaluation results."""
    print("\n[Plots] Generating all figures...")

    # Fig 1: Recall vs QPS
    if 'recall_qps' in eval_results:
        plot_recall_qps(eval_results['recall_qps'])
        plot_memory_comparison(eval_results['recall_qps'])

    # Fig 2: Reconstruction error
    if 'reconstruction' in eval_results:
        rows = eval_results['reconstruction']
        m_vals, std_errs, lpq_errs = [], [], []
        for i in range(0, len(rows), 2):
            m = int(rows[i][0].split('=')[1].split(')')[0])
            m_vals.append(m)
            std_errs.append(float(rows[i][1]))
            lpq_errs.append(float(rows[i+1][1]))
        plot_reconstruction_error(m_vals, std_errs, lpq_errs)

    # Fig 3: Filtering speedup
    if 'filtering' in eval_results:
        plot_filtering_speedup(eval_results['filtering'])

    # Fig 4: Ablation
    if 'ablation' in eval_results:
        abl = eval_results['ablation']
        rf_vals = [int(r[0].split('=')[1]) for r in abl]
        recalls  = [float(r[1]) for r in abl]
        qps_vals = [float(r[2]) for r in abl]
        plot_ablation_rerank(rf_vals, recalls, qps_vals)

    print("[Plots] All figures saved to results/")


if __name__ == "__main__":
    # Quick demo with fake data
    fake_results = [
        {'name': 'Exact (Brute Force)',           'recall': 1.00, 'qps': 120,  'mem_bytes': 30e6},
        {'name': 'HNSW',                          'recall': 0.96, 'qps': 4500, 'mem_bytes': 45e6},
        {'name': 'IVF-PQ (FAISS)',                'recall': 0.88, 'qps': 3200, 'mem_bytes': 10e6},
        {'name': 'Hybrid Std-PQ + HNSW',          'recall': 0.92, 'qps': 3800, 'mem_bytes': 38e6},
        {'name': 'Hybrid Learned-PQ+HNSW [OURS]', 'recall': 0.95, 'qps': 4100, 'mem_bytes': 36e6},
    ]
    plot_recall_qps(fake_results)
    plot_memory_comparison(fake_results)
    print("Demo plots saved.")
