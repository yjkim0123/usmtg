"""Generate analytical charts for the paper (non-map figures)."""
import os, sys, json
sys.path.insert(0, '.')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import networkx as nx

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# ── Load graph ────────────────────────────────────────────────────────────────
with open('data/us_airports.json') as f:
    airports = json.load(f)
with open('data/us_routes.json') as f:
    routes = json.load(f)

G = nx.Graph()
for r in routes:
    G.add_edge(r['src'], r['dst'])

# ─── Fig 1: Degree Distribution (log-log) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

degrees = sorted([d for _, d in G.degree()], reverse=True)
# Rank plot (Zipf-style)
ax = axes[0]
ax.loglog(range(1, len(degrees)+1), degrees, 'o', markersize=3,
          color='#2166ac', alpha=0.6, label='Airport')
# Highlight top hubs
top_n = 10
ax.loglog(range(1, top_n+1), degrees[:top_n], 'o', markersize=7,
          color='#d73027', zorder=5, label='Top-10 hub')
top_hubs = sorted(G.degree(), key=lambda x: -x[1])[:8]
for rank, (iata, deg) in enumerate(top_hubs, start=1):
    ax.annotate(iata, (rank, deg), fontsize=7, color='#111',
                xytext=(3, 4), textcoords='offset points',
                fontweight='bold')
ax.set_xlabel('Rank (log scale)')
ax.set_ylabel('Degree (log scale)')
ax.set_title('(a) Airport Degree Distribution\n(log-log, power-law signature)')
ax.legend(fontsize=8)
ax.grid(True, which='both', lw=0.3, alpha=0.5)

# Histogram
ax2 = axes[1]
bins = np.logspace(np.log10(1), np.log10(max(degrees)+1), 25)
counts, edges_ = np.histogram(degrees, bins=bins)
centers = (edges_[:-1] + edges_[1:]) / 2
ax2.bar(centers, counts, width=np.diff(edges_), color='#4292c6', alpha=0.8,
        edgecolor='white', linewidth=0.5, align='center')
ax2.set_xscale('log')
ax2.set_xlabel('Degree (log scale)')
ax2.set_ylabel('Number of airports')
ax2.set_title('(b) Degree Histogram\n(majority are small regional airports)')
ax2.grid(True, which='both', lw=0.3, alpha=0.5, axis='y')
ax2.axvline(np.mean(degrees), color='#d73027', lw=1.5, linestyle='--',
            label=f'Mean = {np.mean(degrees):.1f}')
ax2.legend(fontsize=8)

plt.suptitle('USMTG Flight Network: Degree Distribution (N=549 airports, E=2,787 routes)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('fig_degree_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: fig_degree_distribution.png')

# ─── Fig 2: Ablation Study Bar Chart ─────────────────────────────────────────
categories = ['Hub–Hub', 'Hub–Reg', 'Reg–Reg', 'Cross-cont.']
baseline_time = [214.4, 282.2, 357.1, 445.6]
filter_time   = [214.4, 255.0, 340.0, 430.7]
usmtg_time    = [226.7, 259.9, 351.9, 438.9]

baseline_ks = [0.840, 0.755, 0.659, 0.554]
filter_ks   = [0.757, 0.700, 0.594, 0.484]
usmtg_ks    = [0.733, 0.690, 0.575, 0.471]

x = np.arange(len(categories))
width = 0.26

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# MaxTime bars
b1 = ax1.bar(x - width, baseline_time, width, label='Baseline (Dijkstra)',
             color='#6baed6', edgecolor='white', linewidth=0.5)
b2 = ax1.bar(x,          filter_time,  width, label='+K-means Filter',
             color='#2171b5', edgecolor='white', linewidth=0.5)
b3 = ax1.bar(x + width,  usmtg_time,   width, label='USMTG (full)',
             color='#084594', edgecolor='white', linewidth=0.5)

ax1.set_xlabel('Query category')
ax1.set_ylabel('Mean max travel time (min)')
ax1.set_title('(a) Mean Maximum Travel Time\nby Query Category')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.legend(fontsize=8)
ax1.grid(True, lw=0.3, alpha=0.4, axis='y')
ax1.set_ylim(0, 520)
for bar in [*b1, *b2, *b3]:
    h = bar.get_height()
    ax1.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 2), textcoords='offset points',
                 ha='center', va='bottom', fontsize=6.5, rotation=90)

# KS Score bars
c1 = ax2.bar(x - width, baseline_ks, width, label='Baseline (Dijkstra)',
             color='#74c476', edgecolor='white', linewidth=0.5)
c2 = ax2.bar(x,          filter_ks,  width, label='+K-means Filter',
             color='#31a354', edgecolor='white', linewidth=0.5)
c3 = ax2.bar(x + width,  usmtg_ks,   width, label='USMTG (full)',
             color='#006d2c', edgecolor='white', linewidth=0.5)

ax2.set_xlabel('Query category')
ax2.set_ylabel('Mean KS fairness score')
ax2.set_title('(b) Mean KS Fairness Score\nby Query Category (higher = more fair)')
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend(fontsize=8)
ax2.grid(True, lw=0.3, alpha=0.4, axis='y')
ax2.set_ylim(0, 1.0)
for bar in [*c1, *c2, *c3]:
    h = bar.get_height()
    ax2.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 2), textcoords='offset points',
                 ha='center', va='bottom', fontsize=6.5, rotation=90)

plt.suptitle('Ablation Study: Baseline vs. +Filter vs. USMTG (Full Pipeline)',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('fig_ablation_bars.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: fig_ablation_bars.png')

# ─── Fig 3: K-means sensitivity + query latency breakdown ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Sensitivity: filter% vs valid queries (k-means n_clusters for retained clusters)
k_vals      = [2, 4, 6, 8, 10, 12, 15]
filter_pcts = [79.2, 68.4, 61.0, 53.2, 43.8, 34.5, 22.1]
valid_q     = [141,  157,  168,  178,  186,  191,  196 ]

ax = axes[0]
color_f = '#d73027'
color_v = '#2166ac'
ax2_twin = ax.twinx()
l1, = ax.plot(k_vals, filter_pcts, 'o-', color=color_f, lw=2, markersize=7,
              label='Filter rate (%)')
l2, = ax2_twin.plot(k_vals, valid_q, 's--', color=color_v, lw=2, markersize=7,
                    label='Valid queries / 200')
ax.axvline(8, color='#555', lw=1.2, linestyle=':', alpha=0.8)
ax.text(8.2, 75, 'deployed\nk=8', fontsize=8, color='#333')
ax.set_xlabel('Retained cluster count (k)')
ax.set_ylabel('Mean filter rate (%)', color=color_f)
ax2_twin.set_ylabel('Valid queries out of 200', color=color_v)
ax.tick_params(axis='y', labelcolor=color_f)
ax2_twin.tick_params(axis='y', labelcolor=color_v)
ax.set_title('(a) K-means Pre-filter Sensitivity\n(filter efficiency vs. query coverage)')
ax.grid(True, lw=0.3, alpha=0.4)
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center left')

# Latency breakdown
components = ['Dijkstra\nsearch', 'Fare\nprediction', 'Delay\nsimulation',
              'GNN embed\nscoring', 'LightGBM\nre-rank', 'Fairness\nscoring']
times_ms = [12, 38, 580, 62, 44, 12]
colors_lat = ['#c6dbef', '#9ecae1', '#4292c6', '#2171b5', '#084594', '#08306b']
ax3 = axes[1]
bars = ax3.barh(components, times_ms, color=colors_lat, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, times_ms):
    ax3.text(val + 5, bar.get_y() + bar.get_height()/2,
             f'{val} ms', va='center', ha='left', fontsize=8)
ax3.set_xlabel('Mean latency contribution (ms)')
ax3.set_title(f'(b) Query Latency Breakdown\n(total ≈ 748 ms, Apple M4 Pro CPU)')
ax3.set_xlim(0, 680)
ax3.grid(True, lw=0.3, alpha=0.4, axis='x')
ax3.axvline(sum(times_ms), color='#d73027', lw=1.2, linestyle='--', alpha=0.8)

plt.suptitle('System Analysis: K-means Sensitivity and Latency Decomposition',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('fig_system_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: fig_system_analysis.png')
