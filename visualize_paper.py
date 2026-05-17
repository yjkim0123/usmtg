"""
Paper-quality figures: white background, high contrast, suitable for print.
Regenerates all meeting-point figures and the hub map.
"""
import json, math, os, sys
sys.path.insert(0, '.')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
from meeting_finder_v2 import load, resolve_location, find_meeting_v2, fmt_time

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})

LAND_COLOR   = '#f5f5f0'
OCEAN_COLOR  = '#dce8f5'
ROUTE_COLOR  = '#b0c4de'
SMALL_COLOR  = '#9ecae1'
REGIONAL_COLOR = '#4292c6'
HUB_COLOR    = '#08519c'
ORIGIN_A_COLOR = '#d73027'
ORIGIN_B_COLOR = '#4575b4'
BEST_COLOR   = '#f59b00'
ALT_COLOR    = '#74c476'


def draw_us_outline(ax):
    """Draw a simple lat/lon box for CONUS context."""
    ax.set_facecolor(OCEAN_COLOR)
    ax.axhspan(23, 50, xmin=0, xmax=1, color=LAND_COLOR, zorder=0)
    ax.set_xlim(-128, -65)
    ax.set_ylim(23, 50)
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)


def plot_hub_map(airports, graph, save_path='fig_us_hub_map.png'):
    degree = defaultdict(int)
    for src, nbrs in graph.items():
        degree[src] += len(nbrs)

    fig, ax = plt.subplots(figsize=(14, 8))
    draw_us_outline(ax)

    # Background routes (top hubs only for readability)
    top_iatas = {iata for iata, _ in sorted(degree.items(), key=lambda x: -x[1])[:80]}
    plotted = set()
    for src, nbrs in graph.items():
        if src not in top_iatas: continue
        a = airports.get(src)
        if not a: continue
        for dst, *_ in nbrs[:4]:
            if dst not in airports: continue
            key = tuple(sorted([src, dst]))
            if key in plotted: continue
            plotted.add(key)
            b = airports[dst]
            if not (-128 < a['lon'] < -65 and 23 < a['lat'] < 50): continue
            if not (-128 < b['lon'] < -65 and 23 < b['lat'] < 50): continue
            ax.plot([a['lon'], b['lon']], [a['lat'], b['lat']],
                    color=ROUTE_COLOR, lw=0.4, alpha=0.6, zorder=1)

    iata_list = list(airports.keys())
    lons = [airports[i]['lon'] for i in iata_list]
    lats = [airports[i]['lat'] for i in iata_list]
    degs = [degree.get(i, 1) for i in iata_list]

    sizes  = [max(8, (d / max(degs)) * 300) for d in degs]
    colors = [HUB_COLOR if d > 150 else REGIONAL_COLOR if d > 50 else SMALL_COLOR
              for d in degs]

    ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.85,
               linewidths=0.3, edgecolors='white', zorder=3)

    for iata, deg in sorted(degree.items(), key=lambda x: -x[1])[:15]:
        ap = airports.get(iata)
        if not ap or not (-128 < ap['lon'] < -65 and 23 < ap['lat'] < 50): continue
        ax.annotate(iata, (ap['lon'], ap['lat']),
                    fontsize=7, color='#111', fontweight='bold',
                    xytext=(3, 3), textcoords='offset points', zorder=5)

    patches = [
        mpatches.Patch(color=HUB_COLOR,      label='Major hub (>150 routes)'),
        mpatches.Patch(color=REGIONAL_COLOR,  label='Regional hub (50–150)'),
        mpatches.Patch(color=SMALL_COLOR,     label='Small airport (<50)'),
    ]
    ax.legend(handles=patches, loc='lower left', fontsize=8,
              framealpha=0.9, edgecolor='#ccc')

    n_routes = sum(len(v) for v in graph.values()) // 2
    ax.set_title(f'USMTG: US Domestic Flight Network\n'
                 f'{len(airports)} airports · {n_routes} direct routes',
                 fontsize=12, pad=10)
    ax.grid(True, lw=0.3, alpha=0.4, color='#aaa')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {save_path}')


def plot_meeting(loc_a, name_a, loc_b, name_b,
                 best, top5, pareto, airports,
                 save_path='fig_meeting_result.png'):

    fig = plt.figure(figsize=(14, 8))
    ax   = fig.add_axes([0.0, 0.0, 0.72, 1.0])
    ax_p = fig.add_axes([0.75, 0.15, 0.23, 0.68])

    # Map background
    draw_us_outline(ax)

    # Background airports
    bg_lons = [ap['lon'] for ap in airports.values()
               if -128 < ap['lon'] < -65 and 23 < ap['lat'] < 50]
    bg_lats = [ap['lat'] for ap in airports.values()
               if -128 < ap['lon'] < -65 and 23 < ap['lat'] < 50]
    ax.scatter(bg_lons, bg_lats, s=3, c='#aaaaaa', alpha=0.35, zorder=1)

    # Pareto alternatives
    pareto_iatas = {r['airport'] for r in pareto}
    for r in pareto:
        if r['airport'] == best['airport']: continue
        ax.scatter(r['lon'], r['lat'], s=100, c=ALT_COLOR, zorder=4,
                   marker='D', edgecolors='#333', linewidths=0.6)
        ax.annotate(f"{r['airport']}", (r['lon'], r['lat']),
                    color='#333', fontsize=7,
                    xytext=(5, 4), textcoords='offset points', zorder=5)

    # Other top results
    for r in top5[1:5]:
        if r['airport'] in pareto_iatas or r['airport'] == best['airport']: continue
        ax.scatter(r['lon'], r['lat'], s=35, c='#bbbbbb', zorder=3,
                   marker='o', edgecolors='#888', linewidths=0.4)

    # Best meeting point
    blon, blat = best['lon'], best['lat']
    ax.scatter(blon, blat, s=350, c=BEST_COLOR, zorder=6, marker='*',
               edgecolors='#333', linewidths=0.8)

    label = (f"★ {best['airport']} — {best['city']}\n"
             f"A: {fmt_time(best['time_a_min'])}  ${best['fare_a_usd']}\n"
             f"B: {fmt_time(best['time_b_min'])}  ${best['fare_b_usd']}\n"
             f"Max: {fmt_time(best['max_time_min'])}  Total: ${best['total_cost_usd']}")
    ax.annotate(label, (blon, blat), color='#111', fontsize=8,
                fontweight='bold', xytext=(12, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.4', fc='#fffde7',
                          ec=BEST_COLOR, alpha=0.95, lw=1.5),
                zorder=8)

    # Origins
    ax.scatter(loc_a[1], loc_a[0], s=180, c=ORIGIN_A_COLOR, zorder=5,
               marker='o', edgecolors='white', linewidths=1.0)
    ax.annotate(name_a, (loc_a[1], loc_a[0]), color=ORIGIN_A_COLOR,
                fontsize=9, fontweight='bold',
                xytext=(-5, 10), textcoords='offset points')

    ax.scatter(loc_b[1], loc_b[0], s=180, c=ORIGIN_B_COLOR, zorder=5,
               marker='o', edgecolors='white', linewidths=1.0)
    ax.annotate(name_b, (loc_b[1], loc_b[0]), color=ORIGIN_B_COLOR,
                fontsize=9, fontweight='bold',
                xytext=(-5, 10), textcoords='offset points')

    # Travel lines
    ax.plot([loc_a[1], blon], [loc_a[0], blat],
            '--', c=ORIGIN_A_COLOR, lw=2.0, alpha=0.8, zorder=4)
    ax.plot([loc_b[1], blon], [loc_b[0], blat],
            '--', c=ORIGIN_B_COLOR, lw=2.0, alpha=0.8, zorder=4)

    ax.set_title(
        f'USMTG Meeting Point: {name_a} ↔ {name_b}\n'
        f'Optimal: {best["city"]} ({best["airport"]})  |  '
        f'Max {fmt_time(best["max_time_min"])}  |  '
        f'Total ${best["total_cost_usd"]}  |  '
        f'Imbalance {best["imbalance_min"]} min',
        fontsize=10, pad=8)
    ax.grid(True, lw=0.3, alpha=0.3, color='#aaa')

    # Legend
    leg_items = [
        mpatches.Patch(color=ORIGIN_A_COLOR, label=f'Origin A: {name_a}'),
        mpatches.Patch(color=ORIGIN_B_COLOR, label=f'Origin B: {name_b}'),
        plt.scatter([], [], s=120, c=BEST_COLOR, marker='*', label='Best meeting point'),
        plt.scatter([], [], s=80,  c=ALT_COLOR,  marker='D', label='Pareto-optimal alt.'),
    ]
    ax.legend(handles=leg_items, loc='lower left', fontsize=8,
              framealpha=0.9, edgecolor='#ccc')

    # Pareto inset
    ax_p.set_facecolor('#fafafa')
    for spine in ax_p.spines.values():
        spine.set_edgecolor('#cccccc')
    ax_p.grid(True, lw=0.4, alpha=0.5, color='#ddd')

    if pareto:
        times  = [r['max_time_min'] / 60 for r in pareto]
        costs  = [r['total_cost_usd'] for r in pareto]
        labels = [r['airport'] for r in pareto]

        ax_p.plot(times, costs, 'o-', color='#2c7bb6', lw=1.5,
                  markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        btime = best['max_time_min'] / 60
        bcost = best['total_cost_usd']
        ax_p.scatter([btime], [bcost], s=130, c=BEST_COLOR, marker='*',
                     zorder=5, edgecolors='#333', linewidths=0.8)
        for t, c, lbl in zip(times, costs, labels):
            ax_p.annotate(lbl, (t, c), color='#333', fontsize=7,
                          xytext=(4, 4), textcoords='offset points')
        ax_p.fill_betweenx(costs, [min(times)] * len(costs), times,
                           alpha=0.08, color='#2c7bb6')
        ax_p.set_xlabel('Max travel time (h)', fontsize=8)
        ax_p.set_ylabel('Total cost ($)',       fontsize=8)
        ax_p.set_title('Pareto Front\n(time vs cost)', fontsize=9, pad=6)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    airports, graph = load()

    print('Generating hub map...')
    plot_hub_map(airports, graph)

    pairs = [
        ('New York',  'Los Angeles'),
        ('Seattle',   'Miami'),
        ('Boston',    'Dallas'),
        ('Chicago',   'San Francisco'),
        ('Honolulu',  'New York'),
        ('Anchorage', 'Miami'),
    ]

    for city_a, city_b in pairs:
        loc_a, name_a = resolve_location(city_a, airports)
        loc_b, name_b = resolve_location(city_b, airports)
        if not loc_a or not loc_b:
            print(f'Cannot resolve: {city_a} / {city_b}'); continue

        best, top, pareto = find_meeting_v2(*loc_a, *loc_b, airports, graph)
        if not best:
            print(f'No result: {city_a} <-> {city_b}'); continue

        fname = (f'fig_meeting_'
                 f'{city_a.lower().replace(" ", "_")}_'
                 f'{city_b.lower().replace(" ", "_")}.png')
        plot_meeting(loc_a, name_a, loc_b, name_b,
                     best, top, pareto, airports, fname)
