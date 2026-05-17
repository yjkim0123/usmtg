"""
USMTG Visualization: US airport hub map + meeting point results
"""
import json, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
from meeting_finder_us import (load, resolve_location, find_meeting_airport,
                                haversine_km)

DATA_DIR = "data"

def plot_hub_map(airports, graph, save_path="fig_us_hub_map.png"):
    """Plot all US airports sized by degree (# direct routes)."""
    degree = defaultdict(int)
    for src, neighbors in graph.items():
        degree[src] += len(neighbors)

    lons = [ap["lon"] for ap in airports.values()]
    lats = [ap["lat"] for ap in airports.values()]
    degs = [degree.get(iata, 1) for iata in airports.keys()]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0a0a1a")

    # US boundary (approximate)
    ax.set_xlim(-128, -65)
    ax.set_ylim(23, 50)

    # Plot routes (sample top-200 by degree for readability)
    top_airports = sorted(degree.items(), key=lambda x: -x[1])[:80]
    top_iatas = {iata for iata, _ in top_airports}
    plotted_edges = set()
    for src, neighbors in graph.items():
        if src not in top_iatas:
            continue
        a = airports.get(src)
        if not a:
            continue
        for dst, _ in neighbors[:5]:  # limit edges per node
            if dst not in airports:
                continue
            key = tuple(sorted([src, dst]))
            if key in plotted_edges:
                continue
            plotted_edges.add(key)
            b = airports[dst]
            if not ((-128 < a["lon"] < -65) and (23 < a["lat"] < 50)):
                continue
            if not ((-128 < b["lon"] < -65) and (23 < b["lat"] < 50)):
                continue
            ax.plot([a["lon"], b["lon"]], [a["lat"], b["lat"]],
                    color="#1a3a5c", linewidth=0.3, alpha=0.5, zorder=1)

    # Plot airports
    max_deg = max(degs) if degs else 1
    sizes = [max(5, (d / max_deg) * 400) for d in degs]
    colors = ["#ff6b35" if d > 150 else "#4fc3f7" if d > 50 else "#90caf9"
              for d in degs]

    scatter = ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.8,
                         linewidths=0, zorder=3)

    # Label top hubs
    top15 = sorted(degree.items(), key=lambda x: -x[1])[:15]
    for iata, deg in top15:
        if iata not in airports:
            continue
        ap = airports[iata]
        if not (-128 < ap["lon"] < -65 and 23 < ap["lat"] < 50):
            continue
        ax.annotate(f"{iata}", (ap["lon"], ap["lat"]),
                    fontsize=7, color="white", fontweight="bold",
                    xytext=(3, 3), textcoords="offset points", zorder=5)

    # Legend
    patches = [
        mpatches.Patch(color="#ff6b35", label="Major hub (>150 routes)"),
        mpatches.Patch(color="#4fc3f7", label="Regional hub (50-150)"),
        mpatches.Patch(color="#90caf9", label="Small airport (<50)"),
    ]
    ax.legend(handles=patches, loc="lower left", facecolor="#111", labelcolor="white",
              fontsize=9, framealpha=0.8)

    ax.set_title("USMTG: US Domestic Flight Network\n"
                 f"{len(airports)} airports · {sum(len(v) for v in graph.values())//2} direct routes",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def plot_meeting_result(loc_a, name_a, loc_b, name_b, best, top5,
                        airports, save_path="fig_meeting_result.png"):
    """Plot meeting point result on US map."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.set_xlim(-128, -65)
    ax.set_ylim(23, 52)

    # All airports (background)
    lons = [ap["lon"] for ap in airports.values()
            if -128 < ap["lon"] < -65 and 23 < ap["lat"] < 50]
    lats = [ap["lat"] for ap in airports.values()
            if -128 < ap["lon"] < -65 and 23 < ap["lat"] < 50]
    ax.scatter(lons, lats, s=4, c="#1e3a5f", alpha=0.5, zorder=1)

    # Top alternatives
    for r in top5[1:]:
        ax.scatter(r["lon"], r["lat"], s=60, c="#555", zorder=3, marker="o")
        ax.annotate(f"{r['airport']}\n{r['max_min']}min",
                    (r["lon"], r["lat"]), color="#888", fontsize=7,
                    xytext=(4, 4), textcoords="offset points")

    # Best meeting airport
    blon, blat = best["lon"], best["lat"]
    ax.scatter(blon, blat, s=300, c="#ffd700", zorder=5, marker="*")
    ax.annotate(f"* {best['airport']} ({best['city']})\n"
                f"A: {best['time_a_min']}min  B: {best['time_b_min']}min\n"
                f"Max: {best['max_min']}min",
                (blon, blat), color="white", fontsize=9, fontweight="bold",
                xytext=(8, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#ffd700", alpha=0.9),
                zorder=6)

    # Origins
    ax.scatter(loc_a[1], loc_a[0], s=200, c="#ff4444", zorder=5, marker="o")
    ax.annotate(name_a, (loc_a[1], loc_a[0]), color="#ff6666",
                fontsize=9, xytext=(-5, 8), textcoords="offset points")

    ax.scatter(loc_b[1], loc_b[0], s=200, c="#44aaff", zorder=5, marker="o")
    ax.annotate(name_b, (loc_b[1], loc_b[0]), color="#66bbff",
                fontsize=9, xytext=(-5, 8), textcoords="offset points")

    # Lines: origin → best
    ax.plot([loc_a[1], blon], [loc_a[0], blat], "--", c="#ff6666", lw=1.5, alpha=0.8)
    ax.plot([loc_b[1], blon], [loc_b[0], blat], "--", c="#66bbff", lw=1.5, alpha=0.8)

    ax.set_title(f"USMTG Meeting Point: {name_a} ↔ {name_b}\n"
                 f"Optimal: {best['city']} ({best['airport']})  |  "
                 f"Max travel {best['max_min']} min  |  Imbalance {best['imbalance_min']} min",
                 color="white", fontsize=11, pad=8)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    airports, graph = load()

    print("Generating hub map...")
    plot_hub_map(airports, graph)

    print("Running sample meeting queries...")
    pairs = [
        ("New York", "Los Angeles"),
        ("Seattle",  "Miami"),
        ("Boston",   "Dallas"),
    ]
    for city_a, city_b in pairs:
        loc_a, name_a = resolve_location(city_a, airports)
        loc_b, name_b = resolve_location(city_b, airports)
        best, top = find_meeting_airport(*loc_a, *loc_b, airports, graph)
        if best:
            safe = f"fig_meeting_{city_a.lower().replace(' ','_')}_{city_b.lower().replace(' ','_')}.png"
            plot_meeting_result(loc_a, name_a, loc_b, name_b, best, top, airports, safe)
            print(f"  {city_a} ↔ {city_b} → {best['city']} ({best['airport']}), max {best['max_min']}min")
