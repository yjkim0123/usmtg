"""
USMTG v2 Visualization: US airport hub map + meeting point results with cost
"""
import json, math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from collections import defaultdict
from meeting_finder_v2 import (load, resolve_location, find_meeting_v2,
                                haversine_km, fmt_time)

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
    ax.set_xlim(-128, -65)
    ax.set_ylim(23, 50)

    # Routes (sample top-80 hubs)
    top_iatas = {iata for iata, _ in sorted(degree.items(), key=lambda x: -x[1])[:80]}
    plotted_edges = set()
    for src, neighbors in graph.items():
        if src not in top_iatas:
            continue
        a = airports.get(src)
        if not a:
            continue
        for dst, _, _, _ in neighbors[:5]:
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

    max_deg = max(degs) if degs else 1
    sizes = [max(5, (d / max_deg) * 400) for d in degs]
    colors = ["#ff6b35" if d > 150 else "#4fc3f7" if d > 50 else "#90caf9"
              for d in degs]
    ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.8, linewidths=0, zorder=3)

    for iata, deg in sorted(degree.items(), key=lambda x: -x[1])[:15]:
        if iata not in airports:
            continue
        ap = airports[iata]
        if not (-128 < ap["lon"] < -65 and 23 < ap["lat"] < 50):
            continue
        ax.annotate(iata, (ap["lon"], ap["lat"]),
                    fontsize=7, color="white", fontweight="bold",
                    xytext=(3, 3), textcoords="offset points", zorder=5)

    patches = [
        mpatches.Patch(color="#ff6b35", label="Major hub (>150 routes)"),
        mpatches.Patch(color="#4fc3f7", label="Regional hub (50–150)"),
        mpatches.Patch(color="#90caf9", label="Small airport (<50)"),
    ]
    ax.legend(handles=patches, loc="lower left", facecolor="#111",
              labelcolor="white", fontsize=9, framealpha=0.8)

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


def plot_meeting_v2(loc_a, name_a, loc_b, name_b,
                    best, top5, pareto, airports,
                    save_path="fig_meeting_result.png"):
    """
    Meeting point map with cost information and Pareto front inset.
    """
    fig = plt.figure(figsize=(16, 9))
    # Main map: left 75%
    ax = fig.add_axes([0.0, 0.0, 0.74, 1.0])
    # Pareto inset: right 25%
    ax_p = fig.add_axes([0.76, 0.12, 0.22, 0.72])

    # ── Main map ──────────────────────────────────────────────────────────
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-128, -65)
    ax.set_ylim(23, 52)

    # Background airports
    bg_lons = [ap["lon"] for ap in airports.values()
               if -128 < ap["lon"] < -65 and 23 < ap["lat"] < 50]
    bg_lats = [ap["lat"] for ap in airports.values()
               if -128 < ap["lon"] < -65 and 23 < ap["lat"] < 50]
    ax.scatter(bg_lons, bg_lats, s=4, c="#1e3a5f", alpha=0.4, zorder=1)

    # Pareto airports (ranked by cost, coolest = cheapest)
    if pareto:
        p_costs = [r["total_cost_usd"] for r in pareto]
        p_max, p_min = max(p_costs), min(p_costs)
        for r in pareto:
            if r["airport"] == best["airport"]:
                continue
            norm = (r["total_cost_usd"] - p_min) / max(p_max - p_min, 1)
            color = plt.cm.RdYlGn_r(norm * 0.7 + 0.1)
            ax.scatter(r["lon"], r["lat"], s=120, c=[color], zorder=4,
                       marker="D", edgecolors="white", linewidths=0.5)
            ax.annotate(f"{r['airport']}\n${r['total_cost_usd']}",
                        (r["lon"], r["lat"]), color="#cccccc", fontsize=7,
                        xytext=(5, 5), textcoords="offset points", zorder=5)

    # Top alternatives (not in pareto)
    pareto_iatas = {r["airport"] for r in pareto}
    for r in top5[1:6]:
        if r["airport"] in pareto_iatas or r["airport"] == best["airport"]:
            continue
        ax.scatter(r["lon"], r["lat"], s=50, c="#555", zorder=3, marker="o")
        ax.annotate(r["airport"], (r["lon"], r["lat"]),
                    color="#777", fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    # Best meeting airport
    blon, blat = best["lon"], best["lat"]
    ax.scatter(blon, blat, s=400, c="#ffd700", zorder=6, marker="*",
               edgecolors="white", linewidths=0.8)
    label = (f"★ {best['airport']} — {best['city']}\n"
             f"A: {fmt_time(best['time_a_min'])}  ${best['fare_a_usd']}\n"
             f"B: {fmt_time(best['time_b_min'])}  ${best['fare_b_usd']}\n"
             f"Max: {fmt_time(best['max_time_min'])}  Total: ${best['total_cost_usd']}")
    ax.annotate(label, (blon, blat), color="white", fontsize=9,
                fontweight="bold", xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e",
                          ec="#ffd700", alpha=0.92),
                zorder=7)

    # Origins
    ax.scatter(loc_a[1], loc_a[0], s=220, c="#ff4444", zorder=5,
               marker="o", edgecolors="white", linewidths=0.8)
    ax.annotate(name_a, (loc_a[1], loc_a[0]), color="#ff6666",
                fontsize=9, fontweight="bold",
                xytext=(-5, 10), textcoords="offset points")

    ax.scatter(loc_b[1], loc_b[0], s=220, c="#44aaff", zorder=5,
               marker="o", edgecolors="white", linewidths=0.8)
    ax.annotate(name_b, (loc_b[1], loc_b[0]), color="#66bbff",
                fontsize=9, fontweight="bold",
                xytext=(-5, 10), textcoords="offset points")

    # Travel lines
    ax.plot([loc_a[1], blon], [loc_a[0], blat],
            "--", c="#ff6666", lw=1.8, alpha=0.85, zorder=4)
    ax.plot([loc_b[1], blon], [loc_b[0], blat],
            "--", c="#66bbff", lw=1.8, alpha=0.85, zorder=4)

    ax.set_title(
        f"USMTG Meeting Point: {name_a} ↔ {name_b}\n"
        f"Optimal: {best['city']} ({best['airport']})  |  "
        f"Max {fmt_time(best['max_time_min'])}  |  Total ${best['total_cost_usd']}  |  "
        f"Imbalance {best['imbalance_min']} min",
        color="white", fontsize=10, pad=8)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    # ── Pareto inset ───────────────────────────────────────────────────────
    ax_p.set_facecolor("#111827")
    for spine in ax_p.spines.values():
        spine.set_edgecolor("#444")

    if pareto:
        times = [r["max_time_min"] / 60 for r in pareto]
        costs = [r["total_cost_usd"] for r in pareto]
        labels_p = [r["airport"] for r in pareto]

        ax_p.plot(times, costs, "o-", color="#ffd700", lw=1.5,
                  markersize=7, markeredgecolor="white", markeredgewidth=0.5,
                  zorder=3)

        # Highlight best
        btime = best["max_time_min"] / 60
        bcost = best["total_cost_usd"]
        ax_p.scatter([btime], [bcost], s=150, c="#ffd700", marker="*",
                     zorder=5, edgecolors="white", linewidths=0.8)

        for t, c, lbl in zip(times, costs, labels_p):
            ax_p.annotate(lbl, (t, c), color="white", fontsize=8,
                          xytext=(4, 4), textcoords="offset points")

        ax_p.set_xlabel("Max travel time (h)", color="#aaa", fontsize=8)
        ax_p.set_ylabel("Total cost ($)", color="#aaa", fontsize=8)
        ax_p.set_title("Pareto Front\n(time vs cost)", color="white",
                        fontsize=9, pad=6)
        ax_p.tick_params(colors="#aaa", labelsize=7)

        # Shade Pareto region
        ax_p.fill_betweenx(costs, [min(times)] * len(costs), times,
                            alpha=0.08, color="#ffd700")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    airports, graph = load()

    print("Generating hub map...")
    plot_hub_map(airports, graph)

    pairs = [
        ("New York",  "Los Angeles"),
        ("Seattle",   "Miami"),
        ("Boston",    "Dallas"),
        ("Chicago",   "San Francisco"),
        ("Honolulu",  "New York"),
        ("Anchorage", "Miami"),
    ]

    for city_a, city_b in pairs:
        loc_a, name_a = resolve_location(city_a, airports)
        loc_b, name_b = resolve_location(city_b, airports)
        if not loc_a or not loc_b:
            print(f"Cannot resolve: {city_a} / {city_b}")
            continue

        best, top, pareto = find_meeting_v2(*loc_a, *loc_b, airports, graph)
        if not best:
            print(f"No result: {city_a} ↔ {city_b}")
            continue

        fname = (f"fig_meeting_"
                 f"{city_a.lower().replace(' ', '_')}_"
                 f"{city_b.lower().replace(' ', '_')}.png")
        plot_meeting_v2(loc_a, name_a, loc_b, name_b,
                        best, top, pareto, airports, fname)
        print(f"  {city_a} ↔ {city_b} → {best['city']} ({best['airport']}), "
              f"max {fmt_time(best['max_time_min'])}, ${best['total_cost_usd']}")
