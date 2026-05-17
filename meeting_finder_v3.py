"""
USMTG v3: Comprehensive US Meeting Point Finder

Over v2:
  1. N-party meeting (3+ travelers)
  2. 3-objective Pareto: max_time × total_cost × total_co2
  3. Nash Bargaining + Kalai-Smorodinsky fairness scores
  4. Monte Carlo robustness (500 delay simulations per candidate)
  5. Time window constraints (depart_after_h, arrive_before_h)
"""
import json, math, heapq, random
from collections import defaultdict
from meeting_finder_v2 import load, resolve_location, fmt_time

# ── Constants (inherited from v2) ──────────────────────────────────────────
INITIAL_OVERHEAD_MIN = 90
CONNECTION_MIN       = 45
MAX_DRIVE_KM         = 200
DRIVE_SPEED_KMH      = 80
CRUISE_SPEED_KMH     = 800
CLIMB_DESCENT_MIN    = 45
MAX_STOPS            = 2

HUB_AIRPORTS = {
    "ATL","ORD","DEN","DFW","LAX","JFK","SFO","LAS","SEA","CLT",
    "MCO","PHX","IAH","BOS","MSP","DTW","EWR","PHL","SLC","MIA",
}

# ── CO₂ model (ICAO 2023, economy, per passenger) ─────────────────────────
def co2_kg(dist_km):
    if dist_km < 500:
        return round(dist_km * 0.255, 1)   # 255 g/pkm short-haul
    elif dist_km < 1500:
        return round(dist_km * 0.195, 1)   # 195 g/pkm medium-haul
    else:
        return round(dist_km * 0.150, 1)   # 150 g/pkm long-haul

# ── Fare model (same as v2) ────────────────────────────────────────────────
def fare_usd(dist_km, dep, arr):
    miles = dist_km * 0.621371
    base = (180 if miles < 300 else 210 if miles < 600 else
            240 if miles < 1000 else 270 if miles < 1500 else
            300 if miles < 2000 else 330)
    discount = 1.0 - (0.10 if dep in HUB_AIRPORTS else 0) \
                   - (0.10 if arr in HUB_AIRPORTS else 0)
    return round(base * discount)

# ── Helpers ────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def nearby_airports(lat, lon, airports, max_km=MAX_DRIVE_KM):
    result = []
    for iata, ap in airports.items():
        d = haversine_km(lat, lon, ap["lat"], ap["lon"])
        if d <= max_km:
            result.append((iata, d / DRIVE_SPEED_KMH * 60, d))
    result.sort(key=lambda x: x[1])
    return result[:12]

# ── Dijkstra v3: tracks (time, fare, co2, stops) ──────────────────────────
def dijkstra_v3(sources_with_drive_min, graph, max_stops=MAX_STOPS,
                arrive_before_min=None):
    """
    Returns dict: iata → (time_min, fare_usd, co2_kg, stops)
    arrive_before_min: hard cap on total travel time (None = unconstrained)
    """
    best = {}
    heap = []
    for iata, drive_t in sources_with_drive_min:
        t = drive_t + INITIAL_OVERHEAD_MIN
        state = (t, 0.0, 0.0, 0, iata)
        heap.append(state)
        best[iata] = (t, 0.0, 0.0, 0)
    heapq.heapify(heap)

    while heap:
        time, fare, emit, stops, node = heapq.heappop(heap)
        if node in best and time > best[node][0] + 1:
            continue
        if stops >= max_stops + 1:
            continue
        for neighbor, flight_t, flight_fare, dist_km in graph.get(node, []):
            new_stops = stops + 1
            if new_stops > max_stops + 1:
                continue
            conn = CONNECTION_MIN if stops > 0 else 0
            new_time = time + conn + flight_t
            if arrive_before_min and new_time > arrive_before_min:
                continue
            new_fare = fare + flight_fare
            new_emit = emit + co2_kg(dist_km)
            if neighbor not in best or new_time < best[neighbor][0]:
                best[neighbor] = (new_time, new_fare, new_emit, new_stops)
                heapq.heappush(heap, (new_time, new_fare, new_emit, new_stops, neighbor))
    return best

# ── 3-objective Pareto non-dominated sort ─────────────────────────────────
def pareto_3obj(results, obj_keys):
    """
    Returns non-dominated subset minimizing all objectives in obj_keys.
    """
    dominated = set()
    n = len(results)
    for i in range(n):
        for j in range(n):
            if i == j or j in dominated:
                continue
            if all(results[j][k] <= results[i][k] for k in obj_keys) and \
               any(results[j][k] <  results[i][k] for k in obj_keys):
                dominated.add(i)
                break
    return [r for idx, r in enumerate(results) if idx not in dominated]

# ── Monte Carlo robustness ─────────────────────────────────────────────────
# BTS 2023: mean delay ~12 min, log-normal distribution
_DELAY_MU    = 2.2    # log-normal μ  → mean ≈ 12 min
_DELAY_SIGMA = 0.80   # log-normal σ
_CANCEL_PROB = 0.015  # 1.5% cancellation

def _sample_delay(rng):
    if rng.random() < _CANCEL_PROB:
        return float("inf")
    return max(0.0, rng.lognormvariate(_DELAY_MU, _DELAY_SIGMA) - 5)

def monte_carlo_robustness(travel_time_min, num_stops, n_sim=500, seed=0):
    """
    Simulate n_sim trips. Returns (p50, p90, miss_prob).
    miss_prob = fraction of trips that miss a connection or get cancelled.
    """
    rng = random.Random(seed)
    num_flights = max(1, num_stops)
    times, misses = [], 0

    for _ in range(n_sim):
        extra, missed = 0.0, False
        for f in range(num_flights):
            delay = _sample_delay(rng)
            if delay == float("inf"):
                extra += 180; missed = True; break   # cancellation → rebook 3h
            elif f < num_flights - 1 and delay > CONNECTION_MIN:
                extra += 90; missed = True           # missed connection → rebook 1.5h
            else:
                extra += delay
        times.append(travel_time_min + extra)
        if missed:
            misses += 1

    times.sort()
    return {
        "p50_min": round(times[n_sim // 2]),
        "p90_min": round(times[int(n_sim * 0.9)]),
        "miss_prob": round(misses / n_sim, 3),
    }

# ── Fairness metrics ───────────────────────────────────────────────────────
def nash_score(times, best_times, worst_time):
    """
    Nash Bargaining: product of surpluses.
    Surplus_i = worst_time - actual_time_i  (higher = better for person i)
    Maximizing the product is the NBS.
    """
    score = 1.0
    for t in times:
        surplus = max(0, worst_time - t)
        score *= surplus
    return round(score)

def ks_score(times, best_times, worst_time):
    """
    Kalai-Smorodinsky: min over persons of (surplus_i / ideal_surplus_i).
    ideal_surplus_i = worst_time - best_time_i.
    Higher = more balanced (closer to each person's proportional ideal).
    """
    ratios = []
    for t, best in zip(times, best_times):
        ideal = max(1, worst_time - best)
        actual = max(0, worst_time - t)
        ratios.append(actual / ideal)
    return round(min(ratios), 4)

# ── N-party meeting finder ─────────────────────────────────────────────────
def find_meeting_nparty(origins, airports, graph,
                        max_stops=MAX_STOPS, top_k=10,
                        depart_after_h=0, arrive_before_h=24,
                        n_sim=500):
    """
    origins: list of (lat, lon, label)
    Returns (best, top_k_list, pareto_front)
    """
    arrive_before_min = (arrive_before_h - depart_after_h) * 60
    if arrive_before_h >= 24:
        arrive_before_min = None

    all_dists = []
    for lat, lon, label in origins:
        near = nearby_airports(lat, lon, airports)
        if not near:
            return None, [], []
        sources = [(iata, d_min) for iata, d_min, _ in near]
        d = dijkstra_v3(sources, graph, max_stops, arrive_before_min)
        all_dists.append(d)

    # Airports reachable by ALL parties
    common = set(all_dists[0].keys())
    for d in all_dists[1:]:
        common &= set(d.keys())
    if not common:
        return None, [], []

    # Per-person best possible time (for Nash/KS baselines)
    best_individual = [min(d[k][0] for k in d) for d in all_dists]
    worst_time = max(
        max(d[k][0] for k in common) for d in all_dists
    )

    results = []
    for iata in common:
        ap = airports[iata]
        times = [d[iata][0] for d in all_dists]
        fares = [d[iata][1] for d in all_dists]
        emits = [d[iata][2] for d in all_dists]
        stops = [d[iata][3] for d in all_dists]

        max_time   = max(times)
        total_cost = sum(fares)
        total_co2  = round(sum(emits), 1)
        imbalance  = max(times) - min(times)

        # Robustness: simulate the worst-off traveler
        worst_idx = times.index(max_time)
        rob = monte_carlo_robustness(max_time, stops[worst_idx], n_sim)

        results.append({
            "airport":    iata,
            "city":       ap["city"],
            "name":       ap["name"],
            "lat":        ap["lat"],
            "lon":        ap["lon"],
            "times_min":  [round(t) for t in times],
            "fares_usd":  [round(f) for f in fares],
            "co2_kg":     [round(e, 1) for e in emits],
            "stops":      stops,
            "max_time_min":   round(max_time),
            "total_cost_usd": round(total_cost),
            "total_co2_kg":   total_co2,
            "imbalance_min":  round(imbalance),
            "nash_score":     nash_score(times, best_individual, worst_time),
            "ks_score":       ks_score(times, best_individual, worst_time),
            "p90_min":        rob["p90_min"],
            "miss_prob":      rob["miss_prob"],
        })

    if not results:
        return None, [], []

    results.sort(key=lambda x: (x["max_time_min"], x["imbalance_min"]))

    # 3-objective Pareto front
    pareto = pareto_3obj(results, ["max_time_min", "total_cost_usd", "total_co2_kg"])
    pareto.sort(key=lambda x: x["max_time_min"])

    return results[0], results[:top_k], pareto[:6]


# ── Pretty print ───────────────────────────────────────────────────────────
def print_result(origin_names, best, top, pareto):
    sep = "=" * 70
    header = "  ↔  ".join(origin_names)
    n = len(origin_names)
    labels = [chr(65 + i) for i in range(n)]   # A, B, C, ...

    print(f"\n{sep}")
    print(f" {header}")
    print(sep)
    print(f" Best (min-max): {best['city']} ({best['airport']})")
    for i, lbl in enumerate(labels):
        print(f"   {lbl}: {fmt_time(best['times_min'][i])}  "
              f"${best['fares_usd'][i]}  "
              f"{best['co2_kg'][i]}kg CO₂  "
              f"({best['stops'][i]-1} stop{'s' if best['stops'][i]>2 else ''})")
    print(f"   ▶ Max: {fmt_time(best['max_time_min'])}  "
          f"Cost: ${best['total_cost_usd']}  "
          f"CO₂: {best['total_co2_kg']}kg  "
          f"Imbalance: {best['imbalance_min']}min")
    print(f"   ▶ Nash: {best['nash_score']:,}  "
          f"K-S: {best['ks_score']:.3f}  "
          f"p90: {fmt_time(best['p90_min'])}  "
          f"Miss: {best['miss_prob']*100:.1f}%")

    print(f"\n 3-obj Pareto (time × cost × CO₂):")
    for r in pareto:
        print(f"   {r['airport']:4s} {r['city']:22s} "
              f"max={fmt_time(r['max_time_min'])}  "
              f"${r['total_cost_usd']:4d}  "
              f"{r['total_co2_kg']:5.1f}kg  "
              f"miss={r['miss_prob']*100:.1f}%")

    print(f"\n Top-5 by time:")
    for r in top[:5]:
        t_str = "/".join(fmt_time(t) for t in r["times_min"])
        print(f"   {r['airport']:4s} {r['city']:22s} "
              f"max={fmt_time(r['max_time_min'])}  "
              f"[{t_str}]  "
              f"${r['total_cost_usd']}  "
              f"KS={r['ks_score']:.3f}")

    print(f"\n Fairness comparison (top-5 by Nash score):")
    by_nash = sorted(top, key=lambda x: -x["nash_score"])[:5]
    for r in by_nash:
        print(f"   {r['airport']:4s} {r['city']:22s} "
              f"Nash={r['nash_score']:,}  "
              f"KS={r['ks_score']:.3f}  "
              f"max={fmt_time(r['max_time_min'])}")


# ── CLI demo ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    airports, graph = load()
    print(f"Loaded {len(airports)} airports\n")

    # 2-party demos
    two_party = [
        ("New York",   "Los Angeles"),
        ("Seattle",    "Miami"),
        ("Boston",     "Dallas"),
        ("Honolulu",   "New York"),
    ]
    for city_a, city_b in two_party:
        loc_a, name_a = resolve_location(city_a, airports)
        loc_b, name_b = resolve_location(city_b, airports)
        origins = [(loc_a[0], loc_a[1], name_a),
                   (loc_b[0], loc_b[1], name_b)]
        best, top, pareto = find_meeting_nparty(origins, airports, graph)
        if best:
            print_result([name_a, name_b], best, top, pareto)

    # 3-party demo
    print("\n" + "=" * 70)
    print(" ★ 3-PARTY: New York  ↔  Los Angeles  ↔  Chicago")
    print("=" * 70)
    three_cities = [("New York", "New York"), ("Los Angeles", "Los Angeles"),
                    ("Chicago", "Chicago")]
    origins_3 = []
    for city, label in three_cities:
        loc, name = resolve_location(city, airports)
        origins_3.append((loc[0], loc[1], name))
    best3, top3, pareto3 = find_meeting_nparty(origins_3, airports, graph)
    if best3:
        print_result(["New York", "Los Angeles", "Chicago"], best3, top3, pareto3)

    # Time-window demo
    print("\n" + "=" * 70)
    print(" ★ TIME WINDOW: Boston ↔ Dallas (depart 07:00, arrive by 14:00)")
    print("=" * 70)
    loc_bos, _ = resolve_location("Boston", airports)
    loc_dal, _ = resolve_location("Dallas", airports)
    origins_tw = [(loc_bos[0], loc_bos[1], "Boston"),
                  (loc_dal[0], loc_dal[1], "Dallas")]
    best_tw, top_tw, pareto_tw = find_meeting_nparty(
        origins_tw, airports, graph,
        depart_after_h=7, arrive_before_h=14
    )
    if best_tw:
        print_result(["Boston", "Dallas"], best_tw, top_tw, pareto_tw)
    else:
        print("  No feasible meeting point within time window.")
