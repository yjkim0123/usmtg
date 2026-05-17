"""
USMTG v2: Enhanced US Meeting Point Finder
Improvements:
  1. Proper layover model (initial boarding vs connection overhead)
  2. Cost estimation (distance-based fare model + hub discount)
  3. Max stops limit (0, 1, or 2 connections)
  4. Bi-objective Pareto: minimize max_time AND max_cost
"""
import json, math, heapq
from collections import defaultdict

DATA_DIR = "data"

# ── Constants ──────────────────────────────────────────────────────────────
INITIAL_OVERHEAD_MIN = 90    # drive + park + security + boarding (at departure)
CONNECTION_MIN = 45          # minimum connection time at layover airport
MAX_DRIVE_KM = 200           # max drive radius to departure airport
DRIVE_SPEED_KMH = 80
CRUISE_SPEED_KMH = 800
CLIMB_DESCENT_MIN = 45
MAX_STOPS = 2                # maximum layovers allowed

# Major hub airports (get fare discount due to competition)
HUB_AIRPORTS = {
    "ATL","ORD","DEN","DFW","LAX","JFK","SFO","LAS","SEA","CLT",
    "MCO","PHX","IAH","BOS","MSP","DTW","EWR","PHL","SLC","MIA",
}

# ── Helpers ────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def drive_min(dist_km):
    return dist_km / DRIVE_SPEED_KMH * 60

def flight_min(dist_km):
    return dist_km / CRUISE_SPEED_KMH * 60 + CLIMB_DESCENT_MIN

def estimate_fare_usd(dist_km, dep_iata, arr_iata):
    """
    Distance-based US domestic fare estimate (BTS 2023 averages).
    Hub airports get a 20% discount due to competition.
    Very short routes (<200km) often have high per-km fares.
    """
    miles = dist_km * 0.621371
    if miles < 300:
        base = 180
    elif miles < 600:
        base = 210
    elif miles < 1000:
        base = 240
    elif miles < 1500:
        base = 270
    elif miles < 2000:
        base = 300
    else:
        base = 330

    discount = 1.0
    if dep_iata in HUB_AIRPORTS:
        discount -= 0.10
    if arr_iata in HUB_AIRPORTS:
        discount -= 0.10
    return round(base * discount)

# ── Load data ──────────────────────────────────────────────────────────────

def load():
    with open(f"{DATA_DIR}/us_airports.json") as f:
        airports = json.load(f)
    with open(f"{DATA_DIR}/us_routes.json") as f:
        routes_list = json.load(f)

    # Build adjacency with distance and fare
    graph = defaultdict(list)
    for r in routes_list:
        src, dst = r["src"], r["dst"]
        dist = r["dist_km"]
        t = r["flight_time_min"]
        fare = estimate_fare_usd(dist, src, dst)
        graph[src].append((dst, t, fare, dist))
        graph[dst].append((src, t, fare, dist))
    return airports, graph

# ── Nearby airports ────────────────────────────────────────────────────────

def nearby_airports(lat, lon, airports, max_km=MAX_DRIVE_KM):
    result = []
    for iata, ap in airports.items():
        d = haversine_km(lat, lon, ap["lat"], ap["lon"])
        if d <= max_km:
            result.append((iata, drive_min(d), d))
    result.sort(key=lambda x: x[1])
    return result[:12]

# ── Enhanced Dijkstra: tracks (time, cost, stops) ─────────────────────────

def multi_source_dijkstra_v2(sources_with_drive_min, graph, max_stops=MAX_STOPS):
    """
    State: (total_time, iata, num_stops, total_fare)
    Returns dict: iata → (best_time_min, best_fare_usd, num_stops, path)
    """
    # best[iata] = (time, fare, stops)
    best = {}
    # heap: (time, fare, stops, iata)
    heap = []

    for iata, drive_t in sources_with_drive_min:
        # Initial cost: drive + security + boarding overhead
        init_time = drive_t + INITIAL_OVERHEAD_MIN
        init_fare = 0  # no flight yet
        state = (init_time, 0.0, 0, iata)
        heap.append(state)
        best[iata] = (init_time, 0.0, 0)

    heapq.heapify(heap)

    while heap:
        time, fare, stops, node = heapq.heappop(heap)

        # Prune stale states
        if node in best and time > best[node][0] + 1:
            continue
        if stops >= max_stops + 1:  # already at destination, no more hops
            continue

        for neighbor, flight_t, flight_fare, dist_km in graph.get(node, []):
            new_stops = stops + 1
            if new_stops > max_stops + 1:
                continue
            # Time: current + connection overhead + flight time
            conn = CONNECTION_MIN if stops > 0 else 0  # no extra connection at first hop (already in INITIAL_OVERHEAD)
            new_time = time + conn + flight_t
            new_fare = fare + flight_fare

            if neighbor not in best or new_time < best[neighbor][0]:
                best[neighbor] = (new_time, new_fare, new_stops)
                heapq.heappush(heap, (new_time, new_fare, new_stops, neighbor))

    return best

# ── Meeting point search ───────────────────────────────────────────────────

def find_meeting_v2(lat_a, lon_a, lat_b, lon_b, airports, graph,
                    max_stops=MAX_STOPS, top_k=10):

    near_a = nearby_airports(lat_a, lon_a, airports)
    near_b = nearby_airports(lat_b, lon_b, airports)

    if not near_a or not near_b:
        return None, []

    sources_a = [(iata, d_min) for iata, d_min, _ in near_a]
    sources_b = [(iata, d_min) for iata, d_min, _ in near_b]

    dist_a = multi_source_dijkstra_v2(sources_a, graph, max_stops)
    dist_b = multi_source_dijkstra_v2(sources_b, graph, max_stops)

    common = set(dist_a.keys()) & set(dist_b.keys())

    results = []
    for iata in common:
        t_a, f_a, s_a = dist_a[iata]
        t_b, f_b, s_b = dist_b[iata]
        max_time = max(t_a, t_b)
        total_cost = f_a + f_b
        imbalance = abs(t_a - t_b)
        results.append({
            "airport": iata,
            "city": airports[iata]["city"],
            "name": airports[iata]["name"],
            "lat": airports[iata]["lat"],
            "lon": airports[iata]["lon"],
            "time_a_min": round(t_a),
            "time_b_min": round(t_b),
            "fare_a_usd": round(f_a),
            "fare_b_usd": round(f_b),
            "stops_a": s_a,
            "stops_b": s_b,
            "max_time_min": round(max_time),
            "total_cost_usd": round(total_cost),
            "imbalance_min": round(imbalance),
        })

    if not results:
        return None, []

    # ── Pareto front: non-dominated on (max_time, total_cost) ──
    pareto = []
    results.sort(key=lambda x: (x["max_time_min"], x["total_cost_usd"]))
    best_cost_so_far = float("inf")
    for r in results:
        if r["total_cost_usd"] < best_cost_so_far:
            pareto.append(r)
            best_cost_so_far = r["total_cost_usd"]

    # Also sort by min-max time for general top-k
    results.sort(key=lambda x: (x["max_time_min"], x["imbalance_min"]))

    return results[0], results[:top_k], pareto[:5]

# ── City lookup ────────────────────────────────────────────────────────────

CITIES = {
    "new york": (40.7128, -74.0060), "nyc": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437), "la": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio": (29.4241, -98.4936),
    "san diego": (32.7157, -117.1611),
    "dallas": (32.7767, -96.7970),
    "san jose": (37.3382, -121.8863),
    "austin": (30.2672, -97.7431),
    "san francisco": (37.7749, -122.4194), "sf": (37.7749, -122.4194),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "boston": (42.3601, -71.0589),
    "nashville": (36.1627, -86.7816),
    "atlanta": (33.7490, -84.3880),
    "miami": (25.7617, -80.1918),
    "minneapolis": (44.9778, -93.2650),
    "las vegas": (36.1699, -115.1398),
    "portland": (45.5051, -122.6750),
    "charlotte": (35.2271, -80.8431),
    "memphis": (35.1495, -90.0490),
    "louisville": (38.2527, -85.7585),
    "indianapolis": (39.7684, -86.1581),
    "columbus": (39.9612, -82.9988),
    "honolulu": (21.3069, -157.8583),
    "anchorage": (61.2181, -149.9003),
    "raleigh": (35.7796, -78.6382),
    "richmond": (37.5407, -77.4360),
    "salt lake city": (40.7608, -111.8910),
    "kansas city": (39.0997, -94.5786),
    "pittsburgh": (40.4406, -79.9959),
    "cincinnati": (39.1031, -84.5120),
    "cleveland": (41.4993, -81.6944),
    "detroit": (42.3314, -83.0458),
    "milwaukee": (43.0389, -87.9065),
    "new orleans": (29.9511, -90.0715),
    "oklahoma city": (35.4676, -97.5164),
    "tucson": (32.2226, -110.9747),
    "albuquerque": (35.0844, -106.6504),
}

def resolve_location(query, airports):
    q = query.lower().strip()
    if q in CITIES:
        return CITIES[q], query.title()
    q_up = query.upper().strip()
    if q_up in airports:
        ap = airports[q_up]
        return (ap["lat"], ap["lon"]), f"{ap['city']} ({q_up})"
    for iata, ap in airports.items():
        if q in ap["city"].lower():
            return (ap["lat"], ap["lon"]), f"{ap['city']} ({iata})"
    return None, None


# ── CLI demo ───────────────────────────────────────────────────────────────

def fmt_time(minutes):
    h, m = divmod(round(minutes), 60)
    return f"{h}h{m:02d}m"

if __name__ == "__main__":
    airports, graph = load()
    print(f"Loaded {len(airports)} airports\n")

    test_pairs = [
        ("New York",   "Los Angeles"),
        ("Seattle",    "Miami"),
        ("Boston",     "Dallas"),
        ("Chicago",    "San Francisco"),
        ("Honolulu",   "New York"),     # extreme distance
        ("Anchorage",  "Miami"),        # diagonal US
    ]

    for city_a, city_b in test_pairs:
        loc_a, name_a = resolve_location(city_a, airports)
        loc_b, name_b = resolve_location(city_b, airports)
        if not loc_a or not loc_b:
            continue

        best, top, pareto = find_meeting_v2(*loc_a, *loc_b, airports, graph)
        if not best:
            print(f"No result: {city_a} ↔ {city_b}")
            continue

        print(f"{'='*62}")
        print(f" {name_a}  ↔  {name_b}")
        print(f"{'='*62}")
        print(f" Best (min-max time): {best['city']} ({best['airport']})")
        print(f"   Person A: {fmt_time(best['time_a_min'])}  ${best['fare_a_usd']}  ({best['stops_a']-1} stop{'s' if best['stops_a']>2 else ''})")
        print(f"   Person B: {fmt_time(best['time_b_min'])}  ${best['fare_b_usd']}  ({best['stops_b']-1} stop{'s' if best['stops_b']>2 else ''})")
        print(f"   Max: {fmt_time(best['max_time_min'])}  Total cost: ${best['total_cost_usd']}")

        print(f"\n Pareto front (time vs cost):")
        for r in pareto:
            print(f"   {r['airport']:4s} {r['city']:20s}  "
                  f"max={fmt_time(r['max_time_min'])}  "
                  f"cost=${r['total_cost_usd']:4d}  "
                  f"diff={r['imbalance_min']}min")

        print(f"\n Top-5 by time:")
        for r in top[:5]:
            print(f"   {r['airport']:4s} {r['city']:20s}  "
                  f"max={fmt_time(r['max_time_min'])}  "
                  f"A={fmt_time(r['time_a_min'])}  B={fmt_time(r['time_b_min'])}  "
                  f"${r['total_cost_usd']}")
        print()
