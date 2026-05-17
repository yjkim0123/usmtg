"""
USMTG: US Multimodal Meeting Point Finder
Mode: Drive-to-Airport → Fly → (meeting city)
"""
import json, math, heapq
from collections import defaultdict

DATA_DIR = "data"

# ── Constants ──────────────────────────────────────────────────────────────
AIRPORT_OVERHEAD_MIN = 60   # security + boarding time
MAX_DRIVE_TO_AIRPORT_KM = 200  # max drive radius to consider as departure airport
DRIVE_SPEED_KMH = 80
CRUISE_SPEED_KMH = 800
CLIMB_DESCENT_MIN = 45      # fixed per flight

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

# ── Load data ──────────────────────────────────────────────────────────────

def load():
    with open(f"{DATA_DIR}/us_airports.json") as f:
        airports = json.load(f)
    with open(f"{DATA_DIR}/us_routes.json") as f:
        routes = json.load(f)

    # Build adjacency: src → list of (dst, flight_min)
    graph = defaultdict(list)
    for r in routes:
        graph[r["src"]].append((r["dst"], r["flight_time_min"]))
        graph[r["dst"]].append((r["src"], r["flight_time_min"]))  # treat as bidirectional
    return airports, graph

# ── Step 1: Find candidate departure airports near an origin ───────────────

def nearby_airports(lat, lon, airports, max_km=MAX_DRIVE_TO_AIRPORT_KM):
    """Return list of (iata, drive_min) sorted by drive time."""
    result = []
    for iata, ap in airports.items():
        d = haversine_km(lat, lon, ap["lat"], ap["lon"])
        if d <= max_km:
            result.append((iata, drive_min(d), d))
    result.sort(key=lambda x: x[1])
    return result[:10]  # top-10 closest airports

# ── Step 2: Dijkstra from a set of source airports ─────────────────────────

def multi_source_dijkstra(sources_with_cost, graph):
    """
    sources_with_cost: list of (iata, initial_cost_min)
    Returns dict: iata → min total time to reach that airport
    """
    dist = {}
    heap = []
    for iata, cost in sources_with_cost:
        if iata not in dist or cost < dist[iata]:
            dist[iata] = cost
            heapq.heappush(heap, (cost, iata))

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        for neighbor, flight_t in graph[node]:
            # cost to reach neighbor = current + overhead + flight
            new_cost = cost + AIRPORT_OVERHEAD_MIN + flight_t
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost
                heapq.heappush(heap, (new_cost, neighbor))
    return dist

# ── Step 3: Find optimal meeting airport ──────────────────────────────────

def find_meeting_airport(lat_a, lon_a, lat_b, lon_b, airports, graph, top_k=10):
    """
    Given two origin coordinates, find the best meeting airport
    minimizing max(total_time_A, total_time_B).
    Total time = drive_to_airport + overhead + flight_to_meeting + (no local drive assumed)
    """
    near_a = nearby_airports(lat_a, lon_a, airports)
    near_b = nearby_airports(lat_b, lon_b, airports)

    if not near_a or not near_b:
        return None, []

    # Sources: (iata, drive_time) — cost before boarding
    sources_a = [(iata, d_min) for iata, d_min, _ in near_a]
    sources_b = [(iata, d_min) for iata, d_min, _ in near_b]

    # Dijkstra: total time from each origin to every reachable airport
    dist_a = multi_source_dijkstra(sources_a, graph)
    dist_b = multi_source_dijkstra(sources_b, graph)

    # Find airports reachable by both
    common = set(dist_a.keys()) & set(dist_b.keys())

    results = []
    for iata in common:
        t_a = dist_a[iata]
        t_b = dist_b[iata]
        score = max(t_a, t_b)         # min-max fairness
        imbalance = abs(t_a - t_b)
        results.append({
            "airport": iata,
            "city": airports[iata]["city"],
            "name": airports[iata]["name"],
            "lat": airports[iata]["lat"],
            "lon": airports[iata]["lon"],
            "time_a_min": round(t_a),
            "time_b_min": round(t_b),
            "max_min": round(score),
            "imbalance_min": round(imbalance),
        })

    results.sort(key=lambda x: (x["max_min"], x["imbalance_min"]))
    best = results[0] if results else None
    return best, results[:top_k]


# ── Major US cities lookup ─────────────────────────────────────────────────

CITIES = {
    "new york":     (40.7128, -74.0060),
    "nyc":          (40.7128, -74.0060),
    "los angeles":  (34.0522, -118.2437),
    "lax":          (33.9425, -118.4081),
    "chicago":      (41.8781, -87.6298),
    "houston":      (29.7604, -95.3698),
    "phoenix":      (33.4484, -112.0740),
    "philadelphia": (39.9526, -75.1652),
    "san antonio":  (29.4241, -98.4936),
    "san diego":    (32.7157, -117.1611),
    "dallas":       (32.7767, -96.7970),
    "san jose":     (37.3382, -121.8863),
    "austin":       (30.2672, -97.7431),
    "jacksonville": (30.3322, -81.6557),
    "san francisco":(37.7749, -122.4194),
    "sf":           (37.7749, -122.4194),
    "columbus":     (39.9612, -82.9988),
    "charlotte":    (35.2271, -80.8431),
    "indianapolis": (39.7684, -86.1581),
    "seattle":      (47.6062, -122.3321),
    "denver":       (39.7392, -104.9903),
    "boston":       (42.3601, -71.0589),
    "nashville":    (36.1627, -86.7816),
    "baltimore":    (39.2904, -76.6122),
    "louisville":   (38.2527, -85.7585),
    "portland":     (45.5051, -122.6750),
    "las vegas":    (36.1699, -115.1398),
    "memphis":      (35.1495, -90.0490),
    "atlanta":      (33.7490, -84.3880),
    "miami":        (25.7617, -80.1918),
    "minneapolis":  (44.9778, -93.2650),
    "honolulu":     (21.3069, -157.8583),
    "anchorage":    (61.2181, -149.9003),
}

def resolve_location(query, airports):
    q = query.lower().strip()
    # Check city dict
    if q in CITIES:
        return CITIES[q], q.title()
    # Check IATA code
    q_upper = query.upper().strip()
    if q_upper in airports:
        ap = airports[q_upper]
        return (ap["lat"], ap["lon"]), f"{ap['city']} ({q_upper})"
    # Fuzzy city match in airports
    for iata, ap in airports.items():
        if q in ap["city"].lower():
            return (ap["lat"], ap["lon"]), f"{ap['city']} ({iata})"
    return None, None


# ── CLI demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    airports, graph = load()
    print(f"Loaded {len(airports)} airports, {sum(len(v) for v in graph.values())//2} routes\n")

    test_pairs = [
        ("New York", "Los Angeles"),
        ("Seattle", "Miami"),
        ("Boston", "Dallas"),
        ("Chicago", "San Francisco"),
    ]

    for city_a, city_b in test_pairs:
        loc_a, name_a = resolve_location(city_a, airports)
        loc_b, name_b = resolve_location(city_b, airports)
        if not loc_a or not loc_b:
            print(f"Could not resolve: {city_a} or {city_b}")
            continue

        best, top = find_meeting_airport(*loc_a, *loc_b, airports, graph)
        if not best:
            print(f"No common airports found for {city_a} ↔ {city_b}")
            continue

        print(f"{'='*55}")
        print(f"{name_a}  ↔  {name_b}")
        print(f"{'='*55}")
        print(f"Best meeting: {best['city']} ({best['airport']}) — {best['name']}")
        print(f"  Person A: {best['time_a_min']} min total")
        print(f"  Person B: {best['time_b_min']} min total")
        print(f"  Max time: {best['max_min']} min  |  Imbalance: {best['imbalance_min']} min")
        print(f"\nTop 5 alternatives:")
        for r in top[:5]:
            print(f"  {r['airport']:4s} {r['city']:20s}  max={r['max_min']:4d}min  A={r['time_a_min']:4d}  B={r['time_b_min']:4d}  diff={r['imbalance_min']:3d}")
        print()
