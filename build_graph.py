"""
US Multimodal Transit Graph (USMTG) - Data Collection & Graph Builder
Modes: Drive (origin→airport), Fly (airport→airport), Drive (airport→destination)
"""
import csv
import json
import math
from collections import defaultdict

DATA_DIR = "data"

# ─── Load airports ──────────────────────────────────────────────────────────

def load_us_airports(path=f"{DATA_DIR}/airports.dat"):
    """Filter OpenFlights airports to US commercial airports (IATA + US country)."""
    airports = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            # Fields: id, name, city, country, IATA, ICAO, lat, lon, alt, tz, DST, tz_db, type, source
            if len(row) < 14:
                continue
            airport_id, name, city, country, iata, icao, lat, lon = row[:8]
            if country != "United States" or iata in ("", "\\N") or len(iata) != 3:
                continue
            try:
                airports[iata] = {
                    "id": airport_id,
                    "name": name,
                    "city": city,
                    "iata": iata,
                    "icao": icao,
                    "lat": float(lat),
                    "lon": float(lon),
                }
            except ValueError:
                continue
    return airports


# ─── Load routes ────────────────────────────────────────────────────────────

def load_us_routes(airports, path=f"{DATA_DIR}/routes.dat"):
    """Load routes between US airports. Fields: airline,id,src,srcId,dst,dstId,codeshare,stops,equipment"""
    routes = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                continue
            airline, _, src, _, dst, _, _, stops, _ = row[:9]
            if stops != "0":  # direct flights only
                continue
            if src not in airports or dst not in airports:
                continue
            key = (src, dst)
            if key in seen:
                continue
            seen.add(key)
            routes.append({"src": src, "dst": dst, "airline": airline})
    return routes


# ─── Haversine distance ──────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def flight_time_min(dist_km):
    """Estimate flight time: 800 km/h cruise + 45 min fixed (taxi+climb+descent)."""
    return round(dist_km / 800 * 60 + 45)


def drive_time_min(dist_km):
    """Estimate drive time at avg 80 km/h."""
    return round(dist_km / 80 * 60)


# ─── Build graph ─────────────────────────────────────────────────────────────

def build_graph(airports, routes):
    edges = []
    for r in routes:
        src, dst = r["src"], r["dst"]
        a, b = airports[src], airports[dst]
        dist = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        flight_min = flight_time_min(dist)
        edges.append({
            "src": src, "dst": dst,
            "dist_km": round(dist),
            "flight_time_min": flight_min,
            "mode": "flight",
            "airline": r["airline"],
        })
    return edges


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading US airports...")
    airports = load_us_airports()
    print(f"  {len(airports)} US airports with IATA codes")

    print("Loading US-domestic routes...")
    routes = load_us_routes(airports)
    print(f"  {len(routes)} direct domestic routes")

    print("Building flight graph...")
    edges = build_graph(airports, routes)

    # Stats
    dists = [e["dist_km"] for e in edges]
    times = [e["flight_time_min"] for e in edges]
    print(f"\n=== USMTG Flight Graph ===")
    print(f"Nodes (airports): {len(airports)}")
    print(f"Edges (direct routes): {len(edges)}")
    print(f"Avg route distance: {sum(dists)/len(dists):.0f} km")
    print(f"Avg flight time: {sum(times)/len(times):.0f} min")
    print(f"Shortest route: {min(dists)} km")
    print(f"Longest route: {max(dists)} km")

    # Degree (number of direct connections per airport)
    degree = defaultdict(int)
    for e in edges:
        degree[e["src"]] += 1
        degree[e["dst"]] += 1
    top10 = sorted(degree.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop 10 hub airports (by direct connections):")
    for iata, deg in top10:
        print(f"  {iata} ({airports[iata]['city']:20s}) — {deg} routes")

    # Save
    with open(f"{DATA_DIR}/us_airports.json", "w") as f:
        json.dump(airports, f, ensure_ascii=False, indent=2)
    with open(f"{DATA_DIR}/us_routes.json", "w") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: data/us_airports.json, data/us_routes.json")
