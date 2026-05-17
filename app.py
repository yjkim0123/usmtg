"""
USMTG Web App — Flask backend (v4: ML-enhanced)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from meeting_finder_v4 import find_meeting_v4
from meeting_finder_v2 import load, resolve_location, fmt_time
from ml_models import get_ml_suite
from flight_api import get_fare, is_configured as amadeus_configured

app = Flask(__name__)

# Load data and ML suite once at startup (loads from cache if available)
print("Loading data...", flush=True)
airports, graph = load()
print("Loading ML suite...", flush=True)
ml = get_ml_suite(airports, graph, verbose=False)
print("Ready.", flush=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    cities      = data.get("cities", [])
    max_stops   = int(data.get("max_stops", 2))
    dep_after   = float(data.get("depart_after_h", 0))
    arr_before  = float(data.get("arrive_before_h", 24))
    travel_date = data.get("travel_date", "")   # ISO date e.g. "2026-06-01"
    top_k       = 10

    if len(cities) < 2:
        return jsonify({"error": "Please enter at least 2 cities"}), 400

    origins = []
    for city in cities:
        city = city.strip()
        if not city:
            continue
        loc, name = resolve_location(city, airports)
        if not loc:
            return jsonify({"error": f"Cannot find location: '{city}'"}), 400
        origins.append({"lat": loc[0], "lon": loc[1], "name": name})

    if len(origins) < 2:
        return jsonify({"error": "Please enter at least 2 cities"}), 400

    origin_tuples = [(o["lat"], o["lon"], o["name"]) for o in origins]
    best, top, pareto = find_meeting_v4(
        origin_tuples, airports, graph, ml,
        max_stops=max_stops,
        top_k=top_k,
        depart_after_h=dep_after,
        arrive_before_h=arr_before,
    )

    if not best:
        return jsonify({"error": "No feasible meeting point found."}), 404

    def fmt_row(r):
        labels = [chr(65+i) for i in range(len(origins))]
        travelers = []
        for i, lbl in enumerate(labels):
            travelers.append({
                "label":   lbl,
                "time":    fmt_time(r["times_min"][i]),
                "fare":    r["fares_usd"][i],
                "co2":     r["co2_kg"][i],
                "stops":   r["stops"][i] - 1,
            })
        return {
            "airport":    r["airport"],
            "city":       r["city"],
            "name":       r["name"],
            "lat":        r["lat"],
            "lon":        r["lon"],
            "max_time":   fmt_time(r["max_time_min"]),
            "max_time_min": r["max_time_min"],
            "total_cost": r["total_cost_usd"],
            "total_co2":  r["total_co2_kg"],
            "imbalance":  r["imbalance_min"],
            "nash":       r["nash_score"],
            "ks":         r["ks_score"],
            "p90":        fmt_time(r["p90_min"]),
            "miss_pct":   round(r["miss_prob"] * 100, 1),
            "emb_sim":    r.get("emb_sim", 0.0),
            "travelers":  travelers,
        }

    # Optionally augment best result's fares with real Amadeus prices
    using_real_fares = False
    if amadeus_configured() and travel_date and best:
        near_iatas = []
        from meeting_finder_v2 import nearby_airports
        for o in origin_tuples:
            nr = nearby_airports(o[0], o[1], airports)
            near_iatas.append(nr[0][0] if nr else None)
        real_fares = []
        for dep_iata in near_iatas:
            if dep_iata:
                f = get_fare(dep_iata, best['airport'], travel_date)
                real_fares.append(f)
            else:
                real_fares.append(None)
        if any(f is not None for f in real_fares):
            for i, f in enumerate(real_fares):
                if f is not None:
                    best['fares_usd'][i] = f
            best['total_cost_usd'] = sum(f for f in best['fares_usd'])
            using_real_fares = True

    return jsonify({
        "origins":          origins,
        "best":             fmt_row(best),
        "top":              [fmt_row(r) for r in top],
        "pareto":           [fmt_row(r) for r in pareto],
        "using_real_fares": using_real_fares,
        "amadeus_active":   amadeus_configured(),
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5050)
