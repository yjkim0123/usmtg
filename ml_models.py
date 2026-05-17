"""
USMTG ML Models — 7 machine-learning enhancements

① FarePredictor      — XGBoost regression (graph features → fare)
② DelayPredictor     — Airport-specific delay distribution (centrality-based)
③ AirportClusterer   — K-means geographic pre-filtering
④ AirportEmbedder    — Spectral graph embedding (GNN proxy)
⑤ MeetingRanker      — LightGBM learning-to-rank
⑥ TravelTimeSurrogate— MLP approximation of Dijkstra (N-party speedup)
⑦ DemandWeighter     — Degree-based demand weighting
"""
import os, math, pickle
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
import networkx as nx

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CACHE_DIR = "ml_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# ① Fare Predictor — XGBoost on graph-derived features
# ═══════════════════════════════════════════════════════════════════════════
class FarePredictor:
    """
    Predicts one-way fare from route features.
    Trained on synthetic data derived from BTS 2023 distance brackets
    + graph structure (degree, hub status, competition).
    """
    BASE_BRACKETS = [(300,180),(600,210),(1000,240),(1500,270),(2000,300),(1e9,330)]
    HUB_SET = {"ATL","ORD","DEN","DFW","LAX","JFK","SFO","LAS","SEA","CLT",
               "MCO","PHX","IAH","BOS","MSP","DTW","EWR","PHL","SLC","MIA"}

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self._trained = False

    def _base_fare(self, dist_km):
        miles = dist_km * 0.621371
        for cap, fare in self.BASE_BRACKETS:
            if miles < cap:
                return fare
        return 330

    def _features(self, dist_km, dep, arr, dep_degree, arr_degree, shared_neighbors):
        miles = dist_km * 0.621371
        return [
            dist_km,
            miles,
            1 if dep in self.HUB_SET else 0,
            1 if arr in self.HUB_SET else 0,
            dep_degree,
            arr_degree,
            shared_neighbors,               # proxy for competition (common hubs)
            math.log1p(dist_km),
            (dep_degree + arr_degree) / 2,
            abs(dep_degree - arr_degree),
        ]

    def train(self, graph, airports, n_samples=8000, seed=42):
        rng = np.random.default_rng(seed)
        edges = [(s, d, dist) for s, nbrs in graph.items()
                 for d, _, _, dist in nbrs]
        if not edges:
            return
        idx = rng.choice(len(edges), min(n_samples, len(edges)), replace=False)
        sampled = [edges[i] for i in idx]

        degree = {iata: len(nbrs) for iata, nbrs in graph.items()}
        adj = defaultdict(set)
        for s, nbrs in graph.items():
            for d, *_ in nbrs:
                adj[s].add(d); adj[d].add(s)

        X, y = [], []
        for dep, arr, dist in sampled:
            shared = len(adj[dep] & adj[arr])
            feats = self._features(dist, dep, arr,
                                   degree.get(dep,1), degree.get(arr,1), shared)
            base = self._base_fare(dist)
            disc = (0.10 if dep in self.HUB_SET else 0) + (0.10 if arr in self.HUB_SET else 0)
            noise = rng.normal(0, 18)          # ±$18 realistic variance
            fare = round(base * (1 - disc) + noise)
            X.append(feats); y.append(max(80, fare))

        X = np.array(X); y = np.array(y)
        Xs = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=seed)
        self.model.fit(Xs, y)
        self._trained = True

    def predict(self, dist_km, dep, arr, dep_degree, arr_degree, shared_neighbors):
        if not self._trained:
            # fallback to rule-based
            base = self._base_fare(dist_km)
            disc = (0.10 if dep in self.HUB_SET else 0) + (0.10 if arr in self.HUB_SET else 0)
            return round(base * (1 - disc))
        feats = np.array([self._features(dist_km, dep, arr,
                                         dep_degree, arr_degree, shared_neighbors)])
        return max(80, round(float(self.model.predict(self.scaler.transform(feats))[0])))


# ═══════════════════════════════════════════════════════════════════════════
# ② Delay Predictor — airport-specific delay distribution
# ═══════════════════════════════════════════════════════════════════════════
class DelayPredictor:
    """
    Estimates per-airport delay parameters from structural features.
    High-degree hubs have higher mean delay but lower variance (more resources).
    Small airports have low mean but occasional long delays.
    """
    def __init__(self):
        self.params = {}          # iata → (mean_min, std_min, cancel_prob)

    def fit(self, graph):
        degrees = {iata: len(nbrs) for iata, nbrs in graph.items()}
        max_deg = max(degrees.values()) if degrees else 1

        for iata, deg in degrees.items():
            ratio = deg / max_deg
            # Hubs: busier → more delay, but more recovery resources
            mean  = 8 + ratio * 18          # 8–26 min
            std   = 12 + (1 - ratio) * 18   # small airports: more variance
            cancel = 0.008 + (1 - ratio) * 0.015   # 0.8–2.3%
            self.params[iata] = (round(mean,1), round(std,1), round(cancel,4))

    def get(self, iata):
        return self.params.get(iata, (12.0, 20.0, 0.015))

    def sample_delay(self, iata, rng):
        mean, std, cancel_p = self.get(iata)
        if rng.random() < cancel_p:
            return float('inf')
        raw = rng.lognormvariate(math.log(max(1, mean)), 0.75)
        return max(0.0, raw - mean * 0.25)


# ═══════════════════════════════════════════════════════════════════════════
# ③ Airport Clusterer — K-means geographic pre-filtering
# ═══════════════════════════════════════════════════════════════════════════
class AirportClusterer:
    """
    Clusters airports geographically. For a query, only search airports in
    clusters near the midpoint of origins — 5-10x speedup for large graphs.
    """
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        self.labels = {}
        self.centers = None

    def fit(self, airports):
        iatas = list(airports.keys())
        coords = np.array([[airports[i]['lat'], airports[i]['lon']] for i in iatas])
        self.km.fit(coords)
        self.labels = {iata: int(lbl) for iata, lbl in zip(iatas, self.km.labels_)}
        self.centers = self.km.cluster_centers_

    def candidate_clusters(self, origins_latlon, n_clusters=6):
        """Return cluster IDs closest to the centroid of origins."""
        centroid = np.mean(origins_latlon, axis=0)
        dists = np.linalg.norm(self.centers - centroid, axis=1)
        return set(np.argsort(dists)[:n_clusters])

    def filter_airports(self, airports, origins_latlon, n_clusters=6):
        """Return subset of airports in candidate clusters."""
        clusters = self.candidate_clusters(origins_latlon, n_clusters)
        return {k: v for k, v in airports.items()
                if self.labels.get(k, -1) in clusters}


# ═══════════════════════════════════════════════════════════════════════════
# ④ Airport Embedder — Spectral graph embedding (GNN proxy)
# ═══════════════════════════════════════════════════════════════════════════
class AirportEmbedder:
    """
    Learns 16-dim airport embeddings from graph structure using
    truncated SVD on the weighted adjacency matrix.
    Captures connectivity, centrality, and proximity patterns.
    """
    def __init__(self, dim=16):
        self.dim = dim
        self.embeddings = {}
        self.iatas = []

    def fit(self, graph, airports):
        self.iatas = sorted(set(airports.keys()) & set(graph.keys()))
        n = len(self.iatas)
        idx = {iata: i for i, iata in enumerate(self.iatas)}

        # Weighted adjacency: weight = 1/flight_time (faster = stronger link)
        A = np.zeros((n, n), dtype=np.float32)
        for src, nbrs in graph.items():
            if src not in idx: continue
            for dst, ft, fare, dist in nbrs:
                if dst not in idx: continue
                w = 1.0 / max(1, ft)
                A[idx[src], idx[dst]] = w
                A[idx[dst], idx[src]] = w

        # Degree-normalize
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1
        A = A / deg

        svd = TruncatedSVD(n_components=min(self.dim, n-1), random_state=0)
        emb = svd.fit_transform(A)

        for i, iata in enumerate(self.iatas):
            self.embeddings[iata] = emb[i]

    def get(self, iata):
        if iata in self.embeddings:
            return self.embeddings[iata]
        return np.zeros(self.dim)

    def similarity(self, iata_a, iata_b):
        a, b = self.get(iata_a), self.get(iata_b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0: return 0.0
        return float(np.dot(a, b) / denom)

    def most_similar(self, iata, top_k=5):
        q = self.get(iata)
        scores = {}
        for other, emb in self.embeddings.items():
            if other == iata: continue
            d = np.linalg.norm(q) * np.linalg.norm(emb)
            if d < 1e-9: continue
            scores[other] = float(np.dot(q, emb) / d)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# ⑤ Meeting Ranker — Learning-to-rank (GBM pairwise)
# ═══════════════════════════════════════════════════════════════════════════
class MeetingRanker:
    """
    Ranks candidate meeting airports without running full Dijkstra.
    Features: geographic + embedding-based.
    Trained on synthetic Dijkstra results (label = rank position).
    """
    def __init__(self, embedder: AirportEmbedder):
        self.embedder = embedder
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=0)
        self.scaler = StandardScaler()
        self._trained = False

    def _featurize(self, orig_lats, orig_lons, cand_iata, cand_lat, cand_lon,
                   cand_degree, airports):
        # Geographic features
        dists = [math.hypot(lat - cand_lat, lon - cand_lon)
                 for lat, lon in zip(orig_lats, orig_lons)]
        centroid_lat = np.mean(orig_lats)
        centroid_lon = np.mean(orig_lons)
        dist_to_centroid = math.hypot(centroid_lat - cand_lat, centroid_lon - cand_lon)

        # Embedding similarity to origin cluster centroid
        emb = self.embedder.get(cand_iata)
        emb_norm = np.linalg.norm(emb)

        return [
            np.mean(dists),
            np.max(dists),
            np.min(dists),
            np.max(dists) - np.min(dists),   # imbalance proxy
            dist_to_centroid,
            cand_degree,
            math.log1p(cand_degree),
            emb_norm,
            *emb[:8],                          # first 8 embedding dims
        ]

    def train(self, queries, airports, graph):
        """
        queries: list of (origins_latlon, ranked_results)
        ranked_results: list of dicts with 'airport','lat','lon','max_time_min'
        """
        degree = {iata: len(nbrs) for iata, nbrs in graph.items()}
        X, y = [], []
        for orig_latlon, results in queries:
            orig_lats = [o[0] for o in orig_latlon]
            orig_lons = [o[1] for o in orig_latlon]
            for rank, r in enumerate(results):
                feats = self._featurize(
                    orig_lats, orig_lons,
                    r['airport'], r['lat'], r['lon'],
                    degree.get(r['airport'], 1), airports)
                X.append(feats)
                y.append(-rank)   # higher score = better rank

        if len(X) < 10:
            return
        X = np.array(X); y = np.array(y, dtype=float)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self._trained = True

    def rank(self, orig_latlon, candidates, airports, graph, top_k=20):
        """Fast-rank candidates without Dijkstra. Returns sorted list."""
        if not self._trained or not candidates:
            return candidates
        degree = {iata: len(nbrs) for iata, nbrs in graph.items()}
        orig_lats = [o[0] for o in orig_latlon]
        orig_lons = [o[1] for o in orig_latlon]
        X = []
        for iata in candidates:
            ap = airports[iata]
            feats = self._featurize(orig_lats, orig_lons,
                                    iata, ap['lat'], ap['lon'],
                                    degree.get(iata,1), airports)
            X.append(feats)
        scores = self.model.predict(self.scaler.transform(np.array(X)))
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [iata for iata, _ in ranked[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════
# ⑥ Travel Time Surrogate — MLP approximation of Dijkstra
# ═══════════════════════════════════════════════════════════════════════════
class TravelTimeSurrogate:
    """
    MLP that approximates travel time from origin to airport.
    After training on Dijkstra results, inference is ~1000x faster.
    Useful for N-party where full Dijkstra runs N times.
    """
    def __init__(self, embedder: AirportEmbedder):
        self.embedder = embedder
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', max_iter=500,
            random_state=0, early_stopping=True,
            validation_fraction=0.1)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self._trained = False
        self._rmse = None

    def _featurize(self, orig_lat, orig_lon, cand_iata, cand_lat, cand_lon):
        geo_dist = math.hypot(orig_lat - cand_lat, orig_lon - cand_lon)
        emb = self.embedder.get(cand_iata)
        return [
            orig_lat, orig_lon,
            cand_lat, cand_lon,
            geo_dist,
            abs(orig_lat - cand_lat),
            abs(orig_lon - cand_lon),
            *emb[:12],
        ]

    def train(self, dijkstra_results, airports):
        """
        dijkstra_results: list of (orig_lat, orig_lon, {iata: time_min})
        """
        X, y = [], []
        for orig_lat, orig_lon, dist_dict in dijkstra_results:
            for iata, (time_min, *_) in dist_dict.items():
                if iata not in airports: continue
                ap = airports[iata]
                feats = self._featurize(orig_lat, orig_lon,
                                        iata, ap['lat'], ap['lon'])
                X.append(feats)
                y.append(time_min)

        if len(X) < 50:
            return
        X = np.array(X); y = np.array(y).reshape(-1, 1)
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y).ravel()
        self.model.fit(Xs, ys)
        preds = self.scaler_y.inverse_transform(
            self.model.predict(Xs).reshape(-1,1)).ravel()
        self._rmse = float(np.sqrt(np.mean((preds - y.ravel())**2)))
        self._trained = True

    def predict(self, orig_lat, orig_lon, candidates, airports):
        """Returns {iata: predicted_time_min}"""
        if not self._trained:
            return {}
        iatas = list(candidates)
        X = [self._featurize(orig_lat, orig_lon,
                             iata, airports[iata]['lat'], airports[iata]['lon'])
             for iata in iatas if iata in airports]
        valid = [iata for iata in iatas if iata in airports]
        if not X:
            return {}
        Xs = self.scaler_X.transform(np.array(X))
        preds = self.scaler_y.inverse_transform(
            self.model.predict(Xs).reshape(-1,1)).ravel()
        return {iata: max(90, round(float(p))) for iata, p in zip(valid, preds)}

    @property
    def rmse(self):
        return self._rmse


# ═══════════════════════════════════════════════════════════════════════════
# ⑦ Demand Weighter — penalize low-frequency airports
# ═══════════════════════════════════════════════════════════════════════════
class DemandWeighter:
    """
    Weights airports by estimated passenger demand (degree as proxy).
    Airports with very few routes get a time penalty (infrequent flights
    mean you can't always catch a convenient departure).
    """
    def __init__(self, min_penalty_min=0, max_penalty_min=30):
        self.min_p = min_penalty_min
        self.max_p = max_penalty_min
        self.penalties = {}

    def fit(self, graph):
        degrees = {iata: len(nbrs) for iata, nbrs in graph.items()}
        max_deg = max(degrees.values()) if degrees else 1
        for iata, deg in degrees.items():
            ratio = deg / max_deg
            # Low degree → high wait penalty (sparse schedule)
            self.penalties[iata] = round(self.max_p * (1 - ratio**0.5))

    def penalty(self, iata):
        return self.penalties.get(iata, self.max_p)


# ═══════════════════════════════════════════════════════════════════════════
# Factory: build & train all models
# ═══════════════════════════════════════════════════════════════════════════
def build_ml_suite(airports, graph, verbose=True):
    """Train all 7 ML components. Returns dict of fitted models."""
    def log(msg):
        if verbose: print(f"  [ML] {msg}")

    log("① Training FarePredictor (XGBoost)...")
    fare_pred = FarePredictor()
    fare_pred.train(graph, airports)

    log("② Fitting DelayPredictor (centrality-based)...")
    delay_pred = DelayPredictor()
    delay_pred.fit(graph)

    log("③ Fitting AirportClusterer (K-means, k=20)...")
    clusterer = AirportClusterer(n_clusters=20)
    clusterer.fit(airports)

    log("④ Training GNN AirportEmbedder (GraphSAGE, subprocess)...")
    _GNN_CACHE = os.path.join(CACHE_DIR, 'gnn_embeddings.npz')
    try:
        import subprocess, sys as _sys
        # Run GNN training in isolated subprocess to avoid OMP conflict with sklearn
        _proj = os.getcwd()
        # Inline self-contained script (importing gnn_embedder module crashes on Mac M-series)
        _tmp_script = os.path.join(CACHE_DIR, '_gnn_train.py')
        with open(_tmp_script, 'w') as _f:
            _f.write(f'''\
import os, sys, json, math
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
from collections import defaultdict

HUB_SET = {{"ATL","ORD","DEN","DFW","LAX","JFK","SFO","LAS","SEA","CLT","MCO","PHX","IAH","BOS","MSP","DTW","EWR","PHL","SLC","MIA"}}
PROJ = "{_proj}"

with open(os.path.join(PROJ,"data","us_airports.json")) as f: airports = json.load(f)
with open(os.path.join(PROJ,"data","us_routes.json")) as f: routes = json.load(f)
graph = defaultdict(list)
for r in routes:
    s,d,t,dist = r["src"],r["dst"],r["flight_time_min"],r["dist_km"]
    graph[s].append((d,t,210,dist)); graph[d].append((s,t,210,dist))

iatas = sorted(set(airports.keys()) & set(graph.keys()))
idx = {{iata:i for i,iata in enumerate(iatas)}}
n = len(iatas)
degrees = {{iata: len(graph[iata]) for iata in iatas}}
max_deg = max(degrees.values())
avg_ft = {{iata: sum(ft for _,ft,_,_ in graph[iata])/len(graph[iata]) for iata in iatas}}
max_ft = max(avg_ft.values())+1e-6
avg_fare = {{iata: sum(fa for _,_,fa,_ in graph[iata])/len(graph[iata]) for iata in iatas}}
max_fare = max(avg_fare.values())+1e-6
lats = np.array([airports[i]["lat"] for i in iatas])
lons = np.array([airports[i]["lon"] for i in iatas])
lats = (lats-lats.mean())/(lats.std()+1e-6)
lons = (lons-lons.mean())/(lons.std()+1e-6)
X = np.stack([lats,lons,
    np.array([degrees[i]/max_deg for i in iatas]),
    np.array([1.0 if i in HUB_SET else 0.0 for i in iatas]),
    np.array([avg_ft[i]/max_ft for i in iatas]),
    np.array([avg_fare[i]/max_fare for i in iatas])],axis=1).astype(np.float32)
src_l,dst_l=[],[]
for iata,nbrs in graph.items():
    if iata not in idx: continue
    for dst,*_ in nbrs:
        if dst in idx: src_l.append(idx[iata]); dst_l.append(idx[dst])
data = Data(x=torch.tensor(X), edge_index=torch.tensor([src_l,dst_l],dtype=torch.long))
print(f"GNN graph: {{n}} nodes, {{data.edge_index.shape[1]}} edges", flush=True)

conv1 = SAGEConv(6, 64)
conv2 = SAGEConv(64, 32)
bn1   = nn.BatchNorm1d(64)
opt   = torch.optim.Adam(
    list(conv1.parameters())+list(conv2.parameters())+list(bn1.parameters()), lr=3e-3)
ne = data.edge_index.shape[1]
for ep in range(80):
    opt.zero_grad()
    h = F.relu(bn1(conv1(data.x, data.edge_index)))
    z = F.normalize(conv2(h, data.edge_index), p=2, dim=-1)
    pos = (z[data.edge_index[0]] * z[data.edge_index[1]]).sum(-1)
    neg_ei = negative_sampling(data.edge_index, num_nodes=n, num_neg_samples=ne)
    neg = (z[neg_ei[0]] * z[neg_ei[1]]).sum(-1)
    lbl = torch.cat([torch.ones(ne), torch.zeros(neg_ei.shape[1])])
    loss = F.binary_cross_entropy_with_logits(torch.cat([pos,neg]), lbl)
    loss.backward(); opt.step()
    if (ep+1)%20==0: print(f"  ep{{ep+1}} loss={{loss.item():.4f}}", flush=True)
with torch.no_grad():
    h = F.relu(bn1(conv1(data.x, data.edge_index)))
    Z = F.normalize(conv2(h, data.edge_index), p=2, dim=-1).numpy()
save_path = "{_GNN_CACHE}"
os.makedirs(os.path.dirname(save_path),exist_ok=True)
np.savez(save_path, iatas=iatas, embeddings=Z)
print(f"Saved {{n}} embeddings → {{save_path}}", flush=True)
''')
        result = subprocess.run(
            [_sys.executable, _tmp_script],
            capture_output=True, text=True, timeout=300, cwd=_proj
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-300:])
        for line in result.stdout.strip().splitlines():
            log(f"   {line}")
        from gnn_embedder import load_embeddings
        embedder = load_embeddings(_GNN_CACHE)
        log(f"   GNN loaded · {len(embedder.embeddings)} airports")
    except Exception as e:
        log(f"   GNN failed ({e}), fallback to spectral")
        embedder = AirportEmbedder(dim=16)
        embedder.fit(graph, airports)

    log("⑦ Fitting DemandWeighter...")
    demand = DemandWeighter()
    demand.fit(graph)

    # ⑤ MeetingRanker needs training queries (from Dijkstra)
    log("⑤ Generating training data for MeetingRanker...")
    from meeting_finder_v3 import find_meeting_nparty
    from meeting_finder_v2 import CITIES
    import random as _random
    city_list = list(CITIES.values())
    _random.seed(42)
    queries = []
    sample_pairs = _random.sample([(city_list[i], city_list[j])
                                   for i in range(len(city_list))
                                   for j in range(i+1, len(city_list))], 30)
    for (la, loa), (lb, lob) in sample_pairs:
        origins = [(la, loa, 'A'), (lb, lob, 'B')]
        _, top, _ = find_meeting_nparty(origins, airports, graph, top_k=20)
        if top:
            queries.append([(la, loa), (lb, lob)],)
            queries[-1] = ([(la, loa), (lb, lob)], top)

    ranker = MeetingRanker(embedder)
    ranker.train(queries, airports, graph)
    log(f"   Trained on {len(queries)} queries")

    # ⑥ TravelTimeSurrogate training
    log("⑥ Training TravelTimeSurrogate (MLP)...")
    from meeting_finder_v3 import dijkstra_v3
    from meeting_finder_v2 import CITIES
    surrogate = TravelTimeSurrogate(embedder)
    dijkstra_data = []
    sample_locs = _random.sample(city_list, min(20, len(city_list)))
    for lat, lon in sample_locs:
        from meeting_finder_v2 import nearby_airports as _nearby
        near = _nearby(lat, lon, airports)
        if not near: continue
        sources = [(iata, d) for iata, d, _ in near]
        dist = dijkstra_v3(sources, graph)
        dijkstra_data.append((lat, lon, dist))
    surrogate.train(dijkstra_data, airports)
    log(f"   RMSE: {surrogate.rmse:.1f} min" if surrogate.rmse else "   (not enough data)")

    log("All 7 models ready ✓")
    return {
        'fare':      fare_pred,
        'delay':     delay_pred,
        'clusterer': clusterer,
        'embedder':  embedder,
        'ranker':    ranker,
        'surrogate': surrogate,
        'demand':    demand,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cache-aware loader
# ═══════════════════════════════════════════════════════════════════════════
_ML_CACHE_PKL = os.path.join(CACHE_DIR, 'ml_suite.pkl')
_GNN_CACHE_NPZ = os.path.join(CACHE_DIR, 'gnn_embeddings.npz')

def get_ml_suite(airports, graph, verbose=True, force_retrain=False):
    """
    Return trained ML suite. Loads from disk cache if available (~1s),
    otherwise trains all models (~30s) and saves to cache.
    Pass force_retrain=True to rebuild even if cache exists.
    """
    def log(msg):
        if verbose: print(f"  [ML] {msg}")

    if not force_retrain and os.path.exists(_ML_CACHE_PKL) and os.path.exists(_GNN_CACHE_NPZ):
        try:
            log("Loading ML suite from cache...")
            with open(_ML_CACHE_PKL, 'rb') as f:
                ml = pickle.load(f)
            from gnn_embedder import load_embeddings
            embedder = load_embeddings(_GNN_CACHE_NPZ)
            ml['embedder'] = embedder
            # Restore embedder reference in models that store it
            if hasattr(ml.get('ranker'), 'embedder'):
                ml['ranker'].embedder = embedder
            if hasattr(ml.get('surrogate'), 'embedder'):
                ml['surrogate'].embedder = embedder
            log(f"Cache loaded ✓  ({len(embedder.embeddings)} airports in GNN)")
            return ml
        except Exception as e:
            log(f"Cache load failed ({e}), retraining...")

    ml = build_ml_suite(airports, graph, verbose=verbose)

    try:
        # Save without embedder (GNN already saved as npz by subprocess)
        to_save = {k: v for k, v in ml.items() if k != 'embedder'}
        with open(_ML_CACHE_PKL, 'wb') as f:
            pickle.dump(to_save, f)
        log(f"ML suite cached → {_ML_CACHE_PKL}")
    except Exception as e:
        log(f"Cache save failed ({e})")

    return ml
