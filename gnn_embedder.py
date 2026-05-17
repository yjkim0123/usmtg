"""
USMTG GNN Airport Embedder
GraphSAGE with link-prediction training (unsupervised).

Node features (6-dim):
  lat, lon, degree, is_hub, avg_flight_time, avg_fare

Trained with: positive edges vs. negative-sampled pairs
Output: 32-dim airport embedding per node
"""
import os, math
import numpy as np

# torch/torch_geometric only needed for training — not for load_embeddings()
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Provide stubs so class definitions below parse without error.
    # These classes are only used for training (subprocess), never at inference.
    class _Stub:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return self
        def parameters(self): return iter([])
        def to(self, *a, **kw): return self
        def train(self): return self
        def eval(self): return self
        def load_state_dict(self, *a): pass
        def state_dict(self): return {}
        def zero_grad(self): pass
        def step(self): pass
        def backward(self): pass
    class _F:
        @staticmethod
        def relu(x): return x
        @staticmethod
        def dropout(x, p=0, training=False): return x
        @staticmethod
        def normalize(x, p=2, dim=-1): return x
        @staticmethod
        def binary_cross_entropy_with_logits(*a, **kw): return _Stub()
    class _NN:
        Module = object
        BatchNorm1d = _Stub
        @staticmethod
        def utils(): pass
    class _Torch:
        device = str
        @staticmethod
        def tensor(*a, **kw): return None
        @staticmethod
        def cat(*a, **kw): return None
        @staticmethod
        def ones(*a, **kw): return None
        @staticmethod
        def zeros(*a, **kw): return None
        @staticmethod
        def randint(*a, **kw): return None
        @staticmethod
        def stack(*a, **kw): return None
        @staticmethod
        def no_grad():
            import contextlib
            return contextlib.nullcontext()
        class optim:
            class Adam(_Stub): pass
            class lr_scheduler:
                class CosineAnnealingLR(_Stub): pass
    nn = _NN()
    F = _F()
    torch = _Torch()
    SAGEConv = _Stub
    Data = _Stub
    def negative_sampling(*a, **kw): return None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

HUB_SET = {
    "ATL","ORD","DEN","DFW","LAX","JFK","SFO","LAS","SEA","CLT",
    "MCO","PHX","IAH","BOS","MSP","DTW","EWR","PHL","SLC","MIA",
}

# ── GraphSAGE model ────────────────────────────────────────────────────────
class AirportSAGE(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.normalize(x, p=2, dim=-1)   # L2-normalize for cosine similarity

    def decode(self, z, edge_index):
        """Dot-product decoder for link prediction."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


# ── Build PyG Data object ──────────────────────────────────────────────────
def build_pyg_data(airports, graph):
    iatas = sorted(set(airports.keys()) & set(graph.keys()))
    idx   = {iata: i for i, iata in enumerate(iatas)}
    n     = len(iatas)

    # Node features
    degrees   = {iata: len(graph[iata]) for iata in iatas}
    max_deg   = max(degrees.values()) if degrees else 1

    avg_ft    = {}
    avg_fare  = {}
    for iata in iatas:
        nbrs = graph[iata]
        if nbrs:
            avg_ft[iata]   = np.mean([ft for _, ft, _, _ in nbrs])
            avg_fare[iata] = np.mean([fare for _, _, fare, _ in nbrs])
        else:
            avg_ft[iata]   = 0.0
            avg_fare[iata] = 0.0

    lats  = np.array([airports[i]['lat'] for i in iatas])
    lons  = np.array([airports[i]['lon'] for i in iatas])
    lats  = (lats - lats.mean()) / (lats.std() + 1e-6)
    lons  = (lons - lons.mean()) / (lons.std() + 1e-6)

    max_ft   = max(avg_ft.values()) + 1e-6
    max_fare = max(avg_fare.values()) + 1e-6

    X = np.stack([
        lats,
        lons,
        np.array([degrees[i] / max_deg for i in iatas]),
        np.array([1.0 if i in HUB_SET else 0.0 for i in iatas]),
        np.array([avg_ft[i]   / max_ft   for i in iatas]),
        np.array([avg_fare[i] / max_fare for i in iatas]),
    ], axis=1).astype(np.float32)

    # Edges (bidirectional)
    src_list, dst_list = [], []
    for iata, nbrs in graph.items():
        if iata not in idx: continue
        for dst, *_ in nbrs:
            if dst in idx:
                src_list.append(idx[iata])
                dst_list.append(idx[dst])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x          = torch.tensor(X, dtype=torch.float)

    return Data(x=x, edge_index=edge_index), iatas, idx


# ── Main GNN Embedder class ────────────────────────────────────────────────
class GNNEmbedder:
    """
    Drop-in replacement for AirportEmbedder (spectral) in ml_models.py.
    Uses GraphSAGE trained with unsupervised link prediction.
    """
    def __init__(self, hidden_dim=64, out_dim=32, epochs=80, lr=3e-3):
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.epochs     = epochs
        self.lr         = lr
        self.model      = None
        self.embeddings = {}
        self.iatas      = []
        self.device     = torch.device('cpu')   # MPS unstable with PyG negative_sampling

    def fit(self, graph, airports, verbose=True):
        data, self.iatas, idx = build_pyg_data(airports, graph)
        data = data.to(self.device)

        self.model = AirportSAGE(
            in_dim=data.x.shape[1],
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim
        ).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs)

        n_edges = data.edge_index.shape[1]
        best_loss = float('inf')
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            opt.zero_grad()
            z = self.model(data.x, data.edge_index)

            # Positive edges
            pos_score = self.model.decode(z, data.edge_index)

            # Random negative sampling (avoids PyG negative_sampling segfault on M-series)
            neg_src = torch.randint(0, data.num_nodes, (n_edges,), device=self.device)
            neg_dst = torch.randint(0, data.num_nodes, (n_edges,), device=self.device)
            neg_ei  = torch.stack([neg_src, neg_dst])
            neg_score = self.model.decode(z, neg_ei)

            labels = torch.cat([
                torch.ones(n_edges, device=self.device),
                torch.zeros(n_edges, device=self.device)
            ])
            scores = torch.cat([pos_score, neg_score])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 20 == 0:
                print(f"     epoch {epoch+1:3d}/{self.epochs}  loss={loss.item():.4f}")

        # Load best model and extract embeddings
        self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            z = self.model(data.x, data.edge_index).cpu().numpy()

        self.embeddings = {iata: z[i] for i, iata in enumerate(self.iatas)}
        if verbose:
            print(f"     GNN trained · best loss={best_loss:.4f} · "
                  f"device={self.device} · emb_dim={self.out_dim}")

    def get(self, iata):
        return self.embeddings.get(iata, np.zeros(self.out_dim))

    def similarity(self, iata_a, iata_b):
        a, b = self.get(iata_a), self.get(iata_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-9: return 0.0
        return float(np.dot(a, b) / denom)

    def most_similar(self, iata, top_k=5):
        """Return top-k most similar airports by embedding cosine similarity."""
        q = self.get(iata)
        scores = {}
        for other, emb in self.embeddings.items():
            if other == iata: continue
            denom = np.linalg.norm(q) * np.linalg.norm(emb)
            if denom < 1e-9: continue
            scores[other] = float(np.dot(q, emb) / denom)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ── Subprocess entry-point: train and save embeddings to disk ─────────────
def train_and_save(project_dir, save_path):
    """Called as subprocess to avoid OMP conflict with sklearn."""
    import json
    from collections import defaultdict
    with open(os.path.join(project_dir, 'data', 'us_airports.json')) as f:
        airports = json.load(f)
    with open(os.path.join(project_dir, 'data', 'us_routes.json')) as f:
        routes = json.load(f)
    graph = defaultdict(list)
    for r in routes:
        s,d,t,dist = r['src'],r['dst'],r['flight_time_min'],r['dist_km']
        graph[s].append((d,t,210,dist))
        graph[d].append((s,t,210,dist))

    gnn = GNNEmbedder(epochs=80)
    gnn.fit(graph, airports, verbose=True)

    iatas = list(gnn.embeddings.keys())
    embs  = np.stack([gnn.embeddings[i] for i in iatas])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, iatas=iatas, embeddings=embs)
    print(f"Saved {len(iatas)} embeddings → {save_path}", flush=True)


class LoadedEmbedder:
    """Lightweight numpy-only embedder loaded from a saved .npz file. Picklable."""
    def __init__(self, iatas, embs, out_dim=32):
        self.out_dim = out_dim
        self.embeddings = {iata: embs[i] for i, iata in enumerate(iatas)}

    def get(self, iata):
        return self.embeddings.get(iata, np.zeros(self.out_dim))

    def similarity(self, a, b):
        ea, eb = self.get(a), self.get(b)
        d = np.linalg.norm(ea) * np.linalg.norm(eb)
        return float(np.dot(ea, eb) / d) if d > 1e-9 else 0.0

    def most_similar(self, iata, top_k=5):
        q = self.get(iata)
        scores = {}
        for o, e in self.embeddings.items():
            if o == iata: continue
            d = np.linalg.norm(q) * np.linalg.norm(e)
            if d < 1e-9: continue
            scores[o] = float(np.dot(q, e) / d)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def load_embeddings(save_path='ml_cache/gnn_embeddings.npz', out_dim=32):
    """Load pre-trained GNN embeddings without importing torch."""
    data = np.load(save_path, allow_pickle=True)
    iatas = list(data['iatas'])
    embs  = data['embeddings']
    return LoadedEmbedder(iatas, embs, out_dim)


# ── Standalone demo ────────────────────────────────────────────────────────
if __name__ == '__main__':
    from meeting_finder_v2 import load
    print("Loading data...")
    airports, graph = load()

    print("Training GNN (GraphSAGE, 80 epochs)...")
    gnn = GNNEmbedder(epochs=80)
    gnn.fit(graph, airports, verbose=True)

    print("\nMost similar airports to ATL (Atlanta hub):")
    for iata, sim in gnn.most_similar('ATL'):
        print(f"  {iata:4s} {airports[iata]['city']:20s} sim={sim:.4f}")

    print("\nMost similar airports to SFO (San Francisco):")
    for iata, sim in gnn.most_similar('SFO'):
        print(f"  {iata:4s} {airports[iata]['city']:20s} sim={sim:.4f}")

    print("\nSimilarity checks (hub vs hub, hub vs small):")
    pairs = [('ATL','ORD'), ('ATL','DFW'), ('ATL','LAX'),
             ('ATL','PIE'), ('JFK','EWR'), ('SEA','PDX')]
    for a, b in pairs:
        if a in airports and b in airports:
            print(f"  {a}↔{b}: {gnn.similarity(a,b):.4f}  "
                  f"({airports[a]['city']} ↔ {airports[b]['city']})")
