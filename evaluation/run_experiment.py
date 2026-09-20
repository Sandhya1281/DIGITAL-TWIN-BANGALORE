"""
Reproduces the original project's experiment logic:
  1. Train each model (BaselineGCN, AdaptiveGNN, TopologyAwareAGNN) on a
     set of "seen" zones.
  2. Zero-shot evaluate on "unseen" (held-out) zones, with NO fine-tuning.
  3. Compute MAE, RMSE, MAPE on each zone; compute "in-distribution"
     performance (avg over seen zones) vs "degradation" on unseen zones.
  4. Correlate topology-distance (seen-zone centroid -> unseen zone) with
     degradation, for each model.

Adapted for Bengaluru: cross-CITY holdout in the original -> cross-ZONE
holdout here (see the approved project plan for the rationale).
"""
import json
import sys
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, "/home/claude/bengaluru-digital-twin")
from topology.extract_topology import load_zone_graphs, build_feature_table, topology_distance_matrix
from models.gnn_models import BaselineGCN, AdaptiveGNN, TopologyAwareAGNN

RNG = np.random.RandomState(0)

TRAIN_ZONES = ["Koramangala", "SilkBoard", "MGRoad_CBD", "Indiranagar", "Jayanagar", "Malleshwaram", "Hebbal"]
TEST_ZONES = ["Whitefield", "ElectronicCity", "Yelahanka"]  # held out entirely -> zero-shot


def zone_adjacency(G):
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    return A, nodes


def make_windows(signal, lookback=8, horizon=1):
    """signal: (T, N). Returns X:(samples,N,lookback) flattened per-node feature,
    y:(samples,N) next-step target, using each node's own recent history as its
    feature vector (a minimal but faithful graph-signal forecasting setup)."""
    T, N = signal.shape
    Xs, ys = [], []
    for t in range(lookback, T - horizon):
        Xs.append(signal[t - lookback:t].T)          # (N, lookback)
        ys.append(signal[t + horizon - 1])            # (N,)
    return np.array(Xs), np.array(ys)


def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def mape(y, yhat): return float(np.mean(np.abs((y - yhat) / np.clip(y, 1e-3, None))) * 100)


def train_model_family(graphs, signals, lookback=8, epochs=40):
    """Trains one instance of each model per training zone (models are
    node-count-specific, so we train per-zone then average zero-shot
    performance the way the original per-city-trained/tested setup does),
    and returns trained models keyed by (model_name, zone)."""
    trained = {"BaselineGCN": {}, "AdaptiveGNN": {}, "TopologyAwareAGNN": {}}
    train_metrics = []

    for zone in TRAIN_ZONES:
        G = graphs[zone]
        A, nodes = zone_adjacency(G)
        sig = np.array(signals[zone])  # (T, N) in node-id order 0..N-1 already
        Xw, yw = make_windows(sig, lookback=lookback)
        # per-sample mean-normalize using training zone stats
        mu, sigma = sig.mean(), sig.std() + 1e-6
        Xn = (Xw - mu) / sigma
        yn = (yw - mu) / sigma
        n_train = int(len(Xn) * 0.8)

        models = {
            "BaselineGCN": BaselineGCN(A, in_dim=lookback, hidden_dim=16, lr=0.02, seed=1),
            "AdaptiveGNN": AdaptiveGNN(A.shape[0], in_dim=lookback, hidden_dim=16, lr=0.02, seed=2),
            "TopologyAwareAGNN": TopologyAwareAGNN(A.shape[0], in_dim=lookback, A_fixed=A, alpha=0.5, hidden_dim=16, lr=0.02, seed=3),
        }
        for name, model in models.items():
            losses = []
            for ep in range(epochs):
                idx = RNG.permutation(n_train)[:64]
                for i in idx:
                    loss = model.train_step(Xn[i], yn[i])
                losses.append(loss)
            # in-sample (seen zone) test metrics on held-out 20% of the zone's own timeline
            preds = np.array([model.forward(Xn[i]) for i in range(n_train, len(Xn))])
            truth = yn[n_train:]
            preds_real = preds * sigma + mu
            truth_real = truth * sigma + mu
            train_metrics.append({
                "zone": zone, "model": name, "split": "seen",
                "MAE": mae(truth_real, preds_real), "RMSE": rmse(truth_real, preds_real),
                "MAPE": mape(truth_real, preds_real),
            })
            trained[name][zone] = {"model": model, "mu": mu, "sigma": sigma, "nodes": nodes, "A": A, "final_loss": losses[-1]}

    return trained, pd.DataFrame(train_metrics)


def zero_shot_eval(trained, graphs, signals, lookback=8):
    """For each unseen test zone, apply every TRAIN_ZONES-trained model
    instance directly (no fine-tuning) using the TEST zone's own graph/
    signal but the SOURCE zone's learned weights, matching the original
    zero-shot cross-city protocol as closely as node-count differences
    allow (weights are shape-compatible since in_dim=lookback is shared;
    adjacency is swapped to the target zone's real graph at inference)."""
    rows = []
    for test_zone in TEST_ZONES:
        G = graphs[test_zone]
        A_test, nodes_test = zone_adjacency(G)
        sig = np.array(signals[test_zone])
        Xw, yw = make_windows(sig, lookback=lookback)

        for model_name, per_zone in trained.items():
            for src_zone, pack in per_zone.items():
                model, mu, sigma = pack["model"], pack["mu"], pack["sigma"]
                Xn = (Xw - mu) / sigma
                yn = (yw - mu) / sigma
                # swap adjacency to the target zone's real graph, keep learned weights
                if model_name == "BaselineGCN":
                    from models.gnn_models import _normalize_adj
                    A_use = _normalize_adj(A_test)
                    preds = np.array([model.__class__.forward.__wrapped__(model, A_use, Xn[i]) if False else None for i in []])
                    # BaselineGCN.forward is bound to its own A_norm; call the shared propagate directly:
                    preds = []
                    for i in range(len(Xn)):
                        out = model._propagate(A_use, Xn[i], model.W1, model.b1, model.W2, model.b2)[-1]
                        preds.append(out.flatten())
                    preds = np.array(preds)
                elif model_name == "AdaptiveGNN":
                    # adaptive adjacency depends on model.E sized to source zone; if node counts
                    # differ we fall back to the model's own learned adjacency size-matched by
                    # truncation/padding to the test zone's node count.
                    preds = _cross_zone_predict_adaptive(model, Xn, A_test)
                else:
                    preds = _cross_zone_predict_topo(model, Xn, A_test)

                preds_real = np.array(preds) * sigma + mu
                truth_real = yn * sigma + mu
                rows.append({
                    "test_zone": test_zone, "trained_on": src_zone, "model": model_name,
                    "MAE": mae(truth_real, preds_real), "RMSE": rmse(truth_real, preds_real),
                    "MAPE": mape(truth_real, preds_real),
                })
    return pd.DataFrame(rows)


def _resize_embed(E, n_target, seed=0):
    rng = np.random.RandomState(seed)
    n_src, d = E.shape
    if n_target == n_src:
        return E
    if n_target < n_src:
        return E[:n_target]
    pad = rng.normal(0, 0.1, size=(n_target - n_src, d))
    return np.vstack([E, pad])


def _cross_zone_predict_adaptive(model, Xn, A_test):
    from models.gnn_models import _normalize_adj
    n_target = A_test.shape[0]
    E_use = _resize_embed(model.E, n_target)
    S = E_use @ E_use.T
    S = S - S.max(axis=1, keepdims=True)
    expS = np.exp(S)
    A_adapt = expS / expS.sum(axis=1, keepdims=True)
    n_take = min(n_target, Xn.shape[1])
    preds = []
    for i in range(len(Xn)):
        Xi = Xn[i][:n_take]
        if n_take < n_target:
            Xi = np.vstack([Xi, np.zeros((n_target - n_take, Xi.shape[1]))])
        out = model._propagate(A_adapt, Xi, model.W1, model.b1, model.W2, model.b2)[-1].flatten()
        preds.append(out[:Xn.shape[1]])
    return preds


def _cross_zone_predict_topo(model, Xn, A_test):
    from models.gnn_models import _normalize_adj
    n_target = A_test.shape[0]
    E_use = _resize_embed(model.E, n_target)
    S = E_use @ E_use.T
    S = S - S.max(axis=1, keepdims=True)
    expS = np.exp(S)
    A_learned = expS / expS.sum(axis=1, keepdims=True)
    A_topo = _normalize_adj(A_test)
    A_mix = model.alpha * A_topo + (1 - model.alpha) * A_learned
    n_take = min(n_target, Xn.shape[1])
    preds = []
    for i in range(len(Xn)):
        Xi = Xn[i][:n_take]
        if n_take < n_target:
            Xi = np.vstack([Xi, np.zeros((n_target - n_take, Xi.shape[1]))])
        out = model._propagate(A_mix, Xi, model.W1, model.b1, model.W2, model.b2)[-1].flatten()
        preds.append(out[:Xn.shape[1]])
    return preds


def compute_degradation(seen_metrics, zeroshot_metrics):
    seen_avg = seen_metrics.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().rename(
        columns=lambda c: f"seen_{c}")
    zs_avg = zeroshot_metrics.groupby(["model", "test_zone"])[["MAE", "RMSE", "MAPE"]].mean().reset_index()
    zs_avg = zs_avg.merge(seen_avg, on="model")
    zs_avg["MAE_degradation_pct"] = (zs_avg["MAE"] - zs_avg["seen_MAE"]) / zs_avg["seen_MAE"] * 100
    zs_avg["RMSE_degradation_pct"] = (zs_avg["RMSE"] - zs_avg["seen_RMSE"]) / zs_avg["seen_RMSE"] * 100
    return zs_avg


def topo_distance_to_degradation(zs_degradation, topo_dist_df, train_zones=TRAIN_ZONES):
    centroid_dist = topo_dist_df.loc[train_zones].mean(axis=0)  # mean distance from each zone to train-zone centroid
    zs_degradation = zs_degradation.copy()
    zs_degradation["topo_distance_from_train"] = zs_degradation["test_zone"].map(centroid_dist)
    return zs_degradation


if __name__ == "__main__":
    base = "/home/claude/bengaluru-digital-twin"
    graphs = load_zone_graphs(f"{base}/data/raw/bengaluru_zone_graphs.json")
    with open(f"{base}/data/raw/bengaluru_traffic_signals.json") as f:
        sig_data = json.load(f)
    signals = sig_data["signals"]

    feat_df = build_feature_table(graphs)
    dist_df = topology_distance_matrix(feat_df)

    trained, seen_metrics = train_model_family(graphs, signals, lookback=8, epochs=40)
    seen_metrics.to_csv(f"{base}/evaluation/seen_zone_metrics.csv", index=False)

    zs_metrics = zero_shot_eval(trained, graphs, signals, lookback=8)
    zs_metrics.to_csv(f"{base}/evaluation/zero_shot_raw_metrics.csv", index=False)

    degradation = compute_degradation(seen_metrics, zs_metrics)
    degradation = topo_distance_to_degradation(degradation, dist_df)
    degradation.to_csv(f"{base}/evaluation/degradation_summary.csv", index=False)

    corr_rows = []
    for model in degradation["model"].unique():
        sub = degradation[degradation["model"] == model]
        r = np.corrcoef(sub["topo_distance_from_train"], sub["MAE_degradation_pct"])[0, 1]
        corr_rows.append({"model": model, "pearson_r_topo_vs_degradation": r})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(f"{base}/evaluation/topology_degradation_correlation.csv", index=False)

    with open(f"{base}/evaluation/results_summary.json", "w") as f:
        json.dump({
            "seen_metrics": seen_metrics.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().reset_index().to_dict(orient="records"),
            "degradation": degradation.to_dict(orient="records"),
            "correlation": corr_df.to_dict(orient="records"),
        }, f, indent=2)

    print("=== Seen-zone avg metrics ===")
    print(seen_metrics.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().round(3))
    print("\n=== Degradation summary ===")
    print(degradation[["model", "test_zone", "MAE", "seen_MAE", "MAE_degradation_pct", "topo_distance_from_train"]].round(3))
    print("\n=== Topology-distance vs degradation correlation ===")
    print(corr_df.round(3))
