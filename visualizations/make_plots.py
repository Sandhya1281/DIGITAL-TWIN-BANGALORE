import json
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/claude/bengaluru-digital-twin")
from topology.extract_topology import load_zone_graphs

BASE = "/home/claude/bengaluru-digital-twin"
OUT = f"{BASE}/visualizations"

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def topology_bar_plots(feat_df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cols = ["N", "avg_degree", "density", "clustering_coeff", "avg_shortest_path", "spectral_gap"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(feat_df)))
    for ax, col in zip(axes.flat, cols):
        order = feat_df[col].sort_values(ascending=False)
        ax.bar(order.index, order.values, color=colors)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=60)
    fig.suptitle("Bengaluru Zone Topology Features")
    fig.tight_layout()
    fig.savefig(f"{OUT}/topology_features.png")
    plt.close(fig)


def degradation_vs_topology_plot(degradation_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, sub in degradation_df.groupby("model"):
        ax.scatter(sub["topo_distance_from_train"], sub["MAE_degradation_pct"], label=model, s=70)
        for _, row in sub.iterrows():
            ax.annotate(row["test_zone"], (row["topo_distance_from_train"], row["MAE_degradation_pct"]), fontsize=7, alpha=0.7)
    ax.set_xlabel("Topology distance from training-zone centroid")
    ax.set_ylabel("MAE degradation (%) vs seen zones")
    ax.set_title("Topology Distance vs Zero-Shot Performance Degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/topology_vs_degradation.png")
    plt.close(fig)


def model_comparison_plot(seen_df, degradation_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    seen_avg = seen_df.groupby("model")[["MAE", "RMSE", "MAPE"]].mean()
    seen_avg[["MAE", "RMSE"]].plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Seen-zone MAE / RMSE by model")
    axes[0].tick_params(axis="x", rotation=20)

    deg_avg = degradation_df.groupby("model")["MAE_degradation_pct"].mean()
    deg_avg.plot(kind="bar", ax=axes[1], color="#55A868")
    axes[1].set_title("Avg zero-shot MAE degradation (%) by model")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison.png")
    plt.close(fig)


def cross_zone_heatmap(zs_raw_df):
    pivot = zs_raw_df.pivot_table(index="trained_on", columns="test_zone", values="MAE", aggfunc="mean")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, model in zip(axes, zs_raw_df["model"].unique()):
        sub = zs_raw_df[zs_raw_df["model"] == model]
        piv = sub.pivot_table(index="trained_on", columns="test_zone", values="MAE", aggfunc="mean")
        im = ax.imshow(piv.values, cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_title(model)
        fig.colorbar(im, ax=ax, fraction=0.046, label="MAE")
    fig.suptitle("Cross-Zone Zero-Shot MAE Heatmap (trained-on x tested-on)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cross_zone_heatmap.png")
    plt.close(fig)


def graph_topology_viz(graphs):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (zone, G) in zip(axes.flat, graphs.items()):
        pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes()}
        nx.draw(G, pos, ax=ax, node_size=40, node_color="#c0392b", edge_color="#7f8c8d", width=0.8)
        ax.set_title(zone, fontsize=9)
    fig.suptitle("Bengaluru Zone Road-Network Graphs")
    fig.tight_layout()
    fig.savefig(f"{OUT}/zone_graphs.png")
    plt.close(fig)


def message_passing_viz(G, zone_name="Koramangala"):
    """Schematic of one round of GNN message passing on a real zone graph."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pos = nx.spring_layout(G, seed=1)
    titles = ["Step 0: node features", "Step 1: aggregate neighbor messages", "Step 2: updated node states"]
    rng = np.random.RandomState(3)
    vals = rng.uniform(0.2, 1.0, size=G.number_of_nodes())
    for step, (ax, title) in enumerate(zip(axes, titles)):
        if step == 1:
            colorvals = np.array([vals[list(G.nodes()).index(n)] * 0.5 + 0.5 * np.mean(
                [vals[list(G.nodes()).index(nb)] for nb in G.neighbors(n)]) for n in G.nodes()])
        elif step == 2:
            colorvals = vals * 0.3 + 0.7 * np.array([vals[list(G.nodes()).index(n)] for n in G.nodes()])
            colorvals = np.clip(colorvals + rng.normal(0, 0.05, size=len(colorvals)), 0, 1)
        else:
            colorvals = vals
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colorvals, cmap="plasma", node_size=180, vmin=0, vmax=1)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
        if step == 1:
            # highlight one node's incoming messages
            center = list(G.nodes())[0]
            for nb in G.neighbors(center):
                nx.draw_networkx_edges(G, pos, edgelist=[(nb, center)], ax=ax, edge_color="red", width=2)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"GNN Message Passing — {zone_name} zone (adaptive adjacency)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/message_passing.png")
    plt.close(fig)


def training_curves(seen_df):
    # synthetic-but-representative curve shape isn't available (we didn't log per-epoch loss
    # to disk); instead show final per-zone loss bars as a stand-in "convergence summary"
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in seen_df.groupby("model"):
        ax.plot(sub["zone"], sub["MAE"], marker="o", label=model)
    ax.set_xticklabels(sub["zone"], rotation=45, ha="right")
    ax.set_ylabel("Held-out-timestep MAE (seen zone)")
    ax.set_title("Per-Zone Final Validation MAE by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/training_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    feat_df = pd.read_csv(f"{BASE}/topology/topology_features.csv", index_col=0)
    seen_df = pd.read_csv(f"{BASE}/evaluation/seen_zone_metrics.csv")
    zs_raw = pd.read_csv(f"{BASE}/evaluation/zero_shot_raw_metrics.csv")
    degradation = pd.read_csv(f"{BASE}/evaluation/degradation_summary.csv")
    graphs = load_zone_graphs(f"{BASE}/data/raw/bengaluru_zone_graphs.json")

    topology_bar_plots(feat_df)
    degradation_vs_topology_plot(degradation)
    model_comparison_plot(seen_df, degradation)
    cross_zone_heatmap(zs_raw)
    graph_topology_viz(graphs)
    message_passing_viz(graphs["Koramangala"], "Koramangala")
    training_curves(seen_df)
    print("all plots written to", OUT)
