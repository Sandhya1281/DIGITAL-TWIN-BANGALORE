"""
Extracts the same topology feature set as the original GSP-Traffic
project: N, E, average degree, density, clustering coefficient,
average shortest path length, diameter, betweenness centrality (mean),
closeness centrality (mean), and spectral gap (of the normalized
Laplacian). Computed identically via NetworkX on each Bengaluru zone
graph (one feature vector per zone, in place of one per city).
"""
import json
import networkx as nx
import numpy as np
import pandas as pd


def load_zone_graphs(path):
    with open(path) as f:
        data = json.load(f)
    graphs = {}
    for zone, g in data.items():
        G = nx.Graph(zone=zone)
        for nd in g["nodes"]:
            G.add_node(nd["id"], **{k: v for k, v in nd.items() if k != "id"})
        for e in g["edges"]:
            G.add_edge(e["u"], e["v"], **{k: v for k, v in e.items() if k not in ("u", "v")})
        graphs[zone] = G
    return graphs


def topology_features(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    avg_degree = 2 * e / n if n > 0 else 0
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    if nx.is_connected(G):
        avg_shortest_path = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        largest = max(nx.connected_components(G), key=len)
        sub = G.subgraph(largest)
        avg_shortest_path = nx.average_shortest_path_length(sub)
        diameter = nx.diameter(sub)
    betweenness = np.mean(list(nx.betweenness_centrality(G).values()))
    closeness = np.mean(list(nx.closeness_centrality(G).values()))

    L = nx.normalized_laplacian_matrix(G).toarray()
    eigvals = np.sort(np.linalg.eigvalsh(L))
    # spectral gap = difference between the two smallest eigenvalues (algebraic connectivity gap)
    spectral_gap = float(eigvals[1] - eigvals[0]) if len(eigvals) > 1 else 0.0

    return {
        "N": n, "E": e, "avg_degree": avg_degree, "density": density,
        "clustering_coeff": clustering, "avg_shortest_path": avg_shortest_path,
        "diameter": diameter, "betweenness_mean": betweenness,
        "closeness_mean": closeness, "spectral_gap": spectral_gap,
    }


def build_feature_table(graphs):
    rows = []
    for zone, G in graphs.items():
        feats = topology_features(G)
        feats["zone"] = zone
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("zone")
    cols = ["N", "E", "avg_degree", "density", "clustering_coeff", "avg_shortest_path",
            "diameter", "betweenness_mean", "closeness_mean", "spectral_gap"]
    return df[cols]


def topology_distance_matrix(feat_df):
    """Euclidean distance on z-normalized topology feature vectors, zone x zone
    (equivalent to the original project's city x city topology distance matrix)."""
    z = (feat_df - feat_df.mean()) / feat_df.std(ddof=0).replace(0, 1)
    zones = list(z.index)
    n = len(zones)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(z.iloc[i].values - z.iloc[j].values)
    return pd.DataFrame(dist, index=zones, columns=zones)


if __name__ == "__main__":
    graphs = load_zone_graphs("/home/claude/bengaluru-digital-twin/data/raw/bengaluru_zone_graphs.json")
    feat_df = build_feature_table(graphs)
    feat_df.to_csv("/home/claude/bengaluru-digital-twin/topology/topology_features.csv")
    dist_df = topology_distance_matrix(feat_df)
    dist_df.to_csv("/home/claude/bengaluru-digital-twin/topology/topology_distance_matrix.csv")
    with open("/home/claude/bengaluru-digital-twin/topology/topology_features.json", "w") as f:
        json.dump(feat_df.reset_index().to_dict(orient="records"), f, indent=2)
    print(feat_df.round(3))
