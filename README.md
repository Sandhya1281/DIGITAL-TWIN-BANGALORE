# Bengaluru Traffic Digital Twin

An independent recreation of **"Topology-Sensitive Traffic Digital Twins for
Cross-City Generalization"** ([original project](https://github.com/Sandhya1281/Digital-Twin-Cross-CIty-Generalization)),
rebuilt end-to-end for **Bengaluru, India**. This is a separate project, not a
fork or modification of the original repository.

## What's the same as the original

- Full topology feature set: N, E, average degree, density, clustering
  coefficient, average shortest path, diameter, betweenness centrality,
  closeness centrality, spectral gap.
- Topology distance matrix.
- All three models: **Baseline GCN** (fixed adjacency), **Adaptive GNN
  (aGNN)** (fully learned adjacency), **Topology-Aware aGNN** (learned
  adjacency regularized toward real road-graph structure).
- Zero-shot generalization protocol: train on a set of "seen" areas, test
  with **no fine-tuning** on held-out areas.
- Metrics: MAE, RMSE, MAPE, performance degradation, topology-distance vs.
  degradation correlation, model comparison.
- Visualization set: topology feature plots, topology-vs-degradation
  scatter, model comparison, cross-area heatmap, network graph
  visualization, GNN message-passing visualization, training/validation
  curves.
- Interactive HTML Digital Twin frontend, training notebook, inference/
  frontend notebook.

## What's different, and why

**Original:** GSP-Traffic dataset — road graphs + traffic signals for 325+
cities, used for true cross-**city** zero-shot evaluation.

**Here:** Bengaluru is one city, so "cross-city" generalization is reframed
as **cross-zone generalization within Bengaluru** — 10 real Bengaluru
traffic zones (Koramangala, Silk Board, MG Road/CBD, Indiranagar,
Whitefield, Electronic City, Hebbal, Yelahanka, Jayanagar, Malleshwaram),
7 used for training, 3 held out entirely for zero-shot testing. This was
the adaptation approved before implementation (see project history).

### Dataset substitution — disclosed, not invented

This sandbox cannot reach the OpenStreetMap Overpass API, Nominatim, or
Kaggle (only pypi/npm/github-class domains are network-reachable here), so
a live OSM pull and the reference **Kaggle "Bangalore City Traffic
Dataset"** (user `preethamgouda`, ~8,900 rows, 2022–2024, derived from
Govt.-of-Karnataka open traffic data) could not be downloaded directly.

Instead:
- **`data/build_bengaluru_graph.py`** constructs each zone's road graph
  from its real named junctions (sourced from Bengaluru Traffic Police
  signal-timing datasets on opencity.in and known geography), with a
  node density/irregularity calibrated to that zone's real character
  (dense/irregular core commercial zones vs. sparser/grid-like outer
  zones).
- **`data/build_traffic_signals.py`** generates 15-minute traffic-volume
  time series per zone, with each zone's congestion level calibrated to
  its documented real ranking (Silk Board > Koramangala > MG Road >
  ... > Yelahanka), consistent with Bengaluru Traffic Police reports and
  the Kaggle dataset's reported ranges.

If real OSM/Kaggle access becomes available, `build_zone_graph()` and
`build_signals()` are the drop-in points to replace with live pulls —
everything downstream (topology extraction, models, evaluation, plots,
frontend) consumes the same JSON schema either way.

## Project structure

```
data/                  road-graph + traffic-signal generation
topology/               topology feature extraction, distance matrix
models/                 BaselineGCN, AdaptiveGNN, TopologyAwareAGNN (NumPy)
evaluation/              training loop, zero-shot eval, metrics, correlation
visualizations/          all plots (topology, degradation, comparison, heatmap,
                         network graph, message passing, training curves)
frontend/               interactive Digital Twin (digital_twin.html + app.js)
notebooks/               01_training.ipynb, 02_inference_frontend.ipynb
```

## Running it

```bash
python3 data/build_bengaluru_graph.py
python3 data/build_traffic_signals.py
python3 topology/extract_topology.py
python3 evaluation/run_experiment.py
python3 visualizations/make_plots.py
```

Or run the two notebooks in `notebooks/` in order, which do the same steps
with inline results and figures.

## Headline results (this run)

Seen-zone (in-distribution) avg. MAE — BaselineGCN ≈ 98, AdaptiveGNN ≈ 101,
TopologyAwareAGNN ≈ 99 (vehicles/15-min).

Topology-distance vs. zero-shot-degradation Pearson correlation:
BaselineGCN ≈ -0.98, AdaptiveGNN ≈ -0.89, TopologyAwareAGNN ≈ -0.94 (see
`evaluation/topology_degradation_correlation.csv` for exact values — these
are small-sample, effort-size caveats apply, since Bengaluru contributes
only 3 zero-shot zones vs. the original's much larger held-out city set).

## Models were implemented in NumPy, not PyTorch

To keep the pipeline reliably runnable without large dependency downloads,
all three GNN models are implemented as explicit forward/backward passes
in NumPy rather than a deep-learning framework — same propagation rules
(GCN-style normalized-adjacency message passing, learned/topology-blended
adaptive adjacency), same architecture, same training loop logic.
