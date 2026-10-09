# SAIL: a learnable generalisation of spatial autocorrelation

SAIL (Spatial Autocorrelation Index Layer) turns classical spatial
autocorrelation statistics such as Moran's I, Geary's C and Getis-Ord G\* into
a differentiable layer for graph-structured data. It learns *which combination
of node features* to measure (`f_map`) and *which neighbours matter*
(`f_affinity`, a learned spatial temperature), and returns both node-level
(local) and graph-level (global) autocorrelation scores. With fixed
components it reproduces the classical statistics exactly.

SAIL can be used

* **unsupervised**: discover *SAIL Modes*, feature combinations that are
  maximally spatially coherent, then map them with LISA-style hotspot tests;
* **supervised**: as a backbone whose global descriptors feed a prediction head.
* 
## Installation

```bash
git clone https://github.com/pkeller00/SAIL.git && cd SAIL
micromamba create -f environment.yml && micromamba activate sail
pip install -e ".[all]"      # or just `pip install -e .` for the core library
```

Optional extras: `spatial` (AnnData/squidpy), `baselines` (esda, libpysal,
Hotspot), `survival` (lifelines), `gnn` (PyTorch Geometric), `wsi`
(tiatoolbox), `maps` (geopandas). A GPU is recommended for large graphs but
not required. The diffusion experiment writes MP4s with ffmpeg + libx264
(installed by `environment.yml`; outside conda, use a system ffmpeg built with
libx264). If ffmpeg isn't found, the animations are saved as GIFs instead.

## Quick start

```python
import sail

# 1. A spatial graph: kNN (default), radius, or dense (all pairs)
g = sail.build_graph(coords, features, method="knn", k=32)          # coords (N, 2), features (N, d)

# 2. Learn 20 SAIL Modes that minimise Geary's C (maximise spatial coherence)
modes = sail.SAILModes(input_dim=features.shape[1], n_modes=20, local_index="geary").fit([g])
modes.W                      # (d, 20) loadings: which features define each mode
modes.scores([g])            # (1, 20) global SAIL score of every mode

# 3. Where is each mode coherent? Permutation-tested LISA clusters
labels = sail.lisa(g, modes.model)                                  # (N, 20) High-High / Low-Low / ...

# 4. Plot a mode over the tissue image stored in an AnnData object
from sail.viz import AnnDataImage, overlay, plot_modes
overlay(g.coords, labels[:, 0], kind="lisa", image=AnnDataImage(adata, img_key="hires"))
plot_modes(g, modes, feature_names, lisa_labels=labels)
```

Several graphs (e.g. one per patient) are passed as a list; modes are then
shared across graphs and `modes.scores(graphs)` gives one row per graph.

### Building blocks

| Component | Options (`sail.*`) |
|---|---|
| Graph `build_graph(method=...)` | `"knn"`, `"radius"`, `"dense"`; Euclidean or haversine (lat/lon) |
| Feature map `f_map` | `StiefelLinear` (orthonormal, default for modes), `NormalisedLinear`, `Linear`, `OneHotMap`, `GumbelSoftmaxMap` |
| Feature affinity `f_affinity` | none (distance only, default), `SharedMapAffinity`, `QueryKeyAffinity` |
| Local index `f_local` | `"moran"`, `"geary"`, `"gstar"`, `"morisita_horn"`, or any callable |
| Reduction `f_reduce` | `"mean"`, `"sum"`, `"max"` |
| Unsupervised | `SAILModes` (decorrelation and sparsity regularisers) |
| Supervised | `sail.supervised.SAILPredictor` (SAIL backbone + linear heads) |
| Hotspots | `sail.lisa` (permutation test, BH or max-T correction) |
| Visualisation | `sail.viz.overlay` with `AnnDataImage(adata, img_key=...)`, `WSIImage(path, coord_scale, offset)` or `ArrayImage`; `plot_modes`; `export_annotation_store` (tiatoolbox viewer) |
| Evaluation | `sail.evaluation`: Dice, survival (KM, log-rank, C-index, Cox HR, RMST), response (Mann-Whitney, AUC, bootstrap CI) |
| Baselines | `sail.baselines`: univariate / mean Moran's I and Geary's C, MULTISPATI-PCA, Hotspot |

All graph types share a single vectorised forward pass over padded neighbour
lists. Memory is linear in the number of edges; `SAIL(mode_chunk=...)` further
bounds memory for large graphs at inference.

## Reproducing the paper

Every experiment is a folder in `experiments/` with a `config.yaml` (all
hyper-parameters) and a `run.py` whose stages are run with

```bash
sail-run experiments/<experiment>/config.yaml [--stage <stage>] [--device cuda]
```

Trained paper models are in `models/` and are used automatically by the
evaluation stages, so tables and figures can be regenerated without
retraining. The tables behind the paper are in `results/`. Data must be
obtained separately; see `datasets/<name>/README.md`.

| Paper | Experiment | Stages |
|---|---|---|
| Fig. 2; Supp. H, Fig. 4 | `synthetic` | `index`, `toys`, `terrain` |
| Fig. 3; Supp. G, Tables II-III | `synthetic` | `sweep`, `evaluate`, `blend` |
| Supp. F, Table I, Figs. 1-2 | `compute_scaling` | `benchmark`, `table`, `plots` |
| Fig. 4, Table I; Supp. I, Tables IV-V, XI | `4i` | `train`, `eval`, `baselines`, `table`, `ablation` |
| Fig. 5, Table II; Supp. J, Tables VI, XII | `pd1_cscc` | `train`, `eval`, `biomarkers`, `ablation` |
| Fig. 6a, Table III; Supp. K, Tables VII, IX | `survival` (`orion_crc.yaml`) | `train`, `eval`, `baselines` |
| Fig. 6b-c; Supp. K, Tables VIII, XIII-XIV | `survival` (`tcga_brca.yaml`, `tcga_crc.yaml`) | `train`, `eval`, `ablation` |
| Fig. 7, Table IV; Supp. M | `mesothelioma` | `train`/`eval`, `baselines`, `table`, `stats`, `interpret` |
| Fig. 8; Supp. N, Figs. 10-11 | `macroeconomics` | `train`, `eval`, `maps` |

## Repository layout

```
sail/            the library
experiments/     one folder per experiment: config.yaml, run.py
datasets/        how to obtain each dataset + prepare.py -> data/<name>/
models/          trained models reported in the paper
results/         result tables reported in the paper
tests/           unit tests (pytest)
```

## Tests

```bash
pytest tests
```

The tests check that SAIL reproduces the classical statistics exactly
(Theorem IV.1), that kNN, radius and dense graphs agree, and that LISA,
the baselines and the visualisation run.

## Licence

Free for academic and other non-commercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). For commercial licensing,
contact the authors.

## Citation

See [`CITATION.cff`](CITATION.cff).
