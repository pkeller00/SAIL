"""Prognostic value of unsupervised SAIL Modes (Section V-D; Supp. Section K).

One spatial graph per patient (cell graph or WSI patch graph). Modes are
learned on all patients jointly without access to outcome; each mode's global
score is then tested for survival association by a median split.

Configs: ``orion_crc.yaml``, ``tcga_brca.yaml``, ``tcga_crc.yaml``.
Inputs (written by ``datasets/<cohort>/prepare.py``): ``samples.pt`` (node
features + coordinates per patient) and ``clinical.csv`` (``id, time, event``).

Stages:

* ``train``     learn the modes
* ``eval``      per-patient scores, survival table for every mode
                (Supp. Tables VII-VIII), KM curve + loadings of the best mode (Fig. 6)
* ``baselines`` fixed-weight comparators on the same graphs: global Geary's C / Moran's I
                per feature and their mean, MULTISPATI-PCA and Hotspot (Table III, Supp. Table IX)
* ``ablation``  one-at-a-time component ablation (Supp. Tables XIII-XIV)
* ``overlays``  tiatoolbox annotation stores of the best mode for the highest/lowest-scoring patients
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sail  # noqa: E402
from sail.baselines import (  # noqa: E402
    global_moran_geary,
    inverse_distance_from_graph,
    multispati_pca,
    run_hotspot,
)
from sail.config import ExperimentConfig  # noqa: E402
from sail.data import load_samples, samples_to_graphs  # noqa: E402
from sail.evaluation import fdr_bh, plot_km, survival_table  # noqa: E402
from sail.experiment import load_modes, run_ablation, train_modes, write_table  # noqa: E402
from sail.viz import export_annotation_store, plot_mode_weights  # noqa: E402

log = logging.getLogger("sail")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def cohort(cfg: ExperimentConfig) -> tuple[dict, pd.DataFrame]:
    """Samples and the clinical table restricted to patients present in both."""
    samples = load_samples(cfg.path("samples"))
    clinical = pd.read_csv(cfg.path("clinical"), dtype={"id": str})
    clinical = clinical[clinical["id"].isin(samples["ids"])].sort_values("id").reset_index(drop=True)
    log.warning("%d patients, %d events", len(clinical), int(clinical["event"].sum()))
    return samples, clinical


def build(cfg: ExperimentConfig, data: Optional[tuple] = None):
    samples, clinical = data or cohort(cfg)
    return samples_to_graphs(samples, clinical["id"].tolist(), **cfg.graph)


def _survival(cfg, scores: np.ndarray, clinical: pd.DataFrame, names) -> pd.DataFrame:
    table = survival_table(scores, clinical["time"].to_numpy(float), clinical["event"].to_numpy(int),
                           tau=cfg.eval["rmst_tau_days"], names=names)
    table["logrank_p_bh"] = fdr_bh(table["logrank_p"].to_numpy())
    return table


# --------------------------------------------------------------------------- #
# SAIL
# --------------------------------------------------------------------------- #
def train(cfg: ExperimentConfig) -> None:
    train_modes(cfg, build(cfg))


def evaluate(cfg: ExperimentConfig) -> None:
    samples, clinical = cohort(cfg)
    graphs = build(cfg, (samples, clinical))
    modes = load_modes(cfg)
    scores = modes.scores(graphs)
    names = [f"Mode {m + 1}" for m in range(scores.shape[1])]
    write_table(pd.concat([clinical, pd.DataFrame(scores, columns=names)], axis=1), cfg.out / "per_patient_scores.csv")
    table = _survival(cfg, scores, clinical, names)
    write_table(table, cfg.out / "survival_all_modes.csv", "survival association of every mode")

    best = int(table["c_index_oriented"].idxmax())
    row = table.loc[best]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_mode_weights(modes.W, samples["feature_names"], best, ax=axs[0], top_k=cfg.eval.get("top_features", 10),
                      title=f"{row['name']} weights")
    plot_km(scores[:, best], clinical["time"], clinical["event"], ax=axs[1],
            title=f"{row['name']}: C-index {row['c_index_oriented']:.2f}, log-rank p = {row['logrank_p']:.2g}, "
                  f"HR {row['hr']:.2f} ({row['hr_lo']:.2f}-{row['hr_hi']:.2f})")
    fig.savefig(cfg.out / "best_mode_km.png", dpi=300, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baselines(cfg: ExperimentConfig) -> None:
    """Fixed inverse-distance kNN weights on raw distances (``baselines.k`` neighbours)."""
    b = cfg.baselines
    samples, clinical = cohort(cfg)
    graphs = samples_to_graphs(samples, clinical["id"].tolist(), method="knn", k=b["k"], log_distances=False)
    names = samples["feature_names"]
    weights = [inverse_distance_from_graph(g.neighbors.numpy(), g.distances.numpy()) for g in graphs]
    tables = []

    # univariate and mean global Geary's C / Moran's I
    stats = [global_moran_geary(g.x.numpy(), g.neighbors.numpy(), g.distances.numpy()) for g in graphs]
    for k, stat in enumerate(["moran", "geary"]):
        S = np.stack([s[k] for s in stats])
        S = np.column_stack([S, S.mean(1)])
        t = _survival(cfg, S, clinical, list(names) + ["Mean"]).assign(method=f"{stat} (univariate / mean)")
        tables.append(t)

    # MULTISPATI-PCA (pooled), per-patient global Moran's I of each kept axis
    Xs = [g.x.numpy() for g in graphs]
    ms = multispati_pca(Xs, weights, eig_frac=b["multispati_eig_frac"])
    ms.axes_table().to_csv(cfg.out / "multispati_axes.csv", index=False)
    S = np.stack([global_moran_geary(ms.transform(g.x.numpy()), g.neighbors.numpy(), g.distances.numpy())[0]
                  for g in graphs])
    tables.append(_survival(cfg, S, clinical, [f"MS{i + 1}" for i in range(S.shape[1])]).assign(method="MULTISPATI"))

    # Hotspot modules (pooled), per-patient global Moran's I of each module score
    hs = run_hotspot(Xs, weights, names, min_genes=b["hotspot_min_genes"])
    hs.modules.to_frame("module").to_csv(cfg.out / "hotspot_modules.csv")
    bounds = np.cumsum([0] + [g.num_nodes for g in graphs])
    S = np.stack([global_moran_geary(hs.scores[a:z], g.neighbors.numpy(), g.distances.numpy())[0]
                  for g, a, z in zip(graphs, bounds[:-1], bounds[1:])])
    tables.append(_survival(cfg, S, clinical, [f"HS{m}" for m in hs.module_ids]).assign(method="Hotspot"))

    write_table(pd.concat(tables, ignore_index=True), cfg.out / "baseline_survival.csv", "baselines")


# --------------------------------------------------------------------------- #
def ablation(cfg: ExperimentConfig) -> None:
    data = cohort(cfg)

    def summarise(run_cfg, modes, graphs):
        scores = modes.scores(graphs)
        table = _survival(run_cfg, scores, data[1], [f"Mode {m + 1}" for m in range(scores.shape[1])])
        r = table.loc[table["c_index_oriented"].idxmax()]
        return dict(best_mode=r["name"], c_index=r["c_index_oriented"], logrank_p=r["logrank_p"],
                    delta_rmst=r["delta_rmst"])

    write_table(run_ablation(cfg, lambda c: build(c, data), summarise),
                cfg.out / "ablation" / "ablation_summary.csv", "ablation")


def overlays(cfg: ExperimentConfig) -> None:
    """Annotation stores (local field + LISA) of one mode for the extreme patients."""
    o = cfg.eval["overlays"]
    samples, clinical = cohort(cfg)
    graphs = build(cfg, (samples, clinical))
    modes = load_modes(cfg)
    m = o["mode"] - 1
    scores = modes.scores(graphs)[:, m]
    order = np.argsort(scores)
    picks = list(order[: o["n_patients"]]) + list(order[-o["n_patients"]:])
    out = cfg.out / "overlays"
    out.mkdir(exist_ok=True)
    for i in picks:
        g = graphs[i]
        res = modes.transform(g)
        labels = sail.lisa(g, modes.model, modes=m)
        export_annotation_store(g.coords.numpy(), out / f"{g.meta['id']}_mode{m + 1}.db",
                                {"local": res.local[0][:, m].cpu().numpy(), "field": res.z[0][:, m].cpu().numpy()},
                                lisa_labels=labels, coord_scale=o.get("coord_scale", 1.0),
                                offset=o.get("offset", (0, 0)), radius=o.get("radius", 20))


STAGES = {"train": train, "eval": evaluate, "baselines": baselines, "ablation": ablation, "overlays": overlays}
DEFAULT_STAGES = ["train", "eval"]
