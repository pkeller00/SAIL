"""4i spatial proteomics: unsupervised SAIL Modes vs. subcellular compartments.

Stages (``sail-run experiments/4i/config.yaml --stage <name>``):

* ``train``     learn 20 SAIL Modes on the 4i field of view (labels never used)
* ``eval``      LISA hotspots per mode, Dice against the 10 compartments
                (Supp. Tables IV-V), mode figure (Supp. Fig. 5) and Fig. 4 panels
* ``baselines`` univariate / mean Moran's I, MULTISPATI-PCA and Hotspot Dice
* ``table``     Supp. Table V: best Dice per compartment for every method + Wilcoxon tests
* ``ablation``  one-at-a-time component ablation (Supp. Table XI)
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sail  # noqa: E402
from sail.baselines import (  # noqa: E402
    composite_mean,
    inverse_distance_knn,
    local_moran_lisa,
    multispati_pca,
    run_hotspot,
)
from sail.config import ExperimentConfig  # noqa: E402
from sail.data import anndata_to_graph  # noqa: E402
from sail.evaluation import best_per_region, lisa_region_dice, paired_wilcoxon  # noqa: E402
from sail.experiment import load_modes, run_ablation, train_modes, write_table  # noqa: E402
from sail.viz import overlay, plot_mode_weights, plot_modes  # noqa: E402

log = logging.getLogger("sail")
METHODS = ["SAIL", "Univariate", "Mean", "MULTISPATI", "Hotspot"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_adata(cfg: ExperimentConfig):
    import squidpy as sq

    path = cfg.data.get("path")
    adata = sq.datasets.four_i(path=str(cfg.path("path")) if path else None)
    adata.var_names_make_unique()  # the panel lists p-ERK twice
    return adata


def regions(cfg: ExperimentConfig, adata) -> dict[str, np.ndarray]:
    col = adata.obs[cfg.data["region_col"]].astype("category")
    return {r: (col.values == r) for r in col.cat.categories}


def build(cfg: ExperimentConfig, adata=None):
    adata = adata if adata is not None else load_adata(cfg)
    return [anndata_to_graph(adata, **cfg.graph)]


# --------------------------------------------------------------------------- #
# SAIL
# --------------------------------------------------------------------------- #
def train(cfg: ExperimentConfig) -> None:
    train_modes(cfg, build(cfg))


def sail_dice(cfg: ExperimentConfig, modes: sail.SAILModes, graph, region_masks) -> pd.DataFrame:
    labels = sail.lisa(graph, modes.model, permutations=cfg.eval["permutations"], alpha=cfg.eval["alpha"])
    return lisa_region_dice(labels, region_masks), labels


def evaluate(cfg: ExperimentConfig) -> None:
    adata = load_adata(cfg)
    graph = build(cfg, adata)[0]
    modes = load_modes(cfg)
    sail.set_seed(cfg.seed)
    dice, labels = sail_dice(cfg, modes, graph, regions(cfg, adata))
    np.save(cfg.out / "lisa_labels.npy", labels)
    write_table(dice.sort_values("dice", ascending=False), cfg.out / "sail_dice.csv")
    write_table(best_per_region(dice), cfg.out / "sail_best_per_region.csv", "Supp. Table IV")

    names = graph.meta["feature_names"]
    plot_modes(graph, modes, names, lisa_labels=labels, ncols=5, point_size=1,
               save_path=cfg.out / "all_modes.png")
    fig, axs = plt.subplots(2, len(cfg.eval["figure_modes"]), figsize=(5 * len(cfg.eval["figure_modes"]), 9))
    for col, m in enumerate(cfg.eval["figure_modes"]):
        overlay(graph.coords, labels[:, m - 1], kind="lisa", ax=axs[0, col], point_size=1, title=f"SAIL Mode {m}")
        plot_mode_weights(modes.W, names, m - 1, ax=axs[1, col])
    fig.savefig(cfg.out / "fig4_modes.png", dpi=300, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def baselines(cfg: ExperimentConfig) -> None:
    b = cfg.baselines
    adata = load_adata(cfg)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=np.float64)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    genes = np.asarray(adata.var_names)
    keep = X.std(0) > 1e-10
    X, genes = X[:, keep].astype(np.float64), genes[keep]
    region_masks = regions(cfg, adata)
    w = inverse_distance_knn(coords, k=b["k"], log_distances=b["log_distances"])

    def dice_of(labels_by_name: dict[str, np.ndarray], method: str) -> pd.DataFrame:
        names = list(labels_by_name)
        df = lisa_region_dice(np.column_stack([labels_by_name[n] for n in names]), region_masks)
        df.insert(0, "component", [names[m - 1] for m in df["mode"]])
        df.insert(0, "method", method)
        return df.drop(columns="mode")

    # univariate: local Moran's I of each of the top-k proteins by global Moran's I
    np.random.seed(cfg.seed)
    moran_i = np.array([_esda_global_moran(X[:, j], w) for j in range(X.shape[1])])
    top = np.argsort(-moran_i)[: b["univariate_top_k"]]
    uni = dice_of({genes[j]: local_moran_lisa(X[:, j], w, b["univariate_permutations"], test="one_sided_raw")
                   for j in top}, "Univariate")

    # mean: local Moran's I of the mean z-scored protein
    mean = dice_of({"mean": local_moran_lisa(composite_mean(X), w, b["mean_permutations"])}, "Mean")

    # MULTISPATI-PCA axes
    ms = multispati_pca([X], [w], eig_frac=b["multispati_eig_frac"])
    S = ms.transform(X)
    multi = dice_of({f"MS{i + 1}": local_moran_lisa(S[:, i], w, b["permutations"]) for i in range(S.shape[1])},
                    "MULTISPATI")

    # Hotspot modules
    hs = run_hotspot([X], [w], genes, min_genes=b["hotspot_min_genes"])
    hot = dice_of({f"HS{m}": local_moran_lisa(hs.scores[:, i], w, b["permutations"])
                   for i, m in enumerate(hs.module_ids)}, "Hotspot")

    write_table(pd.concat([uni, mean, multi, hot]), cfg.out / "baseline_dice.csv")


def _esda_global_moran(values: np.ndarray, w) -> float:
    from esda.moran import Moran

    return float(Moran(values, w.to_libpysal(), permutations=0).I)


def table(cfg: ExperimentConfig) -> None:
    sail_best = best_per_region(pd.read_csv(cfg.out / "sail_dice.csv")).assign(method="SAIL")
    base = pd.read_csv(cfg.out / "baseline_dice.csv")
    best = pd.concat([sail_best, *(best_per_region(g).assign(method=m) for m, g in base.groupby("method"))])
    wide = best.pivot(index="region", columns="method", values="dice")[METHODS]
    wide = wide.sort_values("SAIL", ascending=False)
    write_table(wide.reset_index(), cfg.out / "table_v_dice.csv", "Supp. Table V")
    tests = pd.DataFrame([dict(comparison=f"SAIL > {m}", p_wilcoxon=paired_wilcoxon(wide["SAIL"], wide[m]))
                          for m in METHODS[1:]])
    write_table(tests, cfg.out / "table_v_wilcoxon.csv", "one-sided paired Wilcoxon")


# --------------------------------------------------------------------------- #
# ablation
# --------------------------------------------------------------------------- #
def ablation(cfg: ExperimentConfig) -> None:
    adata = load_adata(cfg)
    region_masks = regions(cfg, adata)

    def score(run_cfg, modes, graphs):
        sail.set_seed(cfg.seed)
        dice, _ = sail_dice(run_cfg, modes, graphs[0], region_masks)
        return dict(mean_best_dice=float(best_per_region(dice)["dice"].mean()))

    summary = run_ablation(cfg, lambda c: build(c, adata), score)
    write_table(summary, cfg.out / "ablation" / "ablation_summary.csv", "Supp. Table XI")


STAGES = {"train": train, "eval": evaluate, "baselines": baselines, "table": table, "ablation": ablation}
DEFAULT_STAGES = ["train", "eval", "baselines", "table"]
