"""cSCC PD-1/PD-L1 blockade: do unsupervised SAIL niches separate responders?

Each patient is one block-diagonal graph whose tissue cores are disconnected
kNN blocks. 20 modes are learned on all baseline (pre-treatment) patients
jointly without labels; each mode's global score is then compared between
responders and non-responders.

Stages (``sail-run experiments/pd1_cscc/config.yaml --stage <name>``):

* ``train``       learn the modes
* ``eval``        per-subject scores, Mann-Whitney / AUC per mode (Supp. Table VI),
                  bootstrap CI of the best mode, Fig. 5 panels, gene loadings (Supp. Fig. 7)
* ``biomarkers``  published comparators: favourable-niche proportion, clinical PD-L1 TPS (Table II)
* ``ablation``    Supp. Table XII
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sail  # noqa: E402
from sail.config import ExperimentConfig  # noqa: E402
from sail.evaluation import association_table, bootstrap_auc_ci  # noqa: E402
from sail.experiment import load_modes, run_ablation, train_modes, write_table  # noqa: E402
from sail.viz import overlay, plot_mode_weights  # noqa: E402

log = logging.getLogger("sail")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_cohort(cfg: ExperimentConfig, qc: bool = True):
    """Labelled patients at the configured timepoints, optionally after cell/gene QC."""
    import anndata as ad
    import scanpy as sc

    d = cfg.data
    adata = ad.read_h5ad(cfg.path("path"))
    keep = adata.obs[d["time_col"]].isin(d["timepoints"]) & adata.obs["response"].isin([d["positive"], d["negative"]])
    adata = adata[keep].copy()
    if qc:  # genes first, then cells (order matters)
        sc.pp.filter_genes(adata, min_cells=d["gene_min_cells"])
        sc.pp.filter_cells(adata, min_counts=d["cell_min_counts"])
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[d["coords_cols"]].to_numpy(np.float32)
    return adata


def gene_features(cfg: ExperimentConfig, adata) -> np.ndarray:
    """log1p(CP10k) of all genes, z-scored over all cells and clipped at ``max_zscore``."""
    import scanpy as sc

    norm = adata.copy()
    sc.pp.normalize_total(norm, target_sum=1e4)
    sc.pp.log1p(norm)
    X = norm.X.toarray() if hasattr(norm.X, "toarray") else np.asarray(norm.X)
    X = X.astype(np.float32)
    mu = X.mean(0, dtype=np.float64)
    sd = np.maximum(X.std(0, dtype=np.float64), 1e-8)
    return np.clip((X - mu.astype(np.float32)) / sd.astype(np.float32), None, cfg.data["max_zscore"])


def build(cfg: ExperimentConfig):
    """One block-diagonal graph per subject; cores with too few cells are dropped."""
    d = cfg.data
    adata = read_cohort(cfg)
    X = gene_features(cfg, adata)
    coords = adata.obsm["spatial"]
    obs = adata.obs
    graphs, rows = [], []
    for subject, idx in sorted(obs.groupby(d["subject_col"], observed=True).indices.items()):
        parts = []
        for core in sorted(obs.iloc[idx][d["core_col"]].unique()):
            rows_core = idx[(obs.iloc[idx][d["core_col"]] == core).to_numpy()]
            if len(rows_core) >= d["core_min_cells"]:
                parts.append(sail.build_graph(coords[rows_core], X[rows_core], **cfg.graph))
        if not parts:
            continue
        graphs.append(sail.concat_graphs(parts, meta={"id": str(subject)}))
        rows.append(dict(subject=str(subject), response=str(obs.iloc[idx]["response"].iloc[0]),
                         n_cores=len(parts), n_cells=graphs[-1].num_nodes))
    meta = pd.DataFrame(rows)
    for g in graphs:
        g.meta["feature_names"] = list(map(str, adata.var_names))
    log.warning("%d subjects, %d genes, %s", len(graphs), X.shape[1], meta.response.value_counts().to_dict())
    return graphs, meta


# --------------------------------------------------------------------------- #
# SAIL
# --------------------------------------------------------------------------- #
def train(cfg: ExperimentConfig) -> None:
    graphs, _ = build(cfg)
    train_modes(cfg, graphs)


def score(cfg: ExperimentConfig, modes: sail.SAILModes, graphs, meta: pd.DataFrame):
    """Per-subject scores and the association table (1-based mode numbers)."""
    scores = modes.scores(graphs)
    cols = [f"Mode {m + 1}" for m in range(scores.shape[1])]
    per_subject = pd.concat([meta, pd.DataFrame(scores, columns=cols)], axis=1)
    y = (per_subject["response"] == cfg.data["positive"]).to_numpy(int)
    return per_subject, association_table(per_subject, y, cols), y


def best_mode(cfg, per_subject, table, y) -> dict:
    """Highest orientation-free AUC (ties: lower p) with its bootstrap CI."""
    top = table.sort_values(["auc_flip", "pvalue"], ascending=[False, True]).iloc[0]
    lo, hi = bootstrap_auc_ci(y, per_subject[top["score"]].to_numpy(float), bool(top["higher_in_positive"]),
                              n_boot=cfg.eval["n_bootstrap"], seed=cfg.seed)
    return dict(best_mode=top["score"], auc=float(top["auc_flip"]), auc_ci95_lo=lo, auc_ci95_hi=hi,
                pvalue=float(top["pvalue"]), higher_in_responders=bool(top["higher_in_positive"]))


def evaluate(cfg: ExperimentConfig) -> None:
    graphs, meta = build(cfg)
    modes = load_modes(cfg)
    per_subject, table, y = score(cfg, modes, graphs, meta)
    write_table(per_subject, cfg.out / "per_subject_scores.csv")
    write_table(table, cfg.out / "response_association.csv", "Supp. Table VI")
    write_table(pd.DataFrame([best_mode(cfg, per_subject, table, y)]), cfg.out / "best_mode.csv", "best mode")

    genes = graphs[0].meta["feature_names"]
    W = pd.DataFrame(modes.W.cpu().numpy(), index=genes, columns=[f"Mode {m + 1}" for m in range(modes.W.shape[1])])
    W.to_csv(cfg.out / "mode_weights.csv")

    m = cfg.eval["figure_mode"]
    col = f"Mode {m}"
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"width_ratios": [1, 2, 2]})
    groups = [cfg.data["negative"], cfg.data["positive"]]
    data = [per_subject.loc[per_subject.response == g, col] for g in groups]
    axs[0].boxplot(data, labels=groups)
    for j, dvals in enumerate(data):
        axs[0].scatter(np.random.default_rng(j).normal(j + 1, 0.05, len(dvals)), dvals, s=20, alpha=0.7)
    axs[0].set_ylabel(f"global SAIL score, {col}")
    plot_mode_weights(modes.W, genes, m - 1, ax=axs[1], top_k=cfg.eval["top_genes"], title=f"{col} gene weights")
    example = int(np.argmax(per_subject[col].to_numpy()))
    out = modes.transform(graphs[example])
    overlay(graphs[example].coords, out.z[0][:, m - 1], ax=axs[2], point_size=2,
            title=f"{col} field, subject {per_subject.subject[example]}")
    fig.savefig(cfg.out / "fig5.png", dpi=300, bbox_inches="tight")

    n_modes = modes.W.shape[1]
    fig, axs = plt.subplots(int(np.ceil(n_modes / 4)), 4, figsize=(20, 3.2 * np.ceil(n_modes / 4)), squeeze=False)
    for k, ax in enumerate(axs.flat):
        if k < n_modes:
            plot_mode_weights(modes.W, genes, k, ax=ax, top_k=cfg.eval["top_genes"], title=f"Mode {k + 1}")
        else:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(cfg.out / "all_mode_weights.png", dpi=300, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# published biomarkers (Table II comparators)
# --------------------------------------------------------------------------- #
def biomarkers(cfg: ExperimentConfig) -> None:
    """Favourable-niche proportion and clinical PD-L1 TPS, as defined in the original study.

    Computed on the unfiltered labelled cohort (no SAIL QC), reproducing the published AUCs.
    """
    import anndata as ad

    d = cfg.data
    full = ad.read_h5ad(cfg.path("path"))
    labelled = full.obs[full.obs["response"].isin([d["positive"], d["negative"]])]
    base = labelled[labelled[d["time_col"]].isin(d["timepoints"])]
    rows = []

    # 1. proportion of cells in the favourable niches, baseline biopsies
    niche = base[d["niche_col"]].astype(str)
    frac = pd.crosstab(base[d["subject_col"]], niche, normalize="index")
    fav = frac.reindex(columns=cfg.eval["favourable_niches"], fill_value=0.0).sum(1)
    resp = base.drop_duplicates(d["subject_col"]).set_index(d["subject_col"])["response"]
    rows.append(_biomarker_row(cfg, "Favourable niches (six-niche classifier)", fav, resp.loc[fav.index]))

    # 2. clinical PD-L1 TPS: all patients with a known value, and patients with a baseline slide
    per_patient = labelled.drop_duplicates(d["subject_col"]).set_index(d["subject_col"])
    tps = per_patient[d["pdl1_col"]].map(_tps).dropna()
    rows.append(_biomarker_row(cfg, "PD-L1 TPS (all slides)", tps, per_patient.loc[tps.index, "response"]))
    with_baseline = tps[tps.index.isin(base[d["subject_col"]].unique())]
    rows.append(_biomarker_row(cfg, "PD-L1 TPS (untreated slide)", with_baseline,
                               per_patient.loc[with_baseline.index, "response"]))
    write_table(pd.DataFrame(rows), cfg.out / "published_biomarkers.csv", "Table II comparators")


def _tps(v) -> float:
    v = str(v)
    if v in ("Unknown", "nan", "__NA__"):
        return np.nan
    return 0.0 if v.startswith("<") else float(v)


def _biomarker_row(cfg, name: str, values: pd.Series, response: pd.Series) -> dict:
    y = (response.to_numpy() == cfg.data["positive"]).astype(int)
    t = association_table(pd.DataFrame({name: values.to_numpy(float)}), y, [name]).iloc[0]
    lo, hi = bootstrap_auc_ci(y, values.to_numpy(float), True, n_boot=cfg.eval["n_bootstrap"], seed=cfg.seed)
    return dict(biomarker=name, n_responders=int(y.sum()), n_nonresponders=int((1 - y).sum()),
                auc=float(t["auc"]), auc_ci95_lo=lo, auc_ci95_hi=hi, pvalue=float(t["pvalue"]))


# --------------------------------------------------------------------------- #
def ablation(cfg: ExperimentConfig) -> None:
    metas = {}

    def build_cached(run_cfg):
        graphs, meta = build(run_cfg)
        metas[id(graphs)] = meta
        return graphs

    def summarise(run_cfg, modes, graphs):
        per_subject, table, y = score(run_cfg, modes, graphs, metas[id(graphs)])
        return best_mode(run_cfg, per_subject, table, y)

    write_table(run_ablation(cfg, build_cached, summarise), cfg.out / "ablation" / "ablation_summary.csv",
                "Supp. Table XII")


STAGES = {"train": train, "eval": evaluate, "biomarkers": biomarkers, "ablation": ablation}
DEFAULT_STAGES = ["train", "eval", "biomarkers"]
