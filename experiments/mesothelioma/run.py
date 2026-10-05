"""Supervised SAIL for mesothelioma subtyping (epithelioid vs non-epithelioid).

Leave-one-TMA-slide-out cross-validation on the St George cohort with the
graphs, folds and ranking loss of MesoGraph. A shared SAIL backbone yields 16
global SAC descriptors per core; an epithelioid (E) and a non-epithelioid (NE)
linear head read them, and the core score is ``NE - E``.

Stages (``sail-run experiments/mesothelioma/config.yaml --stage <name>``):

* ``splits``     regenerate the shared folds (the paper's folds ship as splits.json)
* ``train``      train SAIL on every fold, then score it
* ``eval``       score the fold models (trained here, or the shipped paper models)
* ``baselines``  MesoGraph, PINS, Max-MIL (and naive-MIL) on the same folds (see baselines.py)
* ``table``      Table IV: per-fold mean +/- std AUC and AP for every method
* ``stats``      paired fold-stratified bootstrap and DeLong tests, SAIL vs each baseline
* ``interpret``  Fig. 7: head weights, mode loadings and core maps for one fold
"""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import meso_data as md  # noqa: E402
import sail  # noqa: E402
from sail.config import ExperimentConfig  # noqa: E402
from sail.experiment import device_of, write_table  # noqa: E402
from sail.supervised import SAILPredictor  # noqa: E402
from sail.viz import overlay, plot_mode_weights  # noqa: E402

log = logging.getLogger("sail")


# --------------------------------------------------------------------------- #
def splits(cfg: ExperimentConfig) -> None:
    cores = md.load_cores(cfg.path("graphs"))
    out = cfg.out / "splits.json"
    out.write_text(json.dumps(md.make_splits(cores, cfg.train["val_frac"], cfg.seed), indent=1))
    log.warning("wrote %s (the paper's folds are in %s)", out, cfg.data["splits"])


def _sail_graphs(cfg):
    cores = md.load_cores(cfg.path("graphs"))
    graphs = {c: md.to_sail_graph(g, cfg.graph) for c, g in cores.items()}
    labels = {c: md.label(g) for c, g in cores.items()}
    return graphs, labels


def ranking_loss(scores: torch.Tensor, y: torch.Tensor, margin: float) -> torch.Tensor:
    """MesoGraph pairwise margin ranking loss.

    For every pair of cores with different labels, the E head (column 0) must
    rank the more epithelioid core higher and the NE head (column 1) the less
    epithelioid one, by at least ``margin``.
    """
    i, j = torch.triu_indices(len(y), len(y), offset=1, device=y.device)
    keep = y[i] != y[j]
    if not keep.any():
        return scores.new_zeros(())
    i, j = i[keep], j[keep]
    dz = scores[i] - scores[j]                                   # (P, 2)
    dy = torch.stack([(y[j] - y[i]).clamp(-1, 1), (y[i] - y[j]).clamp(-1, 1)], dim=1)
    return torch.clamp(margin - dy * dz, min=0).mean(dim=1).mean()


@torch.no_grad()
def core_scores(model, cores, graphs, labels, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for c in cores:
        out, _ = model(graphs[c].to(device))
        rows.append(dict(core=c, y=labels[c], y_pred=float(out[0, 1] - out[0, 0])))
    return pd.DataFrame(rows)


def _val_loss(model, cores, graphs, labels, device, margin) -> float:
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(graphs[c].to(device))[0] for c in cores])
        y = torch.tensor([labels[c] for c in cores], dtype=torch.float32, device=device)
        return float(ranking_loss(out, y, margin))


def train_fold(cfg, fold, graphs, labels, device) -> SAILPredictor:
    t = cfg.train
    sail.set_seed(cfg.seed + fold["fold"])
    in_dim = graphs[fold["train"][0]].num_features
    model = SAILPredictor(in_dim, **cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    train_c = fold["train"]
    y_tr = np.array([labels[c] for c in train_c])
    best, best_state = float("inf"), None
    for epoch in range(t["epochs"]):
        model.train()
        order = np.random.permutation(len(train_c))
        for b in range(0, len(order), t["batch_size"]):
            idx = order[b:b + t["batch_size"]]
            if len(np.unique(y_tr[idx])) < 2:
                continue
            opt.zero_grad()
            out = torch.cat([model(graphs[train_c[i]].to(device))[0] for i in idx])
            y = torch.tensor(y_tr[idx], dtype=torch.float32, device=device)
            ranking_loss(out, y, t["margin"]).backward()
            opt.step()
        val = _val_loss(model, fold["val"], graphs, labels, device, t["margin"])
        if val < best:  # model selection on validation loss
            best, best_state = val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0:
            log.info("fold %d epoch %d val loss %.4f (best %.4f)", fold["fold"], epoch, val, best)
    model.load_state_dict(best_state)
    return model


def _fold_model_path(cfg, k: int):
    trained = cfg.out / "sail" / f"sail_fold{k}.pt"
    return trained if trained.exists() else cfg.path("pretrained_dir") / f"sail_fold{k}.pt"


def train_sail(cfg: ExperimentConfig) -> None:
    """Train one SAIL model per fold (selection on validation ranking loss)."""
    device = device_of(cfg)
    graphs, labels = _sail_graphs(cfg)
    (cfg.out / "sail").mkdir(parents=True, exist_ok=True)
    for fold in md.load_splits(cfg.path("splits"))["folds"]:
        model = train_fold(cfg, fold, graphs, labels, device)
        torch.save(model.state_dict(), cfg.out / "sail" / f"sail_fold{fold['fold']}.pt")
    score_sail(cfg, (graphs, labels))


def score_sail(cfg: ExperimentConfig, data=None) -> None:
    """Validation/test AUC and AP of every fold's model (trained here, or the shipped ones)."""
    device = device_of(cfg)
    graphs, labels = data or _sail_graphs(cfg)
    out = cfg.out / "sail"
    out.mkdir(parents=True, exist_ok=True)
    preds, rows = [], []
    for fold in md.load_splits(cfg.path("splits"))["folds"]:
        model = SAILPredictor(graphs[fold["test"][0]].num_features, **cfg.model).to(device)
        model.load_state_dict(torch.load(_fold_model_path(cfg, fold["fold"]), map_location=device))
        for split in ("val", "test"):
            df = core_scores(model, fold[split], graphs, labels, device).assign(fold=fold["fold"])
            auc, ap = md.auc_ap(df["y"].map(md.binarize), df["y_pred"])
            rows.append(dict(method="sail", fold=fold["fold"], split=split, auc=auc, ap=ap))
            if split == "test":
                preds.append(df)
    pd.concat(preds).to_csv(out / "core_predictions.csv", index=False)
    write_table(pd.DataFrame(rows), out / "fold_metrics.csv", "SAIL per fold")


# --------------------------------------------------------------------------- #
def baselines(cfg: ExperimentConfig) -> None:
    import baselines as bl

    for name in cfg.eval["baselines"]:
        bl.RUNNERS[name](cfg)


def table(cfg: ExperimentConfig) -> None:
    rows = []
    for method in ["sail", *cfg.eval["baselines"]]:
        m = pd.read_csv(cfg.out / method / "fold_metrics.csv")
        m = m[m["split"] == "test"]
        rows.append(dict(method=method, auc_mean=m["auc"].mean(), auc_std=m["auc"].std(ddof=0),
                         ap_mean=m["ap"].mean(), ap_std=m["ap"].std(ddof=0), n_folds=len(m)))
    write_table(pd.DataFrame(rows), cfg.out / "table_iv.csv", "Table IV (per-fold mean +/- std)")


def stats(cfg: ExperimentConfig) -> None:
    """Paired, fold- and class-stratified bootstrap of the mean per-fold AUC difference."""
    methods = ["sail", *cfg.eval["baselines"]]
    P = {m: pd.read_csv(cfg.out / m / "core_predictions.csv").set_index("core") for m in methods}
    base = P["sail"]
    folds = sorted(base["fold"].unique())
    ys, Ss = [], []
    for f in folds:
        cores = base.index[base["fold"] == f]
        ys.append(base.loc[cores, "y"].map(md.binarize).to_numpy())
        Ss.append(np.vstack([P[m].loc[cores, "y_pred"].to_numpy() for m in methods]))
    rng = np.random.default_rng(cfg.seed)
    n_boot = cfg.eval["n_bootstrap"]
    boot = np.zeros((n_boot, len(methods)))
    for b in range(n_boot):
        for y, S in zip(ys, Ss):
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            ii = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            boot[b] += _auc_rows(y[ii], S[:, ii])
    boot /= len(folds)
    observed = np.mean([_auc_rows(y, S) for y, S in zip(ys, Ss)], axis=0)
    rows = []
    for j, m in enumerate(methods[1:], start=1):
        d = boot[:, 0] - boot[:, j]
        rows.append(dict(comparison=f"SAIL vs {m}", auc_sail=observed[0], auc_other=observed[j],
                         delta_auc=observed[0] - observed[j],
                         ci95_lo=np.percentile(d, 2.5), ci95_hi=np.percentile(d, 97.5),
                         p_one_sided=(np.sum(d <= 0) + 1) / (n_boot + 1),
                         p_two_sided=min(1.0, 2 * min(np.sum(d <= 0) + 1, np.sum(d >= 0) + 1) / (n_boot + 1))))
    write_table(pd.DataFrame(rows), cfg.out / "stats_sail_vs_baselines.csv", "paired bootstrap")


def _auc_rows(y, S):
    from scipy.stats import rankdata

    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    R = np.apply_along_axis(rankdata, 1, S)
    return (R[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------- #
def interpret(cfg: ExperimentConfig) -> None:
    """Fig. 7: the two most influential SAIL Heads of one fold's model and two test cores."""
    device = device_of(cfg)
    k = cfg.eval["interpret_fold"]
    fold = md.load_splits(cfg.path("splits"))["folds"][k]
    graphs, labels = _sail_graphs(cfg)
    in_dim = graphs[fold["test"][0]].num_features
    model = SAILPredictor(in_dim, **cfg.model).to(device)
    model.load_state_dict(torch.load(_fold_model_path(cfg, k), map_location=device))
    w = model.head_weights()
    effective = (w[1] - w[0]).cpu().numpy()  # NE - E
    top = np.argsort(-np.abs(effective))[:2]
    names = [f"f{i}" for i in range(in_dim)]

    scores = core_scores(model, fold["test"], graphs, labels, device).sort_values("y_pred")
    examples = [scores.iloc[0]["core"], scores.iloc[-1]["core"]]
    fig, axs = plt.subplots(3, 3, figsize=(15, 13))
    for r, h in enumerate(top):
        plot_mode_weights(model.sail.f_map.W, names, int(h), ax=axs[r, 0],
                          title=f"SAIL Head {h + 1} (weight {effective[h]:+.2f})")
    axs[2, 0].bar(range(len(effective)), effective)
    axs[2, 0].set_title("effective head weights (NE - E)")
    for c, core in enumerate(examples, start=1):
        g = graphs[core].to(device)
        with torch.no_grad():
            _, node = model(g)
            z = model.sail(g).z[0]
        overlay(g.coords, (node[:, 1] - node[:, 0]).cpu(), ax=axs[0, c], point_size=4,
                title=f"{core} (label {labels[core]}): local score")
        for r, h in enumerate(top, start=1):
            overlay(g.coords, z[:, h].cpu(), ax=axs[r, c], point_size=4, title=f"f_map, Head {h + 1}")
    fig.savefig(cfg.out / "fig7.png", dpi=300, bbox_inches="tight")


STAGES = {"splits": splits, "train": train_sail, "eval": score_sail, "baselines": baselines, "table": table,
          "stats": stats, "interpret": interpret}
DEFAULT_STAGES = ["train", "baselines", "table", "stats"]
