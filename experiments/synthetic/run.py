"""Synthetic experiments (Section V-A; Supplementary Sections G and H).

Stages (``sail-run experiments/synthetic/config.yaml --stage <name>``):

* ``index``      Fig. 2: SAIL with an identity map reproduces Moran's I / Geary's C
                 on a random and a smooth-gradient graph, with LISA maps
* ``toys``       Supp. Fig. 4a: one learned Moran mode on Perlin, checkerboard,
                 plateau and concentric-ring grids (kNN graph)
* ``terrain``    Supp. Fig. 4b: a 40,000-point 3-D Perlin terrain (kNN graph)
* ``sweep``      regression benchmark stage 1: per-model width selection on one seed
* ``evaluate``   stage 2: every model at its selected width over many seeds
                 (Fig. 3, Supp. Table II, paired Wilcoxon tests)
* ``blend``      spatial-versus-content ablation (Supp. Table III, Fig. 3)
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

import regression as reg  # noqa: E402
import sail  # noqa: E402
from sail.config import ExperimentConfig  # noqa: E402
from sail.evaluation import fdr_bh  # noqa: E402
from sail.experiment import device_of, write_table  # noqa: E402
from sail.viz import overlay  # noqa: E402

log = logging.getLogger("sail")


# --------------------------------------------------------------------------- #
# Fig. 2: classical statistics as special cases
# --------------------------------------------------------------------------- #
def _grid(n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    xs = np.linspace(lo, hi, n)
    xv, yv = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xv.ravel(), yv.ravel()], 1)


def index(cfg: ExperimentConfig) -> None:
    c = cfg.eval["index"]
    sail.set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    random_xy = rng.uniform(size=(c["n_random"], 2))
    grid_xy = _grid(c["grid_size"])
    graphs = {"Random": (random_xy, rng.uniform(size=(c["n_random"], 1))),
              "Homogeneous": (grid_xy, grid_xy.sum(1, keepdims=True))}
    rows = []
    fig, axs = plt.subplots(2, 2, figsize=(11, 10))
    for col, (name, (xy, x)) in enumerate(graphs.items()):
        g = sail.build_graph(xy, x.astype(np.float32), k=c["k"], log_distances=False)
        row = dict(graph=name)
        for stat, std in [("moran", "population"), ("geary", "sample")]:  # Theorem IV.1
            model = sail.SAIL(local_index=stat, standardise=std, learnable_temperature=False)
            row[stat] = float(model(g).global_scores[0, 0])
        rows.append(row)
        model = sail.SAIL(local_index="moran", learnable_temperature=False)
        overlay(xy, x[:, 0], ax=axs[0, col], point_size=30, title=f"{name}: node values")
        overlay(xy, sail.lisa(g, model, modes=0), kind="lisa", ax=axs[1, col], point_size=30,
                title=f"Moran's I {row['moran']:.2f}, Geary's C {row['geary']:.2f}")
    fig.savefig(cfg.out / "fig2_index.png", dpi=300, bbox_inches="tight")
    write_table(pd.DataFrame(rows), cfg.out / "fig2_index.csv", "Fig. 2 global statistics")


# --------------------------------------------------------------------------- #
# Supp. Fig. 4: toy graphs and 3-D terrain
# --------------------------------------------------------------------------- #
def _perlin(n: int, res: int) -> np.ndarray:
    """2-D Perlin noise on an ``n x n`` grid with ``res x res`` gradient cells (``n`` divisible by ``res``).

    Same algorithm and RNG use as ``perlin_numpy.generate_perlin_noise_2d``.
    """
    d = n // res
    grid = np.mgrid[0:res:res / n, 0:res:res / n].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * np.random.rand(res + 1, res + 1)
    grad = np.dstack((np.cos(angles), np.sin(angles))).repeat(d, 0).repeat(d, 1)
    gx, gy = grid[..., 0], grid[..., 1]

    def corner(g, dx, dy):
        return (gx - dx) * g[..., 0] + (gy - dy) * g[..., 1]

    n00, n10 = corner(grad[:-d, :-d], 0, 0), corner(grad[d:, :-d], 1, 0)
    n01, n11 = corner(grad[:-d, d:], 0, 1), corner(grad[d:, d:], 1, 1)
    t = 6 * grid**5 - 15 * grid**4 + 10 * grid**3
    n0 = n00 * (1 - t[..., 0]) + t[..., 0] * n10
    n1 = n01 * (1 - t[..., 0]) + t[..., 0] * n11
    return np.sqrt(2) * ((1 - t[..., 1]) * n0 + t[..., 1] * n1)


def _fit_one_mode(cfg, g):
    sail.set_seed(cfg.seed)
    return sail.SAILModes(g.num_features, cfg.sail, device=device_of(cfg)).fit([g], verbose=False)


def toys(cfg: ExperimentConfig) -> None:
    t = cfg.eval["toys"]
    sail.set_seed(cfg.seed)
    n = t["perlin_size"]
    checker = np.indices((t["checker_size"],) * 2).sum(0) % 2
    plateau = _grid(t["plateau_size"])
    rings = _grid(t["rings_size"], -1, 1)
    cases = {
        "Perlin noise": (np.indices((n, n)).reshape(2, -1).T / (n - 1), _perlin(n, t["perlin_res"]).ravel()),
        "Checkerboard": (np.indices(checker.shape).reshape(2, -1).T / (t["checker_size"] - 1), checker.ravel()),
        "Plateau": (plateau, (plateau[:, 0] > 0.5).astype(float)),
        "Concentric rings": (rings, np.sin(8 * np.linalg.norm(rings, axis=1))),
    }
    fig, axs = plt.subplots(2, 4, figsize=(22, 10))
    for col, (name, (xy, v)) in enumerate(cases.items()):
        g = sail.build_graph(xy, v[:, None].astype(np.float32), **t["graph"])
        modes = _fit_one_mode(cfg, g)
        labels = sail.lisa(g, modes.model, modes=0)
        overlay(xy, modes.transform(g).z[0][:, 0].cpu(), ax=axs[0, col], point_size=8, title=f"{name}: f_map(x)")
        overlay(xy, labels, kind="lisa", ax=axs[1, col], point_size=8, title=f"{name}: LISA")
    fig.savefig(cfg.out / "supp_fig4a_toys.png", dpi=300, bbox_inches="tight")


def terrain(cfg: ExperimentConfig) -> None:
    t = cfg.eval["terrain"]
    sail.set_seed(cfg.seed)
    n = t["size"]
    xy = _grid(n)
    h = _perlin(n, t["perlin_res"]).ravel()
    pts = np.column_stack([xy, h])
    g = sail.build_graph(pts, h[:, None].astype(np.float32), k=t["k"])
    modes = _fit_one_mode(cfg, g)
    labels = sail.lisa(g, modes.model, modes=0, permutations=t["permutations"])
    fig = plt.figure(figsize=(14, 6))
    for i, (vals, kind, title) in enumerate([(modes.transform(g).z[0][:, 0].cpu(), "continuous", "f_map(x)"),
                                             (labels, "lisa", "LISA")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        overlay(pts, vals, kind=kind, ax=ax, point_size=1, title=title)
    fig.savefig(cfg.out / "supp_fig4b_terrain.png", dpi=300, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# regression benchmark
# --------------------------------------------------------------------------- #
def _specs(cfg):
    r = cfg.eval["regression"]
    return reg.model_specs(r["hidden_grid"], r["heads_grid"])


def _fit(cfg, model, splits, use_coords):
    r = cfg.eval["regression"]
    return reg.fit(model, *splits, use_coords, lr=r["lr"], max_epochs=r["max_epochs"], patience=r["patience"],
                   device=device_of(cfg))


def sweep(cfg: ExperimentConfig) -> None:
    """Stage 1: pick each model's width on one seed by validation Pearson (ties -> smaller model)."""
    r = cfg.eval["regression"]
    splits = reg.make_splits(r["data"], r["sweep_seed"])
    rows, selected = [], {}
    for name, spec in _specs(cfg).items():
        best = None
        for hp in spec.grid:
            sail.set_seed(r["sweep_seed"])
            res = _fit(cfg, spec.build(hp, r["data"]["x_dim"]), splits, spec.use_coords)
            rows.append(dict(model=name, **hp, val_pearson=res["val"]["pearson"], params=res["params"]))
            if best is None or res["val"]["pearson"] > best[1] + r["tie_margin"]:
                best = (hp, res["val"]["pearson"])
        selected[name] = best[0]
    write_table(pd.DataFrame(rows), cfg.out / "sweep_results.csv", "stage 1 sweep")
    (cfg.out / "selected.json").write_text(json.dumps(selected, indent=2))


def evaluate(cfg: ExperimentConfig) -> None:
    """Stage 2: retrain the selected configs on ``n_seeds`` fresh datasets."""
    from scipy.stats import wilcoxon

    r = cfg.eval["regression"]
    selected = json.loads((cfg.out / "selected.json").read_text())
    specs = _specs(cfg)
    out_csv = cfg.out / "seed_results.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    plan = json.loads((cfg.path_any(r["seeds_file"])).read_text())["seeds"][: r["n_seeds"]]
    seeds = [s for s in plan if done.empty or s not in set(done["seed"])]
    for seed in seeds:  # resumable: one row per (seed, model) appended as it finishes
        splits = reg.make_splits(r["data"], seed)
        rows = []
        for name, spec in specs.items():
            res = _fit(cfg, spec.build(selected[name], r["data"]["x_dim"]), splits, spec.use_coords)
            rows.append(dict(seed=seed, model=name, **res["test"], params=res["params"]))
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)

    df = pd.read_csv(out_csv)
    summary = df.groupby("model").agg(params=("params", "first"), pearson_mean=("pearson", "mean"),
                                      pearson_std=("pearson", "std"), mse_mean=("mse", "mean"),
                                      mse_std=("mse", "std")).sort_values("pearson_mean", ascending=False)
    write_table(summary.reset_index(), cfg.out / "table_ii.csv", "Supp. Table II")
    wide = df.pivot(index="seed", columns="model", values="pearson")
    others = [m for m in wide.columns if m != "SAIL (dense)"]
    p = [wilcoxon(wide["SAIL (dense)"], wide[m], alternative="greater").pvalue for m in others]
    write_table(pd.DataFrame(dict(comparison=[f"SAIL > {m}" for m in others], p=p, p_fdr_bh=fdr_bh(np.array(p)))),
                cfg.out / "wilcoxon.csv", "paired Wilcoxon (Pearson)")

    fig, ax = plt.subplots(figsize=(7, 4))
    order = summary.index.tolist()
    ax.boxplot([df.loc[df.model == m, "pearson"] for m in order], vert=False, labels=order)
    ax.set_xlabel("Pearson correlation")
    fig.savefig(cfg.out / "fig3_regression.png", dpi=300, bbox_inches="tight")


def blend(cfg: ExperimentConfig) -> None:
    """Target = (1 - gamma) * z(Moran's I) + gamma * z(feature second moment)."""
    r, b = cfg.eval["regression"], cfg.eval["blend"]
    v = torch.tensor(np.random.default_rng(b["content_seed"]).standard_normal(r["data"]["x_dim"]), dtype=torch.float32)
    v = v / v.norm()
    specs = {name: spec for name, spec in _specs(cfg).items() if name in b["models"]}
    rows = []
    for seed in range(b["n_seeds"]):
        splits = reg.make_splits(r["data"], seed, content_direction=v)
        s = np.array([x.y_spatial for x in splits[0]])
        f = np.array([x.y_feat for x in splits[0]])
        ms, ss, mf, sf = s.mean(), s.std() + 1e-8, f.mean(), f.std() + 1e-8   # train statistics only
        for gamma in b["gammas"]:
            for split in splits:
                for x in split:
                    x.y = torch.tensor([(1 - gamma) * (x.y_spatial - ms) / ss + gamma * (x.y_feat - mf) / sf])
            for name, spec in specs.items():
                sail.set_seed(seed)
                res = _fit(cfg, spec.build(dict(hidden=b["hidden"], heads=b["heads"]), r["data"]["x_dim"]),
                           splits, spec.use_coords)
                rows.append(dict(seed=seed, gamma=gamma, model=name, **res["test"]))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out / "blend_raw.csv", index=False)
    summary = df.groupby(["model", "gamma"])["pearson"].agg(["mean", "std"]).reset_index()
    write_table(summary, cfg.out / "table_iii_blend.csv", "Supp. Table III")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, sub in summary.groupby("model"):
        ax.plot(sub["gamma"], sub["mean"], "-o", label=name)
        ax.fill_between(sub["gamma"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.15)
    ax.set_xlabel("gamma (0 = spatial autocorrelation, 1 = feature content)")
    ax.set_ylabel("test Pearson")
    ax.legend()
    fig.savefig(cfg.out / "supp_fig3_blend.png", dpi=300, bbox_inches="tight")


STAGES = {"index": index, "toys": toys, "terrain": terrain, "sweep": sweep, "evaluate": evaluate, "blend": blend}
DEFAULT_STAGES = ["index", "toys", "terrain", "sweep", "evaluate", "blend"]
