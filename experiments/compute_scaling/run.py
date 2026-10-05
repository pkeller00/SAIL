"""Computational scaling of SAIL (Supplementary Section F, Table I, Figs. 1-2).

Wall-clock time per training step, inference time and peak GPU memory as one
factor (nodes ``n``, features ``d``, modes ``h``, neighbours ``k``) is varied
around a reference configuration, for all six combinations of feature map
(linear / orthogonal) and feature affinity (none / shared map / query-key).

Stages (``sail-run experiments/compute_scaling/config.yaml --stage <name>``):

* ``benchmark``  run the sweeps (resumable; raw rows appended to ``raw.csv``)
* ``table``      Supp. Table I and per-configuration summaries
* ``plots``      Supp. Figs. 1-2
"""

from __future__ import annotations

import gc
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import sail  # noqa: E402
from sail.config import ExperimentConfig  # noqa: E402
from sail.experiment import device_of, write_table  # noqa: E402

log = logging.getLogger("sail")
VARIANTS = {  # name -> (stiefel, f_affinity, label)
    "linear": (False, None, "Linear, no affinity"),
    "ortho": (True, None, "Orthogonal, no affinity (default)"),
    "linear_shared": (False, "shared_map", "Linear, shared-map"),
    "ortho_shared": (True, "shared_map", "Orthogonal, shared-map"),
    "linear_qk": (False, "query_key", "Linear, query-key"),
    "ortho_qk": (True, "query_key", "Orthogonal, query-key"),
}
COLUMNS = ["sweep", "variant", "oom", "n", "d", "h", "k", "repeat", "params", "train_step_s", "train_peak_mb",
           "infer_s", "infer_peak_mb", "fit_s", "knn_s"]


def synthetic(n: int, d: int, rng: np.random.Generator, rank: int = 10, n_freq: int = 64, noise: float = 0.5):
    """Uniform points in a square of side sqrt(n); features = rank-``rank`` smooth random-Fourier fields + noise."""
    side = np.sqrt(n)
    xy = rng.uniform(0, side, size=(n, 2)).astype(np.float32)
    omega = rng.normal(0, 8.0 / side, size=(2, n_freq)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=n_freq).astype(np.float32)
    fields = (np.cos(xy @ omega + phase) * np.sqrt(2.0 / n_freq)) @ rng.normal(size=(n_freq, rank)).astype(np.float32)
    X = fields @ rng.normal(size=(rank, d)).astype(np.float32) + noise * rng.standard_normal((n, d), dtype=np.float32)
    return xy, (X - X.mean(0)) / (X.std(0) + 1e-8)


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset(device):
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device):
    return torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else np.nan


def _is_oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def measure(cfg, variant: str, g: sail.SpatialGraph, d: int, h: int, fit: bool) -> list[dict]:
    """Repeated timing of training steps, inference and (optionally) an end-to-end fit."""
    b, device = cfg.eval, device_of(cfg)
    stiefel, affinity, _ = VARIANTS[variant]
    sail_cfg = sail.ModesConfig(**{**cfg.sail.__dict__, "n_modes": h, "stiefel": stiefel, "f_affinity": affinity})
    rows = [dict(repeat=r) for r in range(b["repeats"])]

    _reset(device)
    modes = sail.SAILModes(d, sail_cfg, device=device)
    modes.model.train()
    t0 = time.perf_counter()
    for _ in range(b["warmup"]):
        modes.step([g], record=False)
    _sync(device)
    per_step = (time.perf_counter() - t0) / b["warmup"]
    steps = int(np.clip(b["max_repeat_seconds"] / max(per_step, 1e-9), 1, b["steps_per_repeat"]))
    for r in rows:
        _reset(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            modes.step([g], record=False)
        _sync(device)
        r.update(train_step_s=(time.perf_counter() - t0) / steps, train_peak_mb=_peak_mb(device),
                 params=sum(p.numel() for p in modes.model.parameters() if p.requires_grad))

    # inference with the exported (plain linear) map, as left by SAILModes.fit
    model = modes.model.eval()
    if stiefel:
        lin = sail.NormalisedLinear(d, h).to(device)
        lin.W = model.f_map.W.detach()
        model.f_map = lin
        if isinstance(model.f_affinity, sail.SharedMapAffinity):
            model.f_affinity.f_map = lin
    with torch.no_grad():
        model(g)
        for r in rows:
            _sync(device)
            _reset(device)
            t0 = time.perf_counter()
            model(g)
            _sync(device)
            r.update(infer_s=time.perf_counter() - t0, infer_peak_mb=_peak_mb(device))
    del modes, model

    if fit:
        for r in rows[: b["fit_repeats"]]:
            _reset(device)
            m = sail.SAILModes(d, sail.ModesConfig(**{**sail_cfg.__dict__, "epochs": b["fit_epochs"],
                                                     "min_delta": -float("inf")}), device=device)
            _sync(device)
            t0 = time.perf_counter()
            m.fit([g], verbose=False)
            _sync(device)
            r.update(fit_s=time.perf_counter() - t0)
            del m
    _reset(device)
    return rows


def benchmark(cfg: ExperimentConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False  # FP32 throughout, as reported
    torch.backends.cudnn.allow_tf32 = False
    b, device = cfg.eval, device_of(cfg)
    raw_path = cfg.out / "raw.csv"
    done = set()
    if raw_path.exists():
        prev = pd.read_csv(raw_path)
        done = set(zip(prev["sweep"], prev["variant"], prev["n"], prev["d"], prev["h"], prev["k"]))
    rng = np.random.default_rng(cfg.seed)
    ref = b["reference"]
    data_cache: dict = {}
    for factor, values in b["sweeps"].items():
        oom = set()
        for v in values:
            c = dict(ref, **{factor: v})
            key = (c["n"], c["d"])
            if key not in data_cache:
                data_cache.clear()
                data_cache[key] = synthetic(c["n"], c["d"], rng)
            xy, X = data_cache[key]
            rows = []
            if factor in ("n", "k"):  # graph construction cost (CPU k-d tree)
                for rep in range(b["repeats"]):
                    t0 = time.perf_counter()
                    _knn_only(xy, c["k"])
                    rows.append(dict(sweep=factor, variant="knn_graph", oom=False, repeat=rep, **c,
                                     knn_s=time.perf_counter() - t0))
            g = sail.build_graph(xy, X, k=c["k"], backend="sklearn").to(device)
            for variant in VARIANTS:
                if (factor, variant, c["n"], c["d"], c["h"], c["k"]) in done or variant in oom:
                    continue
                fit = factor == "n" and c["n"] <= b["fit_max_n"]
                try:
                    rows += [dict(sweep=factor, variant=variant, oom=False, **c, **r)
                             for r in measure(cfg, variant, g, c["d"], c["h"], fit)]
                except Exception as e:  # noqa: BLE001
                    if not _is_oom(e):
                        raise
                    log.warning("%s OOM at %s", variant, c)
                    oom.add(variant)  # sweeps are monotone: skip larger values
                    rows.append(dict(sweep=factor, variant=variant, oom=True, **c))
                    _reset(device)
            del g
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(raw_path, mode="a", header=not raw_path.exists(),
                                                                index=False)


def _knn_only(xy: np.ndarray, k: int) -> None:
    from sklearn.neighbors import KDTree

    KDTree(xy).query(xy, k=k + 1)


def table(cfg: ExperimentConfig) -> None:
    raw = pd.read_csv(cfg.out / "raw.csv")
    ref = cfg.eval["reference"]  # noqa: F841  (used in the query string below)
    ok = raw[(raw["variant"] != "knn_graph") & (raw["oom"] == False)]  # noqa: E712
    keys = ["sweep", "variant", "n", "d", "h", "k"]
    summary = ok.groupby(keys)[["train_step_s", "infer_s", "train_peak_mb", "fit_s"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(cfg.out / "summary.csv", index=False)

    at_ref = summary.reset_index().query("sweep == 'n' and n == @ref['n']").set_index("variant")
    max_n = ok[ok["sweep"] == "n"].groupby("variant")["n"].max()
    rows = []
    for v, (_, _, label) in VARIANTS.items():
        r = at_ref.loc[v]
        rows.append({"variant": label,
                     "train step (ms)": f"{1e3 * r.train_step_s_mean:.1f} ± {1e3 * r.train_step_s_std:.1f}",
                     "inference (ms)": f"{1e3 * r.infer_s_mean:.1f} ± {1e3 * r.infer_s_std:.1f}",
                     "peak memory (GB)": f"{r.train_peak_mb_mean / 1024:.2f}",
                     "max n": int(max_n.get(v, 0))})
    write_table(pd.DataFrame(rows), cfg.out / "table_i.csv", "Supp. Table I")


def plots(cfg: ExperimentConfig) -> None:
    s = pd.read_csv(cfg.out / "summary.csv")
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for col, factor in enumerate(["n", "d", "h", "k"]):
        for row, metric in enumerate(["train_step_s", "train_peak_mb"]):
            ax = axs[row, col]
            for v, (_, _, label) in VARIANTS.items():
                sub = s[(s.sweep == factor) & (s.variant == v)].sort_values(factor).dropna(subset=[f"{metric}_mean"])
                if len(sub):
                    ax.errorbar(sub[factor], sub[f"{metric}_mean"], sub[f"{metric}_std"], marker="o", label=label)
            if ax.has_data():  # memory is only recorded on GPU
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_xlabel(factor)
            ax.set_ylabel({"train_step_s": "time / training step (s)", "train_peak_mb": "peak GPU memory (MB)"}[metric])
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.out / "scaling.png", dpi=300, bbox_inches="tight")


STAGES = {"benchmark": benchmark, "table": table, "plots": plots}
DEFAULT_STAGES = ["benchmark", "table", "plots"]
