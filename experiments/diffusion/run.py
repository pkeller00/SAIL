"""Diffusion of two gases: SAIL captures the loss of spatial autocorrelation.

Two equal-mass hard-disc gases (red = +1 on the left, blue = -1 on the right)
start separated by a partition. Particles move ballistically and undergo
perfectly elastic collisions with each other, the box and the partition. In
the ``closed`` control the partition stays; in ``hole`` a small opening is
made at t = 0; in ``open`` the partition is removed at t = 0.

Every snapshot is a spatial graph (particles = nodes, kNN edges) whose node
features are ``[species, speed, noise_1..noise_n]``. For an interdiffusing
binary gas the species field ``c(x, t)`` obeys the diffusion equation, and a
Moran-type statistic measures ``<c^2>``, so for the step initial condition in
a box of width ``L`` with reflecting walls

    I(t) ~ <c^2>(t) = sum_{n odd} 8 / (n pi)^2 exp(-2 n^2 pi^2 D t / L^2),

i.e. SAIL should decay at twice the rate of the species' first Fourier mode.

Stages (``sail-run experiments/diffusion/config.yaml --stage <name>``):

* ``simulate``   hard-disc molecular dynamics for every scenario and seed
* ``train``      one learned SAIL Moran mode on the separate training run
* ``score``      per snapshot: SAIL with fixed components (= Moran's I, Geary's C),
                 the learned SAIL mode, classical Moran's I, Fourier mode, <c^2>
* ``analyse``    decay times and diffusion coefficients (MSD, Fourier, <c^2>, SAIL)
* ``figures``    600-dpi figures
* ``video``      animation of the diffusion with LISA maps and the SAIL time series
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import EllipseCollection  # noqa: E402
from scipy.optimize import curve_fit, minimize_scalar  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import sail  # noqa: E402
from sail.config import ExperimentConfig  # noqa: E402
from sail.experiment import device_of, load_modes, train_modes, write_table  # noqa: E402
from sail.viz import plot_mode_weights  # noqa: E402
from sail.viz.style import LISA_COLORS, lisa_legend_handles  # noqa: E402

log = logging.getLogger("sail")

RED, BLUE = "#ef4444", "#3b82f6"
SCENARIO_COLORS = {"closed": "#6b7280", "hole": "#d97706", "open": "#7c3aed"}
SCENARIO_NAMES = {"closed": "Partition closed", "hole": "Small hole", "open": "Partition removed"}
FIXED_TEMP = -30.0  # softplus(-30) ~ 1e-13: uniform weights over the k neighbours (classical kNN statistics)


# --------------------------------------------------------------------------- #
# hard-disc gas
# --------------------------------------------------------------------------- #
def _jittered_grid(rng, n, x0, x1, y0, y1, r):
    """``n`` non-overlapping disc centres in a rectangle (random cells of a jittered grid)."""
    w, h = x1 - x0, y1 - y0
    nx = int(np.ceil(np.sqrt(n * w / h)))
    ny = int(np.ceil(n / nx))
    cw, ch = w / nx, h / ny
    if min(cw, ch) < 2 * r + 1:
        raise ValueError("too many particles for the box")
    cells = rng.choice(nx * ny, size=n, replace=False)
    cx, cy = cells % nx, cells // nx
    jitter = max(0.0, min(cw, ch) / 2 - r - 0.5)
    return np.column_stack([x0 + (cx + 0.5) * cw + rng.uniform(-jitter, jitter, n),
                            y0 + (cy + 0.5) * ch + rng.uniform(-jitter, jitter, n)])


class Gas:
    """Equal-mass hard discs in a box with an optional vertical partition at ``x = W / 2``.

    ``wall`` is ``"full"``, ``"hole"`` or ``"none"``. The partition is purely
    geometric: it never looks at the species.
    """

    def __init__(self, d: dict, wall: str, rng: np.random.Generator):
        self.W, self.H = map(float, d["box"])
        self.R, self.mid, self.wall = float(d["radius"]), float(d["box"][0]) / 2, wall
        self.gap = (self.H / 2 - d["hole_size"] / 2, self.H / 2 + d["hole_size"] / 2)
        n, m = d["n_particles"], d["margin"]
        h = n // 2
        left = _jittered_grid(rng, h, m, self.mid - m, m, self.H - m, self.R)
        right = _jittered_grid(rng, n - h, self.mid + m, self.W - m, m, self.H - m, self.R)
        self.pos = np.vstack([left, right])
        self.species = np.r_[np.ones(h), -np.ones(n - h)].astype(np.int8)
        # 2-D Maxwell-Boltzmann velocities with rms speed `speed` (left side scaled by sqrt(T_L / T_R))
        self.vel = rng.normal(scale=d["speed"] / np.sqrt(2), size=(n, 2))
        self.vel[:h] *= np.sqrt(d["temperature_ratio"])

    def _box(self):
        p, v, R = self.pos, self.vel, self.R
        for ax, hi in ((0, self.W), (1, self.H)):
            lo_hit, hi_hit = p[:, ax] < R, p[:, ax] > hi - R
            p[lo_hit, ax], v[lo_hit, ax] = R, np.abs(v[lo_hit, ax])
            p[hi_hit, ax], v[hi_hit, ax] = hi - R, -np.abs(v[hi_hit, ax])

    def _partition(self, old_x):
        """Specular reflection off the partition, using the side each particle came from."""
        if self.wall == "none":
            return
        p, v, R, mid = self.pos, self.vel, self.R, self.mid
        blocked = np.ones(len(p), dtype=bool)
        if self.wall == "hole":
            blocked = ~((p[:, 1] - R > self.gap[0]) & (p[:, 1] + R < self.gap[1]))
        left = (old_x < mid) | ((old_x == mid) & (v[:, 0] >= 0))
        m = blocked & left & (p[:, 0] > mid - R)
        p[m, 0], v[m, 0] = mid - R, -np.abs(v[m, 0])
        m = blocked & ~left & (p[:, 0] < mid + R)
        p[m, 0], v[m, 0] = mid + R, np.abs(v[m, 0])

    def _collide(self):
        """Equal-mass elastic disc collisions: exchange the normal velocity components of approaching pairs."""
        p, v, dmin = self.pos, self.vel, 2 * self.R
        for i, j in cKDTree(p).query_pairs(dmin, output_type="ndarray"):
            dx = p[j] - p[i]
            dist = np.hypot(*dx)
            nrm = dx / dist if dist > 1e-9 else np.array([1.0, 0.0])
            push = (dmin - dist) / 2 + 0.01  # positional correction against interpenetration
            p[i] -= nrm * push
            p[j] += nrm * push
            rel = (v[j] - v[i]) @ nrm
            if rel < 0:
                v[i] += rel * nrm
                v[j] -= rel * nrm

    def step(self, dt: float):
        vmax = np.sqrt((self.vel**2).sum(1).max())
        n_sub = max(1, int(np.ceil(vmax * dt / (0.75 * self.R))))
        h = dt / n_sub
        for _ in range(n_sub):
            old_x = self.pos[:, 0].copy()
            self.pos += self.vel * h
            self._box()
            self._partition(old_x)
            before = self.pos[:, 0].copy()
            self._collide()
            self._box()
            self._partition(before)


def _traj_path(cfg, scenario, seed):
    return cfg.out / "trajectories" / f"{scenario}_seed{seed}.npz"


def _load(cfg, scenario, seed):
    return dict(np.load(_traj_path(cfg, scenario, seed)))


def simulate(cfg: ExperimentConfig) -> None:
    d = cfg.data
    runs = [(s, seed) for s in d["scenarios"] for seed in d["seeds"]] + [("open", d["train_seed"])]
    n_steps = int(round(d["duration"] / d["dt"]))
    every = int(round(d["snapshot_every"] / d["dt"]))
    for scenario, seed in runs:
        path = _traj_path(cfg, scenario, seed)
        if path.exists():
            continue
        wall = {"closed": "full", "hole": "hole", "open": "none"}[scenario]
        gas = Gas(d, wall, np.random.default_rng(seed))
        pos, speed = [], []
        for step in range(n_steps + 1):
            if step % every == 0:
                pos.append(gas.pos.astype(np.float32))
                speed.append(np.hypot(*gas.vel.T).astype(np.float32))
            if step < n_steps:
                gas.step(d["dt"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, pos=np.stack(pos), speed=np.stack(speed), species=gas.species,
                            times=np.arange(len(pos)) * every * d["dt"])
        log.warning("simulated %s seed %d (%d snapshots)", scenario, seed, len(pos))


# --------------------------------------------------------------------------- #
# graphs and SAIL
# --------------------------------------------------------------------------- #
def _feature_names(cfg):
    return ["species", "speed"] + [f"noise {i + 1}" for i in range(cfg.data["n_noise"])]


def _graph(cfg, traj, t, seed):
    """kNN graph of snapshot ``t`` with standardised features [species, speed, noise...]."""
    n = len(traj["species"])
    noise = np.random.default_rng([seed, t]).standard_normal((n, cfg.data["n_noise"]))
    x = np.column_stack([traj["species"], traj["speed"][t], noise]).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    return sail.build_graph(traj["pos"][t], x.astype(np.float32), **cfg.graph)


def train(cfg: ExperimentConfig) -> None:
    seed = cfg.data["train_seed"]
    traj = _load(cfg, "open", seed)
    graphs = [_graph(cfg, traj, t, seed) for t in range(0, len(traj["times"]), cfg.eval["train_every"])]
    modes = train_modes(cfg, graphs)
    w = modes.W[:, 0].cpu().numpy()
    log.warning("learned mode loadings: %s | temperature %.3f",
                ", ".join(f"{n}={v:+.3f}" for n, v in zip(_feature_names(cfg), w)), float(modes.model.temperature))


def _classical_moran(xy, s, k):
    """Moran's I with row-standardised binary kNN weights (independent of SAIL)."""
    _, nb = cKDTree(xy).query(xy, k + 1)
    z = s - s.mean()
    return float((z * z[nb[:, 1:]].mean(1)).sum() / (z**2).sum())


def _fourier_mode(xy, s, d):
    """First cosine coefficient of the species profile on [R, W - R]."""
    L = d["box"][0] - 2 * d["radius"]
    return float(2 * np.mean(s * np.cos(np.pi * (xy[:, 0] - d["radius"]) / L)))


def _profile_c2(xy, s, d, n_bins):
    """Unbiased <c^2> of the species concentration profile along x."""
    edges = np.linspace(d["radius"], d["box"][0] - d["radius"], n_bins + 1)
    b = np.clip(np.digitize(xy[:, 0], edges) - 1, 0, n_bins - 1)
    num = np.bincount(b, minlength=n_bins).astype(float)
    tot = np.bincount(b, weights=s.astype(float), minlength=n_bins)
    ok = num > 1
    m = tot[ok] / num[ok]
    c2 = (num[ok] * m**2 - 1) / (num[ok] - 1)  # E[mean^2] = c^2 + (1 - c^2) / n for +-1 values
    return float((num[ok] * c2).sum() / num[ok].sum())


def score(cfg: ExperimentConfig) -> None:
    d, k = cfg.data, cfg.graph.get("k", 8)
    dev = device_of(cfg)
    moran_fixed = sail.SAIL(local_index="moran", learnable_temperature=False, init_temp=FIXED_TEMP).to(dev)
    geary_fixed = sail.SAIL(local_index="geary", standardise="sample", learnable_temperature=False,
                            init_temp=FIXED_TEMP).to(dev)
    modes = load_modes(cfg)
    rows = []
    for scenario in d["scenarios"]:
        for seed in d["seeds"]:
            traj = _load(cfg, scenario, seed)
            s = traj["species"].astype(float)
            graphs = [_graph(cfg, traj, t, seed) for t in range(len(traj["times"]))]
            learned = modes.scores(graphs)[:, 0]
            with torch.no_grad():
                for t, g in enumerate(graphs):
                    xy = traj["pos"][t]
                    rows.append(dict(
                        scenario=scenario, seed=seed, t=float(traj["times"][t]),
                        moran_sail=float(moran_fixed(g).global_scores[0, 0]),
                        moran_classical=_classical_moran(xy, s, k),
                        geary_sail=float(geary_fixed(g).global_scores[0, 0]),
                        sail_learned=float(learned[t]),
                        fourier=_fourier_mode(xy, s, d),
                        profile_c2=_profile_c2(xy, s, d, cfg.eval["profile_bins"]),
                    ))
            log.warning("scored %s seed %d", scenario, seed)
    df = pd.DataFrame(rows)
    df["fourier_norm"] = df["fourier"] / df.groupby(["scenario", "seed"])["fourier"].transform("first")
    gap = (df["moran_sail"] - df["moran_classical"]).abs().max()
    log.warning("max |SAIL (fixed) - classical Moran's I| over %d snapshots: %.2e", len(df), gap)
    df.to_csv(cfg.out / "timeseries.csv", index=False)


# --------------------------------------------------------------------------- #
# diffusion theory and fits
# --------------------------------------------------------------------------- #
def _width(d):
    return d["box"][0] - 2 * d["radius"]


def c2_theory(t, D, L, n_terms=200):
    """<c^2>(t) for a +-1 step in [0, L] with reflecting walls."""
    n = np.arange(1, 2 * n_terms, 2)
    return (8 / (np.pi * n) ** 2 * np.exp(-2 * np.outer(np.atleast_1d(t), n**2) * np.pi**2 * D / L**2)).sum(1)


def fourier_theory(t, D, L):
    return np.exp(-np.pi**2 * D * np.asarray(t) / L**2)


def _fit_D(t, y, model, floor):
    """Least-squares D of ``model(t, D)`` on points above the noise floor."""
    m = y > floor
    res = minimize_scalar(lambda logD: ((model(t[m], np.exp(logD)) - y[m]) ** 2).sum(),
                          bounds=(np.log(1.0), np.log(1e6)), method="bounded")
    return float(np.exp(res.x))


def _fit_tau(t, y):
    try:
        (a, tau, c), _ = curve_fit(lambda t, a, tau, c: a * np.exp(-t / tau) + c, t, y,
                                   p0=(max(y[0] - y[-1], 1e-3), t[-1] / 4, y[-1]), bounds=([0, 1e-3, -1], [2, 1e5, 2]),
                                   maxfev=20000)
        return float(tau)
    except (RuntimeError, ValueError):
        return float("nan")


def _msd_D(cfg):
    """Self-diffusion coefficient from the mean squared displacement of the 'open' runs."""
    d = cfg.data
    lo, hi = cfg.eval["msd_window"]
    lags = np.arange(1, int(round(hi / d["snapshot_every"])) + 1)
    msd = np.zeros(len(lags))
    for seed in d["seeds"]:
        pos = _load(cfg, "open", seed)["pos"].astype(np.float64)
        msd += [((pos[lag:] - pos[:-lag]) ** 2).sum(-1).mean() for lag in lags]
    msd /= len(d["seeds"])
    tl = lags * d["snapshot_every"]
    m = tl >= lo
    return float(np.polyfit(tl[m], msd[m], 1)[0] / 4)


MEASURES = {"moran_sail": "SAIL, fixed (Moran's I)", "sail_learned": "SAIL, learned mode",
            "profile_c2": "Profile <c^2>", "fourier_norm": "First Fourier mode", "geary_sail": "SAIL, fixed (Geary's C)"}


def _curve(df, scenario, col):
    """Mean and std over seeds of one measure, indexed by time."""
    return df[df["scenario"] == scenario].groupby("t")[col].agg(["mean", "std"])


def analyse(cfg: ExperimentConfig) -> None:
    d = cfg.data
    L = _width(d)
    df = pd.read_csv(cfg.out / "timeseries.csv")
    rows = [dict(scenario="open", measure="Mean squared displacement", tau=np.nan, D=_msd_D(cfg))]
    for scenario, g in df.groupby("scenario"):
        mean = g.groupby("t").mean(numeric_only=True)
        t = mean.index.to_numpy()
        for col, name in MEASURES.items():
            y = mean[col].to_numpy()
            tau = _fit_tau(t, 1 - y if col == "geary_sail" else y)
            D = np.nan
            if scenario == "open" and col == "fourier_norm":
                D = _fit_D(t, y, lambda t, D: fourier_theory(t, D, L), 0.1)
            elif scenario == "open" and col in ("moran_sail", "sail_learned", "profile_c2"):
                D = _fit_D(t, y, lambda t, D: c2_theory(t, D, L), 0.05)
            rows.append(dict(scenario=scenario, measure=name, tau=tau, D=D))
    write_table(pd.DataFrame(rows), cfg.out / "decay_fits.csv", "decay times and diffusion coefficients")
    ok = df[["moran_sail", "profile_c2"]].dropna()
    rho = spearmanr(ok["moran_sail"], ok["profile_c2"])[0]
    write_table(pd.DataFrame([dict(pair="SAIL (fixed) vs profile <c^2>", spearman=rho),
                              dict(pair="SAIL (learned) vs profile <c^2>",
                                   spearman=spearmanr(df["sail_learned"], df["profile_c2"])[0])]),
                cfg.out / "correlations.csv", "rank correlation with the diffusion profile")


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def _setup_box(ax, d, scenario, title=None):
    W, H = d["box"]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#334155")
        sp.set_linewidth(1.5)
    mid = W / 2
    if scenario == "closed":
        ax.plot([mid, mid], [0, H], color="#111827", lw=2.5)
    elif scenario == "hole":
        g0, g1 = H / 2 - d["hole_size"] / 2, H / 2 + d["hole_size"] / 2
        ax.plot([mid, mid], [0, g0], color="#111827", lw=2.5)
        ax.plot([mid, mid], [g1, H], color="#111827", lw=2.5)
    if title:
        ax.set_title(title, fontsize=9)


def _discs(ax, pos, colors, d):
    coll = EllipseCollection(widths=2 * d["radius"], heights=2 * d["radius"], angles=0, units="xy",
                             offsets=pos, offset_transform=ax.transData, facecolors=colors, linewidths=0)
    ax.add_collection(coll)
    return coll


def _species_colors(species):
    return np.where(species > 0, RED, BLUE)


def _lisa_colors(labels):
    return np.array([LISA_COLORS[int(i)] for i in labels])


def _lisa(cfg, modes, g):
    gen = torch.Generator().manual_seed(cfg.seed)
    return sail.lisa(g, modes.model, modes=0, permutations=cfg.eval["lisa_permutations"], generator=gen)


def _save(fig, cfg, name):
    for ext in ("png", "pdf"):
        fig.savefig(cfg.out / f"{name}.{ext}", dpi=cfg.eval["dpi"], bbox_inches="tight")
    plt.close(fig)
    log.warning("saved %s", cfg.out / f"{name}.png")


def _band(ax, x, mean, std, color, label, ls="-"):
    ax.plot(x, mean, color=color, lw=1.6, ls=ls, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(cfg: ExperimentConfig) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    d = cfg.data
    L = _width(d)
    df = pd.read_csv(cfg.out / "timeseries.csv")
    fits = pd.read_csv(cfg.out / "decay_fits.csv")
    D_fourier = fits.query("scenario == 'open' and measure == 'First Fourier mode'")["D"].item()
    modes = load_modes(cfg)

    # SAIL scores over time, all scenarios
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    panels = [("moran_sail", "Moran's I (SAIL, fixed components)"), ("sail_learned", "Learned SAIL mode score"),
              ("geary_sail", "Geary's C (SAIL, fixed components)")]
    for ax, (col, ylabel) in zip(axs, panels):
        for scenario in d["scenarios"]:
            c = _curve(df, scenario, col)
            _band(ax, c.index, c["mean"], c["std"], SCENARIO_COLORS[scenario], SCENARIO_NAMES[scenario])
        if col != "geary_sail":
            t = np.linspace(0, d["duration"], 400)
            ax.plot(t, c2_theory(t, D_fourier, L), color="black", ls="--", lw=1,
                    label=r"Diffusion theory $\langle c^2\rangle$")
        ax.axhline(1.0 if col == "geary_sail" else 0.0, color="#94a3b8", lw=0.8, zorder=0)
        ax.set_xlabel("Time since partition opened (s)")
        ax.set_ylabel(ylabel)
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, cfg, "fig_sail_vs_time")

    # quantitative link with diffusion
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axs[0]
    t = np.linspace(0, d["duration"], 400)
    for col, color, label in [("fourier_norm", "#0f766e", "First Fourier mode $a_1/a_1(0)$"),
                              ("moran_sail", SCENARIO_COLORS["open"], "Moran's I (SAIL, fixed)"),
                              ("sail_learned", "#db2777", "Learned SAIL mode")]:
        c = _curve(df, "open", col)
        ax.plot(c.index, c["mean"].clip(lower=1e-3), color=color, lw=1.6, label=label)
    ax.plot(t, fourier_theory(t, D_fourier, L), color="#0f766e", ls="--", lw=1)
    ax.plot(t, c2_theory(t, D_fourier, L), color="black", ls="--", lw=1, label="Theory (D from Fourier mode)")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1.2)
    ax.set_xlabel("Time since partition removed (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("SAIL decays at twice the Fourier rate", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axs[1]
    for scenario in d["scenarios"]:
        s = df[df["scenario"] == scenario]
        ax.scatter(s["profile_c2"], s["moran_sail"], s=3, alpha=0.35, color=SCENARIO_COLORS[scenario],
                   label=SCENARIO_NAMES[scenario], rasterized=True)
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1)
    rho = spearmanr(df["profile_c2"], df["moran_sail"])[0]
    ax.set_xlabel(r"Species profile $\langle c^2\rangle$")
    ax.set_ylabel("Moran's I (SAIL, fixed)")
    ax.set_title(f"Spearman rho = {rho:.3f}", fontsize=9)
    ax.legend(frameon=False, fontsize=7, markerscale=3)

    ax = axs[2]
    est = fits.query("scenario == 'open'").dropna(subset=["D"])
    colors = {"Mean squared displacement": "#64748b", "First Fourier mode": "#0f766e", "Profile <c^2>": "#0284c7",
              MEASURES["moran_sail"]: SCENARIO_COLORS["open"], MEASURES["sail_learned"]: "#db2777"}
    ax.barh(range(len(est)), est["D"], color=[colors[m] for m in est["measure"]])
    ax.set_yticks(range(len(est)))
    ax.set_yticklabels(est["measure"], fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(est["D"]):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)
    ax.set_xlabel(r"Diffusion coefficient D (px$^2$/s)")
    ax.set_title("Estimates of D (partition removed)", fontsize=9)
    fig.tight_layout()
    _save(fig, cfg, "fig_diffusion_theory")

    # learned mode loadings
    fig, ax = plt.subplots(figsize=(4, 3))
    names = _feature_names(cfg)
    plot_mode_weights(modes.W, names, 0, ax=ax, top_k=len(names),
                      title=f"Learned SAIL mode (tau = {float(modes.model.temperature):.2f})")
    ax.set_ylabel("Loading")
    fig.tight_layout()
    _save(fig, cfg, "fig_mode_weights")

    # snapshots: species and LISA of the learned mode
    seed = cfg.eval["video"]["seed"]
    times = cfg.eval["snapshot_times"]
    for scenario in d["scenarios"]:
        traj = _load(cfg, scenario, seed)
        idx = [int(np.argmin(np.abs(traj["times"] - t))) for t in times]
        fig, axs = plt.subplots(2, len(idx), figsize=(2.6 * len(idx), 3.4))
        run = df[(df["scenario"] == scenario) & (df["seed"] == seed)].reset_index(drop=True)
        for col, t in enumerate(idx):
            g = _graph(cfg, traj, t, seed)
            _setup_box(axs[0, col], d, scenario,
                       f"t = {traj['times'][t]:.0f} s, I = {run.loc[t, 'moran_sail']:.2f}")
            _discs(axs[0, col], traj["pos"][t], _species_colors(traj["species"]), d)
            _setup_box(axs[1, col], d, scenario)
            _discs(axs[1, col], traj["pos"][t], _lisa_colors(_lisa(cfg, modes, g)), d)
        axs[0, 0].set_ylabel("Gas")
        axs[1, 0].set_ylabel("LISA (learned mode)")
        fig.legend(handles=lisa_legend_handles(), loc="lower center", ncol=5, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(SCENARIO_NAMES[scenario], fontsize=10)
        fig.tight_layout()
        _save(fig, cfg, f"fig_snapshots_{scenario}")


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def video(cfg: ExperimentConfig) -> None:
    d, v = cfg.data, cfg.eval["video"]
    L = _width(d)
    seed = v["seed"]
    modes = load_modes(cfg)
    df = pd.read_csv(cfg.out / "timeseries.csv")
    fits = pd.read_csv(cfg.out / "decay_fits.csv")
    D_fourier = fits.query("scenario == 'open' and measure == 'First Fourier mode'")["D"].item()
    use_ffmpeg = animation.writers.is_available("ffmpeg")
    if not use_ffmpeg:
        log.warning("ffmpeg not found: writing GIFs instead of MP4")

    for scenario in v["scenarios"]:
        traj = _load(cfg, scenario, seed)
        run = df[(df["scenario"] == scenario) & (df["seed"] == seed)].reset_index(drop=True)
        times = traj["times"]
        lisa_cols = [_lisa_colors(_lisa(cfg, modes, _graph(cfg, traj, t, seed))) for t in range(len(times))]

        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
        ax_gas, ax_lisa, ax_ts = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        _setup_box(ax_gas, d, scenario, "Gas identity (red = +1, blue = -1)")
        _setup_box(ax_lisa, d, scenario, "LISA clusters of the learned SAIL mode")
        gas = _discs(ax_gas, traj["pos"][0], _species_colors(traj["species"]), d)
        lisa_discs = _discs(ax_lisa, traj["pos"][0], lisa_cols[0], d)
        ax_lisa.legend(handles=lisa_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5,
                       frameon=False, fontsize=7)

        ax_ts.set_xlim(0, times[-1])
        ax_ts.set_ylim(-0.1, 1.05)
        ax_ts.axhline(0, color="#94a3b8", lw=0.8)
        ax_ts.set_xlabel("Time since partition opened (s)")
        ax_ts.set_ylabel("Spatial autocorrelation")
        for side in ("top", "right"):
            ax_ts.spines[side].set_visible(False)
        if scenario == "open":
            ax_ts.plot(times, c2_theory(times, D_fourier, L), color="black", ls="--", lw=1,
                       label=r"Diffusion theory $\langle c^2\rangle$")
        (l_moran,) = ax_ts.plot([], [], color=SCENARIO_COLORS["open"], lw=2, label="Moran's I (SAIL, fixed)")
        (l_learn,) = ax_ts.plot([], [], color="#db2777", lw=2, label="Learned SAIL mode")
        ax_ts.legend(frameon=False, loc="upper right", fontsize=9)
        title = fig.suptitle("", fontsize=12)
        fig.tight_layout()

        def update(i):
            gas.set_offsets(traj["pos"][i])
            lisa_discs.set_offsets(traj["pos"][i])
            lisa_discs.set_facecolors(lisa_cols[i])
            l_moran.set_data(times[: i + 1], run["moran_sail"][: i + 1])
            l_learn.set_data(times[: i + 1], run["sail_learned"][: i + 1])
            title.set_text(f"{SCENARIO_NAMES[scenario]}  |  t = {times[i]:5.1f} s  |  "
                           f"Moran's I = {run['moran_sail'][i]:.3f}  |  learned SAIL = {run['sail_learned'][i]:.3f}")
            return gas, lisa_discs, l_moran, l_learn, title

        anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
        if use_ffmpeg:
            path = cfg.out / f"diffusion_{scenario}.mp4"
            writer = animation.FFMpegWriter(fps=v["fps"], codec="libx264", bitrate=4000,
                                            extra_args=["-pix_fmt", "yuv420p"])
        else:
            path = cfg.out / f"diffusion_{scenario}.gif"
            writer = animation.PillowWriter(fps=v["fps"])
        anim.save(path, writer=writer, dpi=v["dpi"])
        plt.close(fig)
        log.warning("saved %s", path)


STAGES = {"simulate": simulate, "train": train, "score": score, "analyse": analyse, "figures": figures,
          "video": video}
DEFAULT_STAGES = list(STAGES)
