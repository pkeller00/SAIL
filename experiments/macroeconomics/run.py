"""Spatio-temporal macroeconomic SAIL Modes (Global Macro Database).

Nodes are countries, one graph per year with a fixed node set; a single set of
modes is learned across all years so each mode keeps one meaning over time.

Stages (``sail-run experiments/macroeconomics/config.yaml --stage <name>``):

* ``train``  learn the modes
* ``eval``   global score of every mode per year (Supp. Fig. 11), mode loadings
             (Supp. Fig. 10), per-country fields and LISA labels, Fig. 8 panels
* ``maps``   world choropleths of the field and LISA clusters of one mode for selected years
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sail  # noqa: E402
from sail.config import ExperimentConfig  # noqa: E402
from sail.experiment import load_modes, train_modes, write_table  # noqa: E402
from sail.viz import LISA_COLORS, lisa_legend_handles, plot_mode_weights  # noqa: E402

META = ["ISO3", "countryname", "year", "lat", "lon"]


def build(cfg: ExperimentConfig):
    panel = pd.read_csv(cfg.path("panel"))
    features = [c for c in panel.columns if c not in META]
    graphs = []
    for year, df in panel.groupby("year", sort=True):
        g = sail.build_graph(df[["lat", "lon"]].to_numpy(), df[features].to_numpy(np.float32),
                             meta={"id": int(year), "iso": df["ISO3"].tolist(), "names": df["countryname"].tolist(),
                                   "feature_names": features}, **cfg.graph)
        g.coords = g.coords[:, [1, 0]]  # plot as (lon, lat)
        graphs.append(g)
    return graphs


def train(cfg: ExperimentConfig) -> None:
    train_modes(cfg, build(cfg))


def evaluate(cfg: ExperimentConfig) -> None:
    graphs = build(cfg)
    modes = load_modes(cfg)
    names = [f"Mode {m + 1}" for m in range(modes.W.shape[1])]
    features = graphs[0].meta["feature_names"]

    scores = pd.DataFrame(modes.scores(graphs), columns=names, index=[g.meta["id"] for g in graphs])
    scores.index.name = "year"
    write_table(scores.reset_index(), cfg.out / "global_score_by_year.csv", "global SAIL score per year")
    pd.DataFrame(modes.W.cpu().numpy(), index=features, columns=names).to_csv(cfg.out / "mode_weights.csv")

    rows = []
    for g in graphs:
        out = modes.transform(g)
        labels = sail.lisa(g, modes.model)
        for m in range(len(names)):
            rows.append(pd.DataFrame(dict(year=g.meta["id"], ISO3=g.meta["iso"], countryname=g.meta["names"],
                                          mode=m + 1, field=out.z[0][:, m].cpu().numpy(),
                                          local=out.local[0][:, m].cpu().numpy(), lisa=labels[:, m])))
    pd.concat(rows).to_csv(cfg.out / "country_scores.csv", index=False)

    fig, axs = plt.subplots(2, 4, figsize=(22, 8))
    for m, ax in enumerate(axs.flat):
        plot_mode_weights(modes.W, features, m, ax=ax, top_k=8, title=names[m])
    fig.tight_layout()
    fig.savefig(cfg.out / "mode_weights.png", dpi=300, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(9, 5))
    scores.plot(ax=ax, marker="o", ms=3)
    ax.set_ylabel("global SAIL score (Geary-style)")
    fig.savefig(cfg.out / "global_score_by_year.png", dpi=300, bbox_inches="tight")

    m = cfg.eval["figure_mode"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_mode_weights(modes.W, features, m - 1, ax=axs[0], top_k=8, title=f"Mode {m} weights")
    scores[f"Mode {m}"].plot(ax=axs[1], marker="o")
    axs[1].set_ylabel("global SAIL score")
    fig.savefig(cfg.out / "fig8.png", dpi=300, bbox_inches="tight")


def maps(cfg: ExperimentConfig) -> None:
    import geopandas as gpd

    world = gpd.read_file(cfg.path("shapefile"))
    scores = pd.read_csv(cfg.out / "country_scores.csv")
    m = cfg.eval["figure_mode"]
    for year in cfg.eval["map_years"]:
        s = scores[(scores["year"] == year) & (scores["mode"] == m)].set_index("ISO3")
        w = world.assign(field=world["iso_a3"].map(s["field"]), lisa=world["iso_a3"].map(s["lisa"]))
        fig, axs = plt.subplots(1, 2, figsize=(20, 5.5))
        w.plot(ax=axs[0], color="#f2f2f2", edgecolor="white", linewidth=0.2)
        w.dropna(subset=["field"]).plot(ax=axs[0], column="field", cmap="RdBu_r", legend=True,
                                        edgecolor="white", linewidth=0.2)
        w.plot(ax=axs[1], color=w["lisa"].map(lambda v: LISA_COLORS.get(v, "#f2f2f2") if v == v else "#f2f2f2"),
               edgecolor="white", linewidth=0.2)
        axs[1].legend(handles=lisa_legend_handles(), loc="lower left", fontsize=7)
        for ax, t in zip(axs, ["f_map", "LISA"]):
            ax.set_axis_off()
            ax.set_title(f"Mode {m}, {t}, {year}")
        fig.savefig(cfg.out / f"map_mode{m}_{year}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


STAGES = {"train": train, "eval": evaluate, "maps": maps}
DEFAULT_STAGES = ["train", "eval", "maps"]
