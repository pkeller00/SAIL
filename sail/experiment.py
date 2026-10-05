"""Helpers shared by the experiment runners in ``experiments/``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

from .config import ExperimentConfig
from .graph import SpatialGraph
from .modes import SAILModes
from .utils import default_device, set_seed

log = logging.getLogger("sail")


def device_of(cfg: ExperimentConfig) -> str:
    return cfg.eval.get("device") or str(default_device())


def model_path(cfg: ExperimentConfig, name: str = "sail_modes.pt") -> Path:
    """Model trained in this run, else the shipped pre-trained model (``data.pretrained``)."""
    trained = cfg.out / name
    if trained.exists():
        return trained
    if cfg.data.get("pretrained"):
        pre = cfg.path("pretrained")
        if pre.exists():
            log.warning("using pre-trained model %s", pre)
            return pre
    raise FileNotFoundError(f"No model at {trained}; run the 'train' stage first")


def load_modes(cfg: ExperimentConfig, path: Optional[Path] = None) -> SAILModes:
    """Load the run's (or the pre-trained) model, applying ``eval.mode_chunk``."""
    modes = SAILModes.load(path or model_path(cfg), device=device_of(cfg))
    modes.model.mode_chunk = cfg.eval.get("mode_chunk")
    return modes


def train_modes(
    cfg: ExperimentConfig, graphs: Sequence[SpatialGraph], path: Optional[Path] = None
) -> SAILModes:
    """Train :class:`SAILModes` with ``cfg.sail`` (seeded with ``cfg.seed``) and save it."""
    set_seed(cfg.seed)
    modes = SAILModes(graphs[0].num_features, cfg.sail, device=device_of(cfg))
    modes.fit(graphs)
    path = path or cfg.out / "sail_modes.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    modes.save(path)
    cfg.save(path.with_suffix(".yaml"))
    log.warning("saved %s (stopped after %d epochs)", path, len(modes.history))
    return modes


def write_table(df: pd.DataFrame, path: Path, title: Optional[str] = None) -> Path:
    """Save a results table as CSV and log it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.warning("%s -> %s\n%s", title or path.stem, path, df.to_string(index=False))
    return path


def run_ablation(
    cfg: ExperimentConfig,
    build: Callable[[ExperimentConfig], Sequence[SpatialGraph]],
    score: Callable[[ExperimentConfig, SAILModes, Sequence[SpatialGraph]], dict],
) -> pd.DataFrame:
    """Train and score the default config and every one-at-a-time sweep setting.

    Parameters
    ----------
    build : cfg -> graphs
        Graph construction (called again only when graph settings change).
    score : (cfg, modes, graphs) -> dict
        Summary metrics of one trained model (one row of the ablation table).
    """
    runs = [("default", "default", cfg)] + list(cfg.ablations())
    graph_cache: dict[tuple, Sequence[SpatialGraph]] = {}
    rows = []
    for block, setting, run_cfg in runs:
        key = tuple(sorted(run_cfg.graph.items()))
        if key not in graph_cache:
            graph_cache[key] = build(run_cfg)
        graphs = graph_cache[key]
        safe = setting.replace("=", "-").replace(",", "_")
        path = cfg.out / "ablation" / f"{safe}.pt"
        modes = load_modes(cfg, path) if path.exists() else train_modes(run_cfg, graphs, path)
        modes.model.mode_chunk = cfg.eval.get("mode_chunk")
        rows.append(dict(block=block, setting=setting, **score(run_cfg, modes, graphs)))
        write_table(pd.DataFrame(rows), cfg.out / "ablation" / "ablation_summary.csv", "ablation (partial)")
    return pd.DataFrame(rows)
