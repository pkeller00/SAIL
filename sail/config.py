"""Experiment configuration files.

Every experiment is described by a YAML file::

    experiment: 4i              # folder under experiments/
    seed: 0
    output_dir: outputs/4i
    data: {...}                 # experiment-specific inputs
    graph: {method: knn, k: 32, log_distances: true}
    sail:  {n_modes: 20, local_index: geary, ...}     # ModesConfig fields
    sweep:                      # optional: one-at-a-time ablation
      lambda_corr: [0, 10, 100]                       # one field, several values
      statistic:                                      # several fields changed together
        - {local_index: moran, objective: maximize}

Relative paths in ``data`` and ``output_dir`` are resolved against the
repository root.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from .modes import ModesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_KEYS = ("k", "log_distances", "method", "radius", "metric")


@dataclass
class ExperimentConfig:
    experiment: str
    seed: int = 0
    output_dir: str = "outputs"
    data: dict[str, Any] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)
    sail: ModesConfig = field(default_factory=ModesConfig)
    model: dict[str, Any] = field(default_factory=dict)   # supervised experiments
    train: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    baselines: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, list] = field(default_factory=dict)

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        p = p if p.is_absolute() else REPO_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, key: str) -> Path:
        """Resolve ``data[key]`` to an absolute path."""
        return self.path_any(self.data[key])

    @staticmethod
    def path_any(value: str | Path) -> Path:
        """Resolve a path relative to the repository root."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def ablations(self) -> Iterator[tuple[str, str, "ExperimentConfig"]]:
        """Yield ``(block, setting, config)`` for every one-at-a-time sweep setting.

        A sweep value is either a single value for the field named by the
        block, or a dict of fields to change together. Graph fields (``k``,
        ``log_distances``, ``method``, ``radius``) change the graph; all other
        fields change the SAIL config.
        """
        for block, values in self.sweep.items():
            for value in values:
                overrides = value if isinstance(value, dict) else {block: value}
                cfg = copy.deepcopy(self)
                for key, v in overrides.items():
                    if key in GRAPH_KEYS:
                        cfg.graph[key] = v
                    elif hasattr(cfg.sail, key):
                        setattr(cfg.sail, key, v)
                    else:
                        raise KeyError(f"sweep field {key!r} is neither a graph nor a SAIL setting")
                setting = ",".join(f"{k}={v}" for k, v in overrides.items())
                yield block, setting, cfg


def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Read an experiment YAML file; ``overrides`` replace top-level keys."""
    raw = yaml.safe_load(Path(path).read_text())
    raw.update(overrides)
    raw["sail"] = ModesConfig(**(raw.get("sail") or {}))
    return ExperimentConfig(**raw)
