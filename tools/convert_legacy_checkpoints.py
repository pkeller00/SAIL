"""Load checkpoints written by the pre-release SAIL code.

Pre-release ``SAILModes.save`` stored only the ``state_dict`` and a partial
config, so the remaining hyper-parameters must be given explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch

from sail.modes import ModesConfig, SAILModes

_RENAMES = {"f_theta.": "f_affinity."}
_AFFINITY_NAMES = {None: None, "SharedFMapTheta": "shared_map", "QKTheta": "query_key"}


def load_legacy_modes(
    path: Union[str, Path], input_dim: int, device: str = "cpu", **config: Any
) -> SAILModes:
    """Convert a pre-release ``SAILModes`` checkpoint to a :class:`SAILModes`.

    Parameters
    ----------
    path : str or Path
        Legacy ``.pth`` file.
    input_dim : int
        Node feature dimension.
    **config
        :class:`ModesConfig` fields the legacy file does not record
        (``local_index``, ``f_affinity``, ``reduce``, ``activation``, ...).
        ``n_modes``, ``objective`` and the regulariser weights are read from
        the file.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    old = ckpt["config"]
    cfg = dict(
        n_modes=old["k"],
        objective=old["objective"],
        epochs=old["epochs"],
        batch_size=old["batch_size"],
        lambda_corr=old["lambda_corr"] if old.get("decorrelate") else 0.0,
        lambda_sparse=old["lambda_sparse"] if old.get("sparse") else 0.0,
    )
    if "f_affinity" in config:
        config["f_affinity"] = _AFFINITY_NAMES.get(config["f_affinity"], config["f_affinity"])
    cfg.update(config)

    modes = SAILModes(input_dim, ModesConfig(**cfg), device=device, _exported=True)
    state = {}
    for key, value in ckpt["model_state"].items():
        for old_prefix, new_prefix in _RENAMES.items():
            if key.startswith(old_prefix):
                key = new_prefix + key[len(old_prefix):]
        state[key] = value
    modes.model.load_state_dict(state)
    modes.mean_scores = ckpt.get("best_mean_I")
    modes.best_loss = ckpt.get("best_score", float("nan"))
    return modes
