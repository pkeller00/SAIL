"""Unsupervised discovery of SAIL Modes (Section III-C, Supplementary Algorithm 2).

A SAIL Mode is a projection ``w_m`` of the node features whose scalar field
``x w_m`` extremises a global SAC statistic. :class:`SAILModes` learns ``h``
modes jointly by optimising::

    L = s * mean_b mean_m I_global[b, m]
        + lambda_corr   * decorrelation(z)      (modes should differ)
        + lambda_sparse * mean |W|              (interpretable loadings)

with ``s = -1`` to maximise (e.g. Moran's I) and ``s = +1`` to minimise (e.g.
Geary's C).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .feature_affinity import QueryKeyAffinity, SharedMapAffinity
from .feature_maps import NormalisedLinear, StiefelLinear, orient_columns
from .graph import SpatialGraph
from .model import SAIL, SAILOutput

log = logging.getLogger(__name__)

FORMAT_VERSION = "sail-modes/1"
ACTIVATIONS = {"identity": nn.Identity, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}


@dataclass
class ModesConfig:
    """Hyper-parameters of :class:`SAILModes` (paper defaults).

    Attributes
    ----------
    n_modes : int
        Number of modes ``h``.
    local_index : str
        ``"geary"``, ``"moran"`` or ``"gstar"``.
    objective : {"minimize", "maximize"}
        Direction of optimisation of the global score. Geary's C is minimised
        (low C = neighbours similar), Moran's I maximised.
    stiefel : bool
        Constrain ``W`` to the Stiefel manifold during training.
    identity_map : bool
        Use a fixed identity projection (classical per-feature statistics).
    f_affinity : {None, "shared_map", "query_key"}
        Learned feature affinity (paper default: none).
    reduce : {"mean", "sum", "max"}
        Global reduction.
    activation : {"identity", "relu", "sigmoid"}
        Activation on the local index.
    learnable_temperature, init_temp
        Spatial temperature settings.
    standardise : {"population", "sample", "none"}
        Standardisation of the projected fields.
    self_loops : bool
        Allow self-affinity.
    lambda_corr, lambda_sparse : float
        Regulariser weights (0 disables the term).
    lr, epochs, batch_size
        Adam learning rate, maximum number of epochs, graphs per step.
    scheduler : {"step", "plateau", None}
        LR schedule; ``"step"`` multiplies the LR by ``scheduler_gamma`` every
        ``scheduler_step_size`` epochs (used for all paper runs).
    patience, min_delta
        Early stopping: stop after ``patience`` epochs whose summed loss did not
        improve on the best by more than ``min_delta``.
    """

    n_modes: int = 20
    local_index: str = "geary"
    objective: Literal["minimize", "maximize"] = "minimize"
    stiefel: bool = True
    identity_map: bool = False
    f_affinity: Optional[str] = None
    reduce: str = "mean"
    activation: str = "identity"
    learnable_temperature: bool = True
    init_temp: float = 1.0
    standardise: str = "population"
    self_loops: bool = False
    lambda_corr: float = 1.0
    lambda_sparse: float = 1.0
    lr: float = 0.05
    epochs: int = 1000
    batch_size: int = 1
    scheduler: Optional[str] = "step"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    scheduler_patience: int = 5
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class EpochLog:
    epoch: int
    loss: float
    objective: float
    corr_penalty: float
    sparse_penalty: float
    lr: float
    temperature: float
    mean_scores: list[float] = field(default_factory=list)


class SAILModes:
    """Learn SAIL Modes without labels.

    Parameters
    ----------
    input_dim : int
        Node feature dimension ``d``.
    config : ModesConfig, optional
        Hyper-parameters; keyword arguments override individual fields.
    device : str or torch.device
        Device to train on.

    Examples
    --------
    >>> modes = SAILModes(input_dim=43, n_modes=20, local_index="geary").fit(graphs)
    >>> modes.W.shape            # (43, 20) mode loadings
    >>> modes.scores(graphs)     # (n_graphs, 20) global SAIL scores
    >>> modes.save("4i_modes.pt")

    Notes
    -----
    The Stiefel constraint is only used while training. After :meth:`fit`
    the best projection is sorted by score, sign-oriented and exported to a
    plain :class:`~sail.feature_maps.NormalisedLinear` map (a single matmul),
    which is what :meth:`save` stores and :meth:`load` restores.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ModesConfig] = None,
        device: Union[str, torch.device] = "cpu",
        _exported: bool = False,
        **overrides: Any,
    ) -> None:
        cfg = config or ModesConfig()
        if overrides:
            cfg = ModesConfig(**{**asdict(cfg), **overrides})
        if cfg.objective not in ("minimize", "maximize"):
            raise ValueError("objective must be 'minimize' or 'maximize'")
        self.config = cfg
        self.input_dim = input_dim
        self.device = torch.device(device)
        self.trained_with_stiefel = cfg.stiefel and not cfg.identity_map

        # `_exported` rebuilds the post-training architecture (plain linear map) for `load`
        use_stiefel = self.trained_with_stiefel and not _exported
        self.model = self._build_model(stiefel=use_stiefel).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.scheduler = self._build_scheduler()

        self.best_loss = float("inf")
        self.mean_scores: Optional[Tensor] = None  # (h,) mean global score per mode
        self.history: list[EpochLog] = []

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_model(self, stiefel: bool) -> SAIL:
        cfg = self.config
        if cfg.identity_map:
            f_map = NormalisedLinear(self.input_dim, cfg.n_modes, trainable=False)
        elif stiefel:
            f_map = StiefelLinear(self.input_dim, cfg.n_modes)
        else:
            f_map = NormalisedLinear(self.input_dim, cfg.n_modes)

        if cfg.f_affinity is None:
            f_affinity = None
        elif cfg.f_affinity == "shared_map":
            f_affinity = SharedMapAffinity(f_map)
        elif cfg.f_affinity == "query_key":
            f_affinity = QueryKeyAffinity(self.input_dim)
        else:
            raise ValueError(f"Unknown f_affinity {cfg.f_affinity!r}")

        return SAIL(
            f_map=f_map,
            f_affinity=f_affinity,
            local_index=cfg.local_index,
            reduce=cfg.reduce,
            f_act=ACTIVATIONS[cfg.activation.lower()](),
            init_temp=cfg.init_temp,
            learnable_temperature=cfg.learnable_temperature,
            standardise=cfg.standardise,
            self_loops=cfg.self_loops,
        )

    def _build_scheduler(self):
        cfg = self.config
        if cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma
            )
        if cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=cfg.scheduler_gamma, patience=cfg.scheduler_patience
            )
        if cfg.scheduler is None:
            return None
        raise ValueError(f"Unknown scheduler {cfg.scheduler!r}")

    # ------------------------------------------------------------------ #
    # losses
    # ------------------------------------------------------------------ #
    @staticmethod
    def decorrelation_penalty(z: Tensor) -> Tensor:
        """Mean squared off-diagonal entry of the mode correlation ``z^T z / N``."""
        c = z.T @ z / z.shape[0]
        off_diag = ~torch.eye(c.shape[0], dtype=torch.bool, device=z.device)
        return (c[off_diag] ** 2).mean()

    def sparsity_penalty(self) -> Tensor:
        """Mean absolute loading ``mean |W|`` (L1 on mode weights)."""
        W = getattr(self.model.f_map, "W", None)
        return W.abs().mean() if W is not None else torch.zeros((), device=self.device)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def step(self, batch: Sequence[SpatialGraph], record: bool = True) -> tuple[dict[str, float], Tensor]:
        """One optimisation step on a batch of graphs.

        Returns the loss terms (``loss``, ``objective``, ``corr``, ``sparse``)
        and the detached global scores of the batch, shape ``(B, h)``. With
        ``record=False`` nothing is copied to the host (no device sync), which
        is what the timing benchmark uses.
        """
        cfg = self.config
        out = self.model(batch)
        objective = (-1.0 if cfg.objective == "maximize" else 1.0) * out.global_scores.mean()
        terms = dict(objective=objective, corr=objective.new_zeros(()), sparse=objective.new_zeros(()))
        loss = objective
        if cfg.lambda_corr > 0 and cfg.n_modes > 1:
            terms["corr"] = torch.stack([self.decorrelation_penalty(z) for z in out.z]).mean()
            loss = loss + cfg.lambda_corr * terms["corr"]
        if cfg.lambda_sparse > 0:
            terms["sparse"] = self.sparsity_penalty()
            loss = loss + cfg.lambda_sparse * terms["sparse"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if not record:
            return {}, out.global_scores.detach()
        terms["loss"] = loss
        return {k: float(v) for k, v in terms.items()}, out.global_scores.detach().cpu()

    def fit(self, graphs: Sequence[SpatialGraph], verbose: bool = True) -> "SAILModes":
        """Learn the modes on a list of graphs.

        Graphs are shuffled with NumPy's global RNG each epoch; seed NumPy and
        torch (:func:`sail.utils.set_seed`) for reproducible runs.
        """
        cfg = self.config
        best_W, epochs_without_improvement = None, 0

        for epoch in range(cfg.epochs):
            self.model.train()
            order = np.arange(len(graphs))
            np.random.shuffle(order)
            totals = dict(loss=0.0, objective=0.0, corr=0.0, sparse=0.0)
            score_sum, n_seen, n_batches = None, 0, 0

            for start in range(0, len(order), cfg.batch_size):
                batch = [graphs[j].to(self.device) for j in order[start : start + cfg.batch_size]]
                terms, batch_scores = self.step(batch)
                for key, value in terms.items():
                    totals[key] += value
                score_sum = batch_scores.sum(0) if score_sum is None else score_sum + batch_scores.sum(0)
                n_seen += batch_scores.shape[0]
                n_batches += 1

            mean_scores = score_sum / n_seen
            epoch_loss = totals["loss"]
            if self.best_loss - epoch_loss > cfg.min_delta:
                self.best_loss = epoch_loss
                best_W = self.model.f_map.W.detach().clone()
                self.mean_scores = mean_scores.clone()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            entry = EpochLog(
                epoch=epoch,
                loss=epoch_loss,
                objective=totals["objective"] / n_batches,
                corr_penalty=totals["corr"] / n_batches,
                sparse_penalty=totals["sparse"] / n_batches,
                lr=self.optimizer.param_groups[0]["lr"],
                temperature=float(self.model.temperature),
                mean_scores=mean_scores.tolist(),
            )
            self.history.append(entry)
            if verbose:
                log.info(
                    "epoch %4d | loss %.4f | objective %.4f | corr %.4g | sparse %.4g | lr %.2e | tau %.3f",
                    epoch, entry.loss, entry.objective, entry.corr_penalty,
                    entry.sparse_penalty, entry.lr, entry.temperature,
                )

            if epochs_without_improvement >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

        if best_W is None:
            raise RuntimeError("training produced no finite loss")
        self._export_best(best_W)
        return self

    def _export_best(self, best_W: Tensor) -> None:
        """Sort modes by score, fix their signs and swap in a plain linear map."""
        maximize = self.config.objective == "maximize"
        order = torch.argsort(self.mean_scores, descending=maximize)
        W = orient_columns(best_W[:, order.to(best_W.device)])
        self.mean_scores = self.mean_scores[order]

        f_map = NormalisedLinear(W.shape[0], W.shape[1]).to(self.device)
        f_map.W = W
        self.model.f_map = f_map
        if isinstance(self.model.f_affinity, SharedMapAffinity):
            self.model.f_affinity.f_map = f_map

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    @property
    def W(self) -> Tensor:
        """Mode loadings, shape ``(d, h)``; column ``m`` defines mode ``m + 1``."""
        return self.model.f_map.W.detach()

    @torch.no_grad()
    def transform(self, graphs: Union[SpatialGraph, Sequence[SpatialGraph]]) -> SAILOutput:
        """Run the trained model (eval mode, no gradients)."""
        self.model.eval()
        return self.model(graphs)

    @torch.no_grad()
    def scores(self, graphs: Sequence[SpatialGraph]) -> np.ndarray:
        """Global SAIL score of every mode for every graph, shape ``(n_graphs, h)``.

        Graphs are evaluated one at a time to bound memory.
        """
        self.model.eval()
        return np.stack([self.model(g).global_scores[0].cpu().numpy() for g in graphs])

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """Save config, trained weights and per-mode scores."""
        torch.save(
            {
                "format": FORMAT_VERSION,
                "input_dim": self.input_dim,
                "config": asdict(self.config),
                "trained_with_stiefel": self.trained_with_stiefel,
                "state_dict": self.model.state_dict(),
                "mean_scores": self.mean_scores,
                "best_loss": self.best_loss,
                "history": [asdict(h) for h in self.history],
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> "SAILModes":
        """Load a model saved with :meth:`save`.

        The exported (non-Stiefel) map is restored; models saved by the
        pre-release code can be converted with ``tools/convert_legacy_checkpoints.py``.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path} is not a {FORMAT_VERSION} checkpoint (see tools/convert_legacy_checkpoints.py)")
        obj = cls(ckpt["input_dim"], ModesConfig(**ckpt["config"]), device=device, _exported=True)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.mean_scores = ckpt["mean_scores"]
        obj.best_loss = ckpt["best_loss"]
        obj.history = [EpochLog(**h) for h in ckpt.get("history", [])]
        return obj
