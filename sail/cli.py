"""``sail-run``: run an experiment from its YAML config.

    sail-run experiments/4i/config.yaml                 # all stages
    sail-run experiments/4i/config.yaml --stage eval    # reuse trained models
    sail-run experiments/4i/config.yaml --stage ablation

Each experiment folder provides ``run.py`` with a ``STAGES`` dict mapping
stage names to ``fn(cfg)``; ``all`` runs the stages listed in ``DEFAULT_STAGES``.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

from .config import load_config
from .utils import set_seed


def _load_runner(config_path: Path):
    run_py = config_path.parent / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"No run.py next to {config_path}")
    sys.path.insert(0, str(config_path.parent))  # runners may import helpers next to them
    spec = importlib.util.spec_from_file_location(f"sail_experiment_{config_path.parent.name}", run_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sail-run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment YAML file")
    parser.add_argument("--stage", default="all", help="stage to run (see the experiment README)")
    parser.add_argument("--output-dir", help="override output_dir")
    parser.add_argument("--device", help="override device (cpu / cuda)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    overrides = {"output_dir": args.output_dir} if args.output_dir else {}
    cfg = load_config(args.config, **overrides)
    if args.device:
        cfg.eval["device"] = args.device
    runner = _load_runner(args.config.resolve())

    stages = runner.DEFAULT_STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage not in runner.STAGES:
            parser.error(f"unknown stage {stage!r}; choose from {list(runner.STAGES)}")
        set_seed(cfg.seed)
        logging.getLogger("sail").warning("[%s] stage: %s", cfg.experiment, stage)
        runner.STAGES[stage](cfg)


if __name__ == "__main__":
    main()
