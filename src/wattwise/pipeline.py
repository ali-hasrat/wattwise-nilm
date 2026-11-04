"""Command-line entry point: ``python -m wattwise.pipeline --stage all``."""

from __future__ import annotations

import argparse
import logging
import time

from .config import load_config, setup_logging

STAGES = ("raw", "prepare", "eda", "model")
log = logging.getLogger("wattwise")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="WattWise NILM pipeline")
    ap.add_argument("--stage", choices=[*STAGES, "all"], default=["all"], nargs="+",
                    help="raw = Task 1; prepare = build 1-min cleaned-data cache; eda = Task 2; model = Task 3")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["_paths"]
    paths.ensure()
    setup_logging(paths.logs / "pipeline.log")
    stages = STAGES if "all" in args.stage else [s for s in STAGES if s in args.stage]

    for stage in stages:
        t0 = time.time()
        log.info("=== stage %s ===", stage)
        if stage == "raw":
            from . import stage_raw
            stage_raw.run(cfg)
        elif stage == "prepare":
            from . import stage_prepare
            stage_prepare.run(cfg)
        elif stage == "eda":
            from . import stage_eda
            stage_eda.run(cfg)
        elif stage == "model":
            from . import stage_model
            stage_model.run(cfg)
        log.info("=== stage %s finished in %.0f s ===", stage, time.time() - t0)


if __name__ == "__main__":
    main()
