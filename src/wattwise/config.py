"""Configuration loading, project paths and logging setup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_dir: Path
    clean_dir: Path
    output_dir: Path

    @property
    def data(self) -> Path:
        return self.output_dir / "data"

    @property
    def figures(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables(self) -> Path:
        return self.output_dir / "tables"

    @property
    def logs(self) -> Path:
        return self.output_dir / "logs"

    def ensure(self) -> None:
        for p in (self.data, self.figures, self.tables, self.logs):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config. Relative paths inside it resolve against the config file's folder."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("paths", "raw_cleaning", "washing_machine", "cycle_detection", "model"):
        if section not in cfg:
            raise KeyError(f"Config is missing required section '{section}'")
    base = cfg_path.resolve().parent
    p = cfg["paths"]
    cfg["_paths"] = Paths(
        root=base,
        raw_dir=(base / p["raw_dir"]).resolve(),
        clean_dir=(base / p["clean_dir"]).resolve(),
        output_dir=(base / p["output_dir"]).resolve(),
    )
    return cfg


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
