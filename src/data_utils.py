"""
Dataset loading and caching utilities.

Handles downloading datasets from Hugging Face Hub, caching locally,
and providing standardised access across the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base_path: str | Path, override: dict) -> dict:
    """Load base config and merge overrides on top."""
    base = load_yaml_config(base_path)
    # Simple shallow merge per top-level key
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def load_full_config(config_path: str | Path) -> dict:
    """Load a stage config, merging with its base_config if specified."""
    cfg = load_yaml_config(config_path)
    if "base_config" in cfg:
        base_path = Path(config_path).parent.parent / cfg["base_config"]
        cfg = merge_configs(base_path, cfg)
    return cfg


def apply_debug_overrides(cfg: dict) -> dict:
    """If debug mode is active, override sample counts and epochs."""
    if not cfg.get("debug", False):
        return cfg

    debug_max = cfg.get("debug_max_samples", 100)
    debug_eval = cfg.get("debug_max_eval_samples", 50)
    debug_epochs = cfg.get("debug_num_epochs", 1)
    debug_ppo = cfg.get("debug_max_ppo_steps", 20)

    # Override dataset limits
    ds = cfg.get("dataset", {})
    if "max_train_samples" in ds:
        ds["max_train_samples"] = min(ds["max_train_samples"], debug_max)
    else:
        ds["max_train_samples"] = debug_max
    if "max_val_samples" in ds:
        ds["max_val_samples"] = min(ds["max_val_samples"], debug_eval)
    else:
        ds["max_val_samples"] = debug_eval
    if "max_samples" in ds:
        ds["max_samples"] = min(ds["max_samples"], debug_max)

    # Override training epochs / steps
    tr = cfg.get("training", {})
    if "num_train_epochs" in tr:
        tr["num_train_epochs"] = debug_epochs

    ppo = cfg.get("ppo", {})
    if "max_steps" in ppo:
        ppo["max_steps"] = min(ppo["max_steps"], debug_ppo)

    logger.info(
        "DEBUG mode active — samples capped at %d train / %d val, epochs=%d",
        debug_max, debug_eval, debug_epochs,
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hh_rlhf(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load Anthropic HH-RLHF dataset.

    Each sample has 'chosen' and 'rejected' fields containing full
    conversation strings with \\n\\nHuman: and \\n\\nAssistant: turns.
    """
    logger.info("Loading Anthropic/hh-rlhf split=%s", split)
    ds = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        logger.info("Subset to %d samples", max_samples)
    return ds


def load_ultrafeedback(
    split: str = "train_prefs",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load UltraFeedback binarized dataset.

    Fields vary by version; typically has 'chosen' and 'rejected' as
    lists of message dicts with 'role' and 'content'.
    """
    logger.info("Loading HuggingFaceH4/ultrafeedback_binarized split=%s", split)
    ds = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", split=split, cache_dir=cache_dir
    )
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))
        logger.info("Subset to %d samples", max_samples)
    return ds


def save_dataset_to_disk(ds: Dataset | DatasetDict, path: str | Path) -> None:
    """Save a dataset to disk in Arrow format."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))
    logger.info("Dataset saved to %s", path)


def load_dataset_from_disk(path: str | Path) -> Dataset | DatasetDict:
    """Load a dataset previously saved with save_to_disk."""
    from datasets import load_from_disk

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved dataset at {path}")
    ds = load_from_disk(str(path))
    logger.info("Dataset loaded from %s (%d samples)", path, len(ds))
    return ds


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)
