#!/usr/bin/env python
"""
CLI entry point for Reward Model training.

Usage:
    python scripts/run_reward.py --config configs/reward_hh.yaml
    python scripts/run_reward.py --config configs/reward_alt.yaml --debug
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_reward")


def main():
    parser = argparse.ArgumentParser(description="Run reward model training")
    parser.add_argument(
        "--config", type=str, default="configs/reward_hh.yaml",
        help="Path to reward model YAML config",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Override to debug mode (tiny subset, 1 epoch)",
    )
    args = parser.parse_args()

    from src.data_utils import load_full_config, apply_debug_overrides, set_seed
    from src.reward_train import run_reward_training

    cfg = load_full_config(args.config)
    if args.debug:
        cfg["debug"] = True
    cfg = apply_debug_overrides(cfg)

    set_seed(cfg.get("seed", 42))

    logger.info("Starting reward model training with config: %s", args.config)
    logger.info("Debug mode: %s", cfg.get("debug", False))

    result = run_reward_training(args.config)

    logger.info("Reward model training complete.")
    logger.info("Model saved to: %s", cfg.get("training", {}).get("output_dir"))
    logger.info("Log saved to: %s", cfg.get("logging", {}).get("csv_log_path"))


if __name__ == "__main__":
    main()
