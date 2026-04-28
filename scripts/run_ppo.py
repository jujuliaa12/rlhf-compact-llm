#!/usr/bin/env python
"""
CLI entry point for PPO-based RLHF training.

Usage:
    python scripts/run_ppo.py --config configs/ppo_qwen.yaml
    python scripts/run_ppo.py --config configs/ppo_qwen.yaml --debug

NOTE: This is the most version-sensitive part of the pipeline.
Requires trl==0.9.6 with transformers==4.41.0.
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
logger = logging.getLogger("run_ppo")


def main():
    parser = argparse.ArgumentParser(description="Run PPO-based RLHF training")
    parser.add_argument(
        "--config", type=str, default="configs/ppo_qwen.yaml",
        help="Path to PPO YAML config",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Override to debug mode (tiny subset, few steps)",
    )
    args = parser.parse_args()

    from src.data_utils import load_full_config, apply_debug_overrides, set_seed
    from src.ppo_train import run_ppo

    cfg = load_full_config(args.config)
    if args.debug:
        cfg["debug"] = True
    cfg = apply_debug_overrides(cfg)

    set_seed(cfg.get("seed", 42))

    logger.info("Starting PPO training with config: %s", args.config)
    logger.info("Debug mode: %s", cfg.get("debug", False))

    result = run_ppo(args.config)

    logger.info("PPO training complete.")
    logger.info("Model saved to: %s", cfg.get("ppo", {}).get("output_dir", "outputs/models/ppo_qwen"))
    logger.info("Log saved to: %s", cfg.get("logging", {}).get("csv_log_path", "outputs/logs/ppo_training_log.csv"))


if __name__ == "__main__":
    main()
