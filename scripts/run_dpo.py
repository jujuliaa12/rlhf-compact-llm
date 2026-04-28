#!/usr/bin/env python
"""
CLI entry point for Direct Preference Optimization (DPO) training.

Usage:
    python scripts/run_dpo.py --config configs/dpo_qwen.yaml
    python scripts/run_dpo.py --config configs/dpo_qwen.yaml --debug

DPO is the modern alternative to PPO at compact scales: it trains directly
on (prompt, chosen, rejected) preference triples with no separate reward
model and no online generation. See ``docs/DESIGN.md`` and the README
"Pipeline" section.

Targets ``trl==0.9.6`` with ``transformers==4.41.0``.
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
logger = logging.getLogger("run_dpo")


def main():
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument(
        "--config", type=str, default="configs/dpo_qwen.yaml",
        help="Path to DPO YAML config",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Override to debug mode (tiny subset, few steps)",
    )
    args = parser.parse_args()

    from src.data_utils import load_full_config, set_seed
    from src.dpo_train import run_dpo

    cfg = load_full_config(args.config)
    set_seed(cfg.get("seed", 42))

    logger.info("Starting DPO training with config: %s", args.config)
    logger.info("Debug mode: %s", args.debug)

    run_dpo(args.config, debug=args.debug)

    logger.info("DPO training complete.")
    logger.info(
        "Model saved to: %s",
        cfg.get("dpo", {}).get("output_dir", "outputs/models/dpo_qwen"),
    )
    logger.info(
        "Log saved to: %s",
        cfg.get("logging", {}).get("csv_log_path", "outputs/logs/dpo_training_log.csv"),
    )


if __name__ == "__main__":
    main()
