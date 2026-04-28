#!/usr/bin/env python
"""
CLI entry point for evaluation and analysis.

Generates comparison outputs from trained models, computes metrics,
and saves tables/figures.

Usage:
    python scripts/run_evaluation.py --config configs/base.yaml
    python scripts/run_evaluation.py --config configs/base.yaml --prompts 50
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
logger = logging.getLogger("run_evaluation")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to base YAML config",
    )
    parser.add_argument(
        "--prompts", type=int, default=50,
        help="Number of evaluation prompts to use",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip model generation, use existing comparison CSV",
    )
    args = parser.parse_args()

    import pandas as pd
    import torch
    from datasets import load_dataset

    from src.data_utils import load_dataset_from_disk, load_yaml_config, set_seed
    from src.evaluation import (
        analyze_reward_hacking,
        analyze_verbosity_bias,
        compute_win_rate,
        evaluate_model_outputs,
    )
    from src.inference import generate_responses, load_samples, save_samples
    from src.model_utils import load_causal_lm, load_peft_model, load_tokenizer
    from src.reward_train import compute_reward_scores

    cfg = load_yaml_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Paths
    model_name = cfg.get("models", {}).get("primary", "Qwen/Qwen2.5-0.5B")
    sft_dir = Path(cfg["paths"]["output_models"]) / "sft_qwen"
    rm_dir = Path(cfg["paths"]["output_models"]) / "reward_model_hh"
    ppo_dir = Path(cfg["paths"]["output_models"]) / "ppo_qwen"
    figures_dir = Path(cfg["paths"]["output_figures"])
    tables_dir = Path(cfg["paths"]["output_tables"])
    samples_dir = Path(cfg["paths"]["output_samples"])
    log_dir = Path(cfg["paths"]["output_logs"])

    for d in [figures_dir, tables_dir, samples_dir]:
        d.mkdir(parents=True, exist_ok=True)

    comparison_path = samples_dir / "model_comparison.csv"

    # ------------------------------------------------------------------
    # Step 1: Generate responses (or load existing)
    # ------------------------------------------------------------------
    if args.skip_generation and comparison_path.exists():
        logger.info("Loading existing comparison data from %s", comparison_path)
        comparison_df = load_samples(comparison_path)
    else:
        logger.info("Generating responses from all available models...")

        # Load evaluation prompts — try preprocessed data first, fall back to raw
        test_path = Path(cfg["paths"]["data_processed"]) / "hh_rlhf" / "test"
        if test_path.exists():
            test_data = load_dataset_from_disk(test_path)
            eval_prompts = test_data["prompt"][:args.prompts]
        else:
            logger.info("No preprocessed test data found, loading from HH-RLHF directly...")
            from src.preprocessing import process_hh_rlhf_to_preference
            raw_test = load_dataset("Anthropic/hh-rlhf", split="test")
            if len(raw_test) > args.prompts * 2:
                raw_test = raw_test.select(range(args.prompts * 2))
            test_pref = process_hh_rlhf_to_preference(raw_test)
            eval_prompts = test_pref["prompt"][:args.prompts]
        logger.info("Using %d evaluation prompts", len(eval_prompts))

        tokenizer = load_tokenizer(model_name)
        prompt_template = "### Human:\n{prompt}\n\n### Assistant:\n"
        model_cfg = {"training": {"fp16": False, "bf16": False}}

        records = []

        # Base model
        logger.info("Generating from base model...")
        base_model = load_causal_lm(model_name, model_cfg)
        base_responses = generate_responses(
            base_model, tokenizer, eval_prompts,
            max_new_tokens=128, prompt_template=prompt_template,
        )
        for p, r in zip(eval_prompts, base_responses):
            records.append({"prompt": p, "model_name": "base", "response": r})
        del base_model

        # SFT model
        if sft_dir.exists():
            logger.info("Generating from SFT model...")
            sft_model = load_peft_model(model_name, sft_dir, model_cfg)
            sft_responses = generate_responses(
                sft_model, tokenizer, eval_prompts,
                max_new_tokens=128, prompt_template=prompt_template,
            )
            for p, r in zip(eval_prompts, sft_responses):
                records.append({"prompt": p, "model_name": "sft", "response": r})
            del sft_model
        else:
            logger.warning("SFT model not found at %s — skipping", sft_dir)

        # PPO model (if available)
        if ppo_dir.exists():
            logger.info("Generating from PPO model...")
            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                base = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=torch.float32,
                )
                ppo_model = PeftModel.from_pretrained(base, str(ppo_dir))
                ppo_responses = generate_responses(
                    ppo_model, tokenizer, eval_prompts,
                    max_new_tokens=128, prompt_template=prompt_template,
                )
                for p, r in zip(eval_prompts, ppo_responses):
                    records.append({"prompt": p, "model_name": "ppo", "response": r})
                del ppo_model, base
            except Exception as e:
                logger.warning("Failed to load PPO model: %s", e)
        else:
            logger.warning("PPO model not found at %s — skipping", ppo_dir)

        comparison_df = pd.DataFrame(records)
        save_samples(comparison_df, comparison_path)
        logger.info("Comparison data saved to %s (%d rows)", comparison_path, len(comparison_df))

    # ------------------------------------------------------------------
    # Step 2: Automatic metrics
    # ------------------------------------------------------------------
    logger.info("Computing automatic metrics...")
    summary_df = evaluate_model_outputs(comparison_df)
    summary_df.to_csv(tables_dir / "evaluation_summary.csv", index=False)
    logger.info("Evaluation summary:\n%s", summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Step 3: Reward-based win rate (if reward model available)
    # ------------------------------------------------------------------
    if rm_dir.exists() and "sft" in comparison_df["model_name"].values:
        logger.info("Computing reward-based win rates...")
        try:
            rm_model = load_peft_model(model_name, rm_dir, model_cfg, is_reward_model=True)
            rm_model.eval()
            rm_tokenizer = load_tokenizer(model_name)

            for challenger in ["ppo", "base"]:
                if challenger in comparison_df["model_name"].values:
                    sft_rows = comparison_df[comparison_df["model_name"] == "sft"]
                    chl_rows = comparison_df[comparison_df["model_name"] == challenger]

                    sft_texts = [f"{p}\n\n{r}" for p, r in zip(sft_rows["prompt"], sft_rows["response"])]
                    chl_texts = [f"{p}\n\n{r}" for p, r in zip(chl_rows["prompt"], chl_rows["response"])]

                    sft_scores = compute_reward_scores(rm_model, rm_tokenizer, sft_texts, 512)
                    chl_scores = compute_reward_scores(rm_model, rm_tokenizer, chl_texts, 512)

                    wr = compute_win_rate(chl_scores, sft_scores, challenger, "sft")
                    logger.info("Win rate %s vs sft: %s", challenger, wr)

            del rm_model
        except Exception as e:
            logger.warning("Reward-based win rate failed: %s", e)

    # ------------------------------------------------------------------
    # Step 4: Verbosity and reward hacking analysis
    # ------------------------------------------------------------------
    ppo_log_path = log_dir / "ppo_training_log.csv"
    if ppo_log_path.exists():
        logger.info("Analysing reward hacking indicators...")
        ppo_df = pd.read_csv(ppo_log_path)
        indicators = analyze_reward_hacking(ppo_df)
        pd.DataFrame([indicators]).to_csv(tables_dir / "reward_hacking_indicators.csv", index=False)
        logger.info("Reward hacking indicators: %s", indicators)

    if "sft" in comparison_df["model_name"].values and "ppo" in comparison_df["model_name"].values:
        sft_resp = comparison_df[comparison_df["model_name"] == "sft"]["response"].tolist()
        ppo_resp = comparison_df[comparison_df["model_name"] == "ppo"]["response"].tolist()
        verbosity = analyze_verbosity_bias(sft_resp, ppo_resp)
        pd.DataFrame([verbosity]).to_csv(tables_dir / "verbosity_analysis.csv", index=False)
        logger.info("Verbosity analysis: %s", verbosity)

    logger.info("Evaluation pipeline complete. Outputs in %s and %s", tables_dir, samples_dir)


if __name__ == "__main__":
    main()
