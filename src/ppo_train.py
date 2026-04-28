"""
PPO-based RLHF training logic.

Uses TRL's PPOTrainer to align the SFT model using reward model scores.

WARNING: TRL's PPO API changes frequently between versions.
This module targets trl==0.9.6. If you upgrade TRL, review the PPO API
documentation and update this code accordingly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset

from src.data_utils import apply_debug_overrides, load_full_config, set_seed

logger = logging.getLogger(__name__)


class PPOLogger:
    """Simple CSV logger for PPO training metrics."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def log(self, step: int, metrics: dict) -> None:
        """Log a single step's metrics."""
        record = {"step": step, **metrics}
        self.records.append(record)

    def save(self) -> None:
        """Save all logged records to CSV."""
        if self.records:
            df = pd.DataFrame(self.records)
            df.to_csv(self.csv_path, index=False)
            logger.info("PPO log saved to %s (%d entries)", self.csv_path, len(df))

    def to_dataframe(self) -> pd.DataFrame:
        """Return logged records as DataFrame."""
        return pd.DataFrame(self.records)


def prepare_ppo_prompts(
    ds: Dataset,
    tokenizer,
    max_prompt_length: int = 256,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """
    Prepare prompts for PPO generation.

    Tokenizes prompts and returns a list of dicts with 'query' (token IDs)
    and 'prompt_text' (original string).
    """
    prompts = ds["prompt"]
    if max_samples is not None:
        prompts = prompts[:max_samples]

    prepared = []
    for prompt_text in prompts:
        encoded = tokenizer.encode(
            prompt_text,
            max_length=max_prompt_length,
            truncation=True,
        )
        prepared.append({
            "query": encoded,
            "prompt_text": prompt_text,
        })

    logger.info("Prepared %d PPO prompts (max_length=%d)", len(prepared), max_prompt_length)
    return prepared


def run_ppo(
    config_path: str | Path,
    prompt_dataset: Optional[Dataset] = None,
) -> dict:
    """
    Run PPO-based RLHF training.

    This function orchestrates:
    1. Loading the SFT model as the starting policy
    2. Loading the reward model for scoring
    3. Generating responses from the policy
    4. Computing reward scores
    5. Running PPO updates

    Args:
        config_path: Path to PPO YAML config.
        prompt_dataset: Dataset with 'prompt' column. If None, loads from config.

    Returns:
        Dict with keys: 'model', 'tokenizer', 'ppo_log'
    """
    # Import TRL PPO components — isolated here for version safety
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    except ImportError as e:
        logger.error(
            "Failed to import TRL PPO components. Ensure trl==0.9.6 is installed. "
            "Error: %s", e
        )
        raise

    cfg = load_full_config(config_path)
    cfg = apply_debug_overrides(cfg)
    set_seed(cfg.get("seed", 42))

    # Save config snapshot
    log_cfg = cfg.get("logging", {})
    snapshot_path = log_cfg.get("config_snapshot_path")
    if snapshot_path:
        import yaml
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # --- Load SFT model as policy ---
    from src.model_utils import load_tokenizer

    sft_cfg = cfg.get("sft_model", {})
    base_model_name = sft_cfg.get("base_model", cfg.get("models", {}).get("primary"))
    sft_path = sft_cfg.get("path", "outputs/models/sft_qwen")

    tokenizer = load_tokenizer(base_model_name)

    # Two-step policy loading:
    # 1. Load base causal LM + attach saved LoRA adapter via PEFT
    # 2. Wrap the PeftModel with AutoModelForCausalLMWithValueHead
    # This avoids passing peft_pretrained_model_name_or_path into
    # from_pretrained() which causes a TypeError in trl==0.9.6 +
    # transformers==4.41.0.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    logger.info("Loading base model + SFT LoRA adapter + value head from %s", sft_path)
    from src.model_utils import get_dtype
    compute_dtype = get_dtype(cfg)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    base_model = PeftModel.from_pretrained(base_model, sft_path)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

    # --- Load reward model ---
    from src.model_utils import load_peft_model
    from src.reward_train import compute_reward_scores

    rm_cfg = cfg.get("reward_model", {})
    rm_base = rm_cfg.get("base_model", base_model_name)
    rm_path = rm_cfg.get("path", "outputs/models/reward_model_hh")

    logger.info("Loading reward model from %s", rm_path)
    reward_model = load_peft_model(rm_base, rm_path, cfg, is_reward_model=True)
    reward_model.eval()
    rm_tokenizer = load_tokenizer(rm_base)

    # --- Prepare prompts ---
    if prompt_dataset is None:
        from src.data_utils import load_hh_rlhf
        from src.preprocessing import process_hh_rlhf_to_preference

        ds_cfg = cfg.get("dataset", {})
        raw = load_hh_rlhf(split="train", max_samples=ds_cfg.get("max_samples"))
        prompt_dataset = process_hh_rlhf_to_preference(raw)

    gen_cfg = cfg.get("generation", {})
    ppo_cfg_dict = cfg.get("ppo", {})

    # --- Configure PPO ---
    ppo_config = PPOConfig(
        learning_rate=ppo_cfg_dict.get("learning_rate", 1.41e-5),
        batch_size=ppo_cfg_dict.get("batch_size", 16),
        mini_batch_size=ppo_cfg_dict.get("mini_batch_size", 4),
        ppo_epochs=ppo_cfg_dict.get("ppo_epochs", 4),
        init_kl_coef=ppo_cfg_dict.get("init_kl_coef", 0.2),
        target=ppo_cfg_dict.get("target_kl", 6.0),
        gamma=ppo_cfg_dict.get("gamma", 1.0),
        lam=ppo_cfg_dict.get("lam", 0.95),
        cliprange=ppo_cfg_dict.get("cliprange", 0.2),
        cliprange_value=ppo_cfg_dict.get("cliprange_value", 0.2),
        seed=cfg.get("seed", 42),
    )

    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )

    # --- PPO Training Loop ---
    max_steps = ppo_cfg_dict.get("max_steps", 500)
    log_every = log_cfg.get("log_every_n_steps", 10)
    max_new_tokens = gen_cfg.get("max_new_tokens", 128)
    batch_size = ppo_cfg_dict.get("batch_size", 16)

    ppo_logger = PPOLogger(
        log_cfg.get("csv_log_path", "outputs/logs/ppo_training_log.csv")
    )

    prompts = prompt_dataset["prompt"]
    prompt_idx = 0

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": gen_cfg.get("min_new_tokens", 16),
        "temperature": gen_cfg.get("temperature", 0.7),
        "top_k": gen_cfg.get("top_k", 50),
        "top_p": gen_cfg.get("top_p", 0.95),
        "do_sample": gen_cfg.get("do_sample", True),
        "pad_token_id": tokenizer.pad_token_id,
    }

    logger.info("Starting PPO training for %d steps...", max_steps)

    for step in range(max_steps):
        # Gather a batch of prompts (cycle if needed)
        batch_prompts = []
        for _ in range(batch_size):
            batch_prompts.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1

        # Tokenize prompts
        query_tensors = [
            tokenizer.encode(p, return_tensors="pt", truncation=True,
                             max_length=cfg.get("dataset", {}).get("max_prompt_length", 256)
                             ).squeeze(0)
            for p in batch_prompts
        ]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors, **generation_kwargs
        )

        # Decode responses
        response_texts = [
            tokenizer.decode(r.squeeze(), skip_special_tokens=True)
            for r in response_tensors
        ]

        # Compute reward scores
        combined_texts = [
            f"{p}\n\n{r}" for p, r in zip(batch_prompts, response_texts)
        ]
        rewards = compute_reward_scores(
            reward_model, rm_tokenizer, combined_texts,
            max_length=cfg.get("model", {}).get("max_seq_length", 512),
        )
        reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        # Log metrics
        if step % log_every == 0:
            mean_reward = sum(rewards) / len(rewards)
            mean_length = sum(len(r) for r in response_texts) / len(response_texts)

            metrics = {
                "mean_reward": mean_reward,
                "response_length_mean": mean_length,
            }

            # Extract KL from PPO stats — key name varies across TRL versions
            kl = None
            for kl_key in ("ppo/mean_kl", "ppo/kl", "objective/kl"):
                if kl_key in stats:
                    kl = stats[kl_key]
                    break
            metrics["kl_divergence"] = kl

            ppo_logger.log(step, metrics)
            logger.info(
                "Step %d/%d — mean_reward=%.4f, mean_len=%.1f",
                step, max_steps, mean_reward, mean_length,
            )

    # Save PPO log
    ppo_logger.save()

    # Save aligned model
    output_dir = ppo_cfg_dict.get("output_dir", "outputs/models/ppo_qwen")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("PPO-aligned model saved to %s", output_dir)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "ppo_log": ppo_logger.to_dataframe(),
    }
