"""
Direct Preference Optimization (DPO) training logic.

DPO replaces the SFT → reward model → PPO stack with a single training step
that consumes (prompt, chosen, rejected) preference triples directly, using
a closed-form loss derived from the Bradley-Terry preference model.

Compared to PPO, DPO:
- removes the explicit reward model
- removes online generation / rollout buffers
- has no KL controller (KL is implicit in the loss via the `beta` coefficient)
- is dramatically more stable, especially at compact scales

This module is intentionally parallel to ``src.ppo_train``: same config style,
same logging targets, same LoRA-on-SFT-base flow.

Targets ``trl==0.9.6``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments

from src.data_utils import apply_debug_overrides, load_full_config, set_seed
from src.model_utils import build_lora_config, load_causal_lm, load_tokenizer

logger = logging.getLogger(__name__)


def build_training_args(cfg: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from the DPO config dict."""
    import torch

    tr = cfg.get("training", {})
    output_dir = cfg.get("dpo", {}).get("output_dir", "outputs/models/dpo_qwen")

    use_fp16 = tr.get("fp16", False) and torch.cuda.is_available()
    use_bf16 = tr.get("bf16", False) and torch.cuda.is_available()
    if use_bf16:
        use_fp16 = False

    use_grad_ckpt = tr.get("gradient_checkpointing", True) and torch.cuda.is_available()

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=tr.get("num_train_epochs", 1),
        per_device_train_batch_size=tr.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 8),
        learning_rate=tr.get("learning_rate", 5e-5),
        weight_decay=tr.get("weight_decay", 0.0),
        warmup_ratio=tr.get("warmup_ratio", 0.1),
        lr_scheduler_type=tr.get("lr_scheduler_type", "cosine"),
        logging_steps=tr.get("logging_steps", 25),
        save_steps=tr.get("save_steps", 500),
        save_total_limit=tr.get("save_total_limit", 1),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_grad_ckpt,
        report_to=tr.get("report_to", "none"),
        seed=cfg.get("seed", 42),
        remove_unused_columns=tr.get("remove_unused_columns", False),
    )

    if use_grad_ckpt:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def load_preference_dataset(cfg: dict) -> Dataset:
    """Load HH-RLHF and convert to the prompt/chosen/rejected format DPO expects."""
    from src.data_utils import load_hh_rlhf
    from src.preprocessing import process_hh_rlhf_to_preference

    ds_cfg = cfg.get("dataset", {})
    raw = load_hh_rlhf(
        split=ds_cfg.get("split_train", "train"),
        max_samples=ds_cfg.get("max_train_samples"),
    )
    return process_hh_rlhf_to_preference(raw)


def run_dpo(
    config_path: str | Path,
    train_dataset: Dataset | None = None,
) -> dict:
    """
    Run DPO training from a config file.

    Args:
        config_path: Path to the DPO YAML config.
        train_dataset: Optional pre-loaded preference dataset with
                       columns ``prompt``, ``chosen``, ``rejected``.

    Returns:
        Dict with ``model``, ``tokenizer``, ``trainer``, ``log_history``.
    """
    import torch
    from peft import PeftModel
    from trl import DPOTrainer

    cfg = load_full_config(config_path)
    cfg = apply_debug_overrides(cfg)
    set_seed(cfg.get("seed", 42))

    log_cfg = cfg.get("logging", {})
    snapshot_path = log_cfg.get("config_snapshot_path")
    if snapshot_path:
        import yaml
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    sft_cfg = cfg.get("sft_model", {})
    base_model_name = sft_cfg.get("base_model", "Qwen/Qwen2.5-0.5B")
    sft_adapter_path = sft_cfg.get("path", "outputs/models/sft_qwen")

    tokenizer = load_tokenizer(base_model_name)
    base = load_causal_lm(base_model_name, cfg)

    # Continue from the SFT adapter as the policy initialisation, then attach
    # a fresh LoRA on top for DPO. This mirrors the canonical PEFT-DPO recipe.
    if Path(sft_adapter_path).exists():
        logger.info("Initialising policy from SFT adapter at %s", sft_adapter_path)
        policy = PeftModel.from_pretrained(base, sft_adapter_path, is_trainable=True)
        # Merge SFT into the base weights so DPO's new LoRA adapter starts clean.
        policy = policy.merge_and_unload()
    else:
        logger.warning(
            "SFT adapter not found at %s — training DPO directly on the raw base model. "
            "Results will be substantially worse than the SFT-initialised path.",
            sft_adapter_path,
        )
        policy = base

    if cfg.get("training", {}).get("gradient_checkpointing", False) and torch.cuda.is_available():
        policy.enable_input_require_grads()
        logger.info("enable_input_require_grads() called for PEFT + grad-ckpt compatibility")

    lora_config = build_lora_config(cfg, task_type="CAUSAL_LM")

    if train_dataset is None:
        train_dataset = load_preference_dataset(cfg)
    logger.info("DPO training set: %d preference triples", len(train_dataset))

    training_args = build_training_args(cfg)
    dpo_cfg = cfg.get("dpo", {})

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,  # PEFT path: ref is the LoRA-disabled base
        args=training_args,
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=dpo_cfg.get("max_length", 512),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 256),
        peft_config=lora_config,
        reference_free=dpo_cfg.get("reference_free", False),
    )

    logger.info("Starting DPO training...")
    trainer.train()

    output_dir = dpo_cfg.get("output_dir", "outputs/models/dpo_qwen")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("DPO model saved to %s", output_dir)

    csv_path = log_cfg.get("csv_log_path", "outputs/logs/dpo_training_log.csv")
    save_training_log(trainer.state.log_history, csv_path)

    return {
        "model": trainer.model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "log_history": trainer.state.log_history,
    }


def save_training_log(log_history: list[dict], csv_path: str | Path) -> None:
    """Save trainer log history to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(log_history)
    df.to_csv(csv_path, index=False)
    logger.info("DPO training log saved to %s (%d entries)", csv_path, len(df))
