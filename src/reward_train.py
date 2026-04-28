"""
Reward model training logic.

Trains a scalar reward model from preference pairs using either
TRL's RewardTrainer or a manual training loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments

from src.data_utils import apply_debug_overrides, load_full_config, set_seed
from src.model_utils import (
    apply_lora,
    build_lora_config,
    load_reward_model,
    load_tokenizer,
    save_model_and_tokenizer,
)

logger = logging.getLogger(__name__)


def tokenize_preference_pair(
    sample: dict,
    tokenizer,
    max_length: int = 512,
) -> dict:
    """
    Tokenize a preference pair for reward model training.

    Concatenates prompt + response for both chosen and rejected,
    then tokenizes each.
    """
    chosen_text = f"{sample['prompt']}\n\n{sample['chosen']}"
    rejected_text = f"{sample['prompt']}\n\n{sample['rejected']}"

    chosen_enc = tokenizer(
        chosen_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    rejected_enc = tokenizer(
        rejected_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    return {
        "input_ids_chosen": chosen_enc["input_ids"],
        "attention_mask_chosen": chosen_enc["attention_mask"],
        "input_ids_rejected": rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
    }


def prepare_reward_dataset(
    ds: Dataset,
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Tokenize a preference dataset for reward training."""
    logger.info("Tokenizing preference pairs (max_length=%d)...", max_length)
    tokenized = ds.map(
        lambda x: tokenize_preference_pair(x, tokenizer, max_length),
        desc="Tokenizing for reward model",
        remove_columns=ds.column_names,
    )
    return tokenized


def build_reward_training_args(cfg: dict) -> TrainingArguments:
    """Build TrainingArguments for reward model training."""
    tr = cfg.get("training", {})
    output_dir = tr.get("output_dir", "outputs/models/reward_model")

    # Mixed-precision: prefer bf16, fall back to fp16, disable on CPU
    use_fp16 = tr.get("fp16", False) and torch.cuda.is_available()
    use_bf16 = tr.get("bf16", False) and torch.cuda.is_available()
    if use_bf16:
        use_fp16 = False

    use_grad_ckpt = tr.get("gradient_checkpointing", True)
    if use_grad_ckpt and not torch.cuda.is_available():
        use_grad_ckpt = False

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=tr.get("num_train_epochs", 1),
        per_device_train_batch_size=tr.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=tr.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 4),
        learning_rate=tr.get("learning_rate", 1e-4),
        weight_decay=tr.get("weight_decay", 0.01),
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        lr_scheduler_type=tr.get("lr_scheduler_type", "cosine"),
        logging_steps=tr.get("logging_steps", 50),
        save_steps=tr.get("save_steps", 500),
        eval_steps=tr.get("eval_steps", 250),
        evaluation_strategy="steps" if tr.get("eval_steps") else "no",
        save_total_limit=tr.get("save_total_limit", 2),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_grad_ckpt,
        report_to=tr.get("report_to", "none"),
        seed=cfg.get("seed", 42),
        remove_unused_columns=False,
    )

    if use_grad_ckpt:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def run_reward_training(
    config_path: str | Path,
    train_dataset: Optional[Dataset] = None,
    val_dataset: Optional[Dataset] = None,
) -> dict:
    """
    Run reward model training from a config file.

    Args:
        config_path: Path to the reward model YAML config.
        train_dataset: Pre-loaded preference dataset with prompt/chosen/rejected.
        val_dataset: Pre-loaded validation preference dataset.

    Returns:
        Dict with keys: 'model', 'tokenizer', 'trainer', 'log_history'
    """
    from trl import RewardTrainer

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

    # Load model and tokenizer
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", cfg.get("models", {}).get("primary"))
    max_seq_length = model_cfg.get("max_seq_length", 512)

    tokenizer = load_tokenizer(model_name, max_seq_length=max_seq_length)
    model = load_reward_model(model_name, cfg, num_labels=1)

    # Apply LoRA for reward model (SEQ_CLS task type)
    lora_config = build_lora_config(cfg, task_type="SEQ_CLS")
    model = apply_lora(model, lora_config)

    # PEFT + gradient checkpointing compatibility (same as SFT)
    use_grad_ckpt = cfg.get("training", {}).get("gradient_checkpointing", False)
    if use_grad_ckpt and not torch.cuda.is_available():
        logger.warning("Disabling gradient checkpointing on CPU for stability")
        use_grad_ckpt = False
        cfg.setdefault("training", {})["gradient_checkpointing"] = False
    if use_grad_ckpt:
        model.enable_input_require_grads()
        logger.info("enable_input_require_grads() called for PEFT + grad-ckpt compatibility")
    model.train()

    # Load data if not provided
    if train_dataset is None:
        from src.preprocessing import process_hh_rlhf_to_preference, clean_dataset, filter_empty_rows
        from src.data_utils import load_hh_rlhf

        ds_cfg = cfg.get("dataset", {})
        raw = load_hh_rlhf(
            split="train",
            max_samples=ds_cfg.get("max_train_samples"),
        )
        train_dataset = process_hh_rlhf_to_preference(raw)
        train_dataset = clean_dataset(train_dataset)
        train_dataset = filter_empty_rows(train_dataset)

    if val_dataset is None:
        from src.preprocessing import process_hh_rlhf_to_preference, clean_dataset, filter_empty_rows
        from src.data_utils import load_hh_rlhf

        ds_cfg = cfg.get("dataset", {})
        raw_val = load_hh_rlhf(
            split="test",
            max_samples=ds_cfg.get("max_val_samples"),
        )
        val_dataset = process_hh_rlhf_to_preference(raw_val)
        val_dataset = clean_dataset(val_dataset)
        val_dataset = filter_empty_rows(val_dataset)

    # Tokenize for reward trainer
    train_tokenized = prepare_reward_dataset(train_dataset, tokenizer, max_seq_length)
    val_tokenized = prepare_reward_dataset(val_dataset, tokenizer, max_seq_length)

    # Training arguments
    training_args = build_reward_training_args(cfg)

    # Create reward trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting reward model training...")
    trainer.train()

    # Save final model
    output_dir = cfg.get("training", {}).get("output_dir", "outputs/models/reward_model")
    save_model_and_tokenizer(model, tokenizer, output_dir)

    # Save training log
    csv_path = log_cfg.get("csv_log_path", "outputs/logs/reward_training_log.csv")
    save_reward_training_log(trainer.state.log_history, csv_path)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "log_history": trainer.state.log_history,
    }


def save_reward_training_log(log_history: list[dict], csv_path: str | Path) -> None:
    """Save reward model training log to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(log_history)
    df.to_csv(csv_path, index=False)
    logger.info("Reward training log saved to %s", csv_path)


# ---------------------------------------------------------------------------
# Reward model inference
# ---------------------------------------------------------------------------

def compute_reward_scores(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    batch_size: int = 8,
) -> list[float]:
    """
    Compute scalar reward scores for a list of texts.

    Args:
        model: Reward model (sequence classification with num_labels=1).
        tokenizer: Corresponding tokenizer.
        texts: List of text strings (prompt + response concatenated).
        max_length: Maximum token length.
        batch_size: Batch size for inference.

    Returns:
        List of scalar reward scores.
    """
    model.eval()
    device = next(model.parameters()).device
    scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            # RewardTrainer outputs logits of shape (batch, 1)
            batch_scores = outputs.logits.squeeze(-1).cpu().tolist()

        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores
