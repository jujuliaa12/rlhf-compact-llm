"""
Supervised Fine-Tuning (SFT) training logic.

Uses TRL's SFTTrainer for stable training with LoRA adapters.
The SFT stage fine-tunes the base model on chosen responses from the
preference dataset as a lightweight instruction-following proxy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments

from src.data_utils import apply_debug_overrides, load_full_config, set_seed
from src.model_utils import (
    apply_lora,
    build_lora_config,
    load_causal_lm,
    load_tokenizer,
    save_model_and_tokenizer,
)

logger = logging.getLogger(__name__)


def build_training_args(cfg: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config dict."""
    tr = cfg.get("training", {})
    output_dir = tr.get("output_dir", "outputs/models/sft")

    # Mixed-precision: prefer bf16 (more stable), fall back to fp16, disable on CPU
    import torch
    use_fp16 = tr.get("fp16", False) and torch.cuda.is_available()
    use_bf16 = tr.get("bf16", False) and torch.cuda.is_available()
    # bf16 takes priority — disable fp16 if both are set
    if use_bf16:
        use_fp16 = False

    use_grad_ckpt = tr.get("gradient_checkpointing", True)
    if use_grad_ckpt and not torch.cuda.is_available():
        use_grad_ckpt = False

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=tr.get("num_train_epochs", 2),
        per_device_train_batch_size=tr.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=tr.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=tr.get("gradient_accumulation_steps", 4),
        learning_rate=tr.get("learning_rate", 2e-4),
        weight_decay=tr.get("weight_decay", 0.01),
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        lr_scheduler_type=tr.get("lr_scheduler_type", "cosine"),
        logging_steps=tr.get("logging_steps", 50),
        save_steps=tr.get("save_steps", 500),
        evaluation_strategy=tr.get("evaluation_strategy", "no"),
        do_eval=tr.get("do_eval", False),
        load_best_model_at_end=tr.get("load_best_model_at_end", False),
        save_total_limit=tr.get("save_total_limit", 1),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=use_grad_ckpt,
        report_to=tr.get("report_to", "none"),
        seed=cfg.get("seed", 42),
        remove_unused_columns=True,
    )

    # PyTorch >=2.1 warns about reentrant checkpointing; pass non-reentrant
    # to silence the warning and avoid subtle gradient bugs.
    if use_grad_ckpt:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return TrainingArguments(**kwargs)


def run_sft(
    config_path: str | Path,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
) -> dict:
    """
    Run SFT training from a config file.

    Args:
        config_path: Path to the SFT YAML config.
        train_dataset: Pre-loaded training dataset with 'text' column.
                       If None, loads and preprocesses from config.
        val_dataset: Pre-loaded validation dataset.

    Returns:
        Dict with keys: 'model', 'tokenizer', 'trainer', 'log_history'
    """
    from trl import SFTTrainer

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
    model = load_causal_lm(model_name, cfg)

    # Apply LoRA
    lora_config = build_lora_config(cfg, task_type="CAUSAL_LM")
    model = apply_lora(model, lora_config)

    # PEFT + gradient checkpointing compatibility:
    # When gradient checkpointing is enabled, the first layer's output is
    # detached from the graph.  enable_input_require_grads() re-attaches it
    # so LoRA parameters receive gradients.  On CPU we skip gradient
    # checkpointing entirely for stability.
    import torch
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
        from src.data_utils import load_hh_rlhf
        from src.preprocessing import prepare_sft_dataset, process_hh_rlhf_to_preference

        ds_cfg = cfg.get("dataset", {})
        raw = load_hh_rlhf(
            split=ds_cfg.get("split_train", "train"),
            max_samples=ds_cfg.get("max_train_samples"),
        )
        pref = process_hh_rlhf_to_preference(raw)
        train_dataset = prepare_sft_dataset(pref)

    # Training arguments
    training_args = build_training_args(cfg)

    # Create SFT trainer — eval_dataset and compute_metrics are explicitly
    # set to None to prevent the Trainer from triggering evaluation logic
    # at save_steps, which can crash if no valid eval data is loaded.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        compute_metrics=None,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save final model
    output_dir = cfg.get("training", {}).get("output_dir", "outputs/models/sft")
    save_model_and_tokenizer(model, tokenizer, output_dir)

    # Save training log as CSV
    csv_path = log_cfg.get("csv_log_path", "outputs/logs/sft_training_log.csv")
    save_training_log(trainer.state.log_history, csv_path)

    return {
        "model": model,
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
    logger.info("Training log saved to %s (%d entries)", csv_path, len(df))
