"""
Model and tokenizer loading utilities.

Handles loading base models with optional LoRA/QLoRA adapters,
configuring tokenizers, and detecting hardware capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Detect the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    else:
        device = torch.device(preference)
        logger.info("Using device: %s", device)
    return device


def get_dtype(cfg: dict) -> torch.dtype:
    """Determine the compute dtype from config."""
    if cfg.get("training", {}).get("bf16", False) and torch.cuda.is_available():
        return torch.bfloat16
    if cfg.get("training", {}).get("fp16", False) and torch.cuda.is_available():
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(
    model_name: str,
    max_seq_length: int | None = None,
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    """
    Load and configure a tokenizer.

    Sets pad_token to eos_token if not already set (common for causal LMs).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token: '%s'", tokenizer.pad_token)

    tokenizer.padding_side = padding_side

    if max_seq_length is not None:
        tokenizer.model_max_length = max_seq_length

    logger.info(
        "Tokenizer loaded: vocab_size=%d, pad_token='%s', padding_side='%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
        tokenizer.padding_side,
    )
    return tokenizer


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def build_lora_config(
    cfg: dict,
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """Build a LoRA config from a config dict."""
    lora_cfg = cfg.get("lora", {})

    # Map task type string to TaskType enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
    }
    tt = task_type_map.get(task_type, TaskType.CAUSAL_LM)

    config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=tt,
    )
    logger.info("LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s",
                config.r, config.lora_alpha, config.lora_dropout,
                config.target_modules)
    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_causal_lm(
    model_name: str,
    cfg: dict,
    quantize_4bit: bool = False,
) -> PreTrainedModel:
    """
    Load a causal language model, optionally with 4-bit quantization.

    Args:
        model_name: HuggingFace model identifier or local path.
        cfg: Full config dict (used for dtype detection).
        quantize_4bit: If True, load in 4-bit using bitsandbytes.

    Returns:
        The loaded model.
    """
    dtype = get_dtype(cfg)

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if quantize_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
            logger.info("4-bit quantization enabled (QLoRA mode)")
        except Exception as e:
            logger.warning("bitsandbytes not available, loading without quantization: %s", e)

    if not quantize_4bit and torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Prepare for kbit training if quantized
    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %s (%.1fM params, %.1fM trainable)",
        model_name, param_count / 1e6, trainable / 1e6,
    )
    return model


def load_reward_model(
    model_name: str,
    cfg: dict,
    num_labels: int = 1,
    quantize_4bit: bool = False,
) -> PreTrainedModel:
    """
    Load a model for sequence classification (reward modeling).

    Uses AutoModelForSequenceClassification with num_labels=1 for scalar reward.
    """
    dtype = get_dtype(cfg)

    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "num_labels": num_labels,
    }

    if quantize_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
        except Exception:
            logger.warning("bitsandbytes not available, loading without quantization")

    if not quantize_4bit and torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)

    if quantize_4bit:
        model = prepare_model_for_kbit_training(model)

    # Ensure pad token config matches
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Reward model loaded: %s (num_labels=%d)", model_name, num_labels)
    return model


# ---------------------------------------------------------------------------
# LoRA adapter application
# ---------------------------------------------------------------------------

def apply_lora(model: PreTrainedModel, lora_config: LoraConfig) -> PreTrainedModel:
    """Apply LoRA adapters to a model."""
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_peft_model(
    base_model_name: str,
    adapter_path: str | Path,
    cfg: dict,
    is_reward_model: bool = False,
) -> PreTrainedModel:
    """
    Load a base model and merge a saved PEFT adapter on top.

    Args:
        base_model_name: HuggingFace identifier for the base model.
        adapter_path: Path to saved LoRA adapter weights.
        cfg: Full config dict.
        is_reward_model: If True, load as sequence classification model.
    """
    if is_reward_model:
        base = load_reward_model(base_model_name, cfg)
    else:
        base = load_causal_lm(base_model_name, cfg)

    model = PeftModel.from_pretrained(base, str(adapter_path))
    logger.info("PEFT adapter loaded from %s", adapter_path)
    return model


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str | Path,
) -> None:
    """Save model (or adapter) and tokenizer to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model and tokenizer saved to %s", output_dir)
