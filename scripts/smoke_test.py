#!/usr/bin/env python
"""
Full-pipeline integration smoke test.

Runs every stage of the RLHF pipeline on a tiny data subset to verify
that preprocessing, SFT, reward modelling, and PPO all execute without
errors and produce the expected artefacts.

All outputs go to  outputs/debug/  so that real experiment results in
outputs/ are never polluted.  The script can optionally delete this
directory on completion (--cleanup flag).

Usage
-----
    # from project root
    python scripts/smoke_test.py                 # full 4-stage smoke test
    python scripts/smoke_test.py --cleanup       # … and delete outputs/debug afterwards
    python scripts/smoke_test.py --stages 1 2    # run only stages 1 & 2
    python scripts/smoke_test.py --model Qwen/Qwen2.5-0.5B   # override model
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `from src.…` imports work
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging — single format for both console and a debug log file
# ---------------------------------------------------------------------------
DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DEBUG_DIR / "smoke_test.log", mode="w"),
    ],
)
logger = logging.getLogger("smoke_test")

# ---------------------------------------------------------------------------
# Paths used throughout the test (all under outputs/debug/)
# ---------------------------------------------------------------------------
DATA_PROC_DIR     = DEBUG_DIR / "data_processed"
SFT_MODEL_DIR     = DEBUG_DIR / "sft_model"
REWARD_MODEL_DIR  = DEBUG_DIR / "reward_model"
PPO_MODEL_DIR     = DEBUG_DIR / "ppo_model"
LOG_DIR           = DEBUG_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tunables — kept deliberately small for a fast smoke test
# ---------------------------------------------------------------------------
NUM_SAMPLES       = 50        # train split
NUM_VAL_SAMPLES   = 20        # val / test split
MAX_SEQ_LEN       = 256       # token limit (halved vs production)
SFT_EPOCHS        = 1
SFT_BATCH          = 2
SFT_GRAD_ACCUM     = 2
REWARD_EPOCHS     = 1
REWARD_BATCH       = 2
REWARD_GRAD_ACCUM  = 2
PPO_STEPS          = 5
PPO_BATCH          = 4        # must be >= mini_batch_size
PPO_MINI_BATCH     = 2
PPO_GEN_TOKENS     = 32       # short generations to save time

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

# LoRA settings — identical across stages for consistency
LORA_R             = 8        # smaller rank for speed
LORA_ALPHA         = 16
LORA_DROPOUT       = 0.05
LORA_TARGETS       = ["q_proj", "v_proj"]


# ===================================================================== #
#  Helpers                                                                #
# ===================================================================== #

def _banner(text: str) -> None:
    rule = "=" * 64
    logger.info("\n%s\n  %s\n%s", rule, text, rule)


def _elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def _detect_device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info("GPU detected: %s (%.1f GB)", name, mem)
        return "cuda"
    logger.info("No GPU detected — running on CPU (will be slow)")
    return "cpu"


# ===================================================================== #
#  Stage 1 — Data preprocessing                                          #
# ===================================================================== #

def stage_1_preprocessing():
    """Load HH-RLHF, parse to prompt/chosen/rejected, save to disk."""
    _banner("STAGE 1: Data Preprocessing")
    t0 = time.time()

    from src.data_utils import load_hh_rlhf, save_dataset_to_disk, set_seed
    from src.preprocessing import (
        clean_dataset,
        compute_length_stats,
        debug_hh_parse,
        filter_empty_rows,
        prepare_sft_dataset,
        process_hh_rlhf_to_preference,
    )

    set_seed(42)

    # Load raw data — small subset
    raw_train = load_hh_rlhf(split="train", max_samples=NUM_SAMPLES)
    raw_val   = load_hh_rlhf(split="test",  max_samples=NUM_VAL_SAMPLES)

    logger.info("Raw train columns: %s  |  rows: %d", raw_train.column_names, len(raw_train))
    logger.info("Raw val   columns: %s  |  rows: %d", raw_val.column_names, len(raw_val))

    # Show one raw example (both chosen and rejected texts)
    sample = raw_train[0]
    logger.info(
        "--- RAW SAMPLE 0 ---\n"
        "  chosen  (first 400 chars): %s\n"
        "  rejected (first 400 chars): %s",
        str(sample["chosen"])[:400],
        str(sample["rejected"])[:400],
    )

    # Run detailed parse diagnostics on 3 raw samples before bulk processing
    logger.info("--- Detailed parse diagnostics on first 3 raw samples ---")
    for i in range(min(3, len(raw_train))):
        debug_hh_parse(raw_train[i]["chosen"], raw_train[i]["rejected"], idx=i)

    # Convert to preference format
    pref_train = process_hh_rlhf_to_preference(raw_train)
    pref_val   = process_hh_rlhf_to_preference(raw_val)

    pref_train = clean_dataset(pref_train)
    pref_val   = clean_dataset(pref_val)
    pref_train = filter_empty_rows(pref_train)
    pref_val   = filter_empty_rows(pref_val)

    # Show 3 processed examples to verify prompt/chosen/rejected alignment
    num_to_show = min(3, len(pref_train))
    for i in range(num_to_show):
        s = pref_train[i]
        logger.info(
            "--- PROCESSED SAMPLE %d/%d ---\n"
            "  prompt   : %s\n"
            "  chosen   : %s\n"
            "  rejected : %s",
            i + 1, num_to_show,
            s["prompt"][:300], s["chosen"][:300], s["rejected"][:300],
        )

    # Length stats
    stats = compute_length_stats(pref_train)
    logger.info("Length statistics (train):\n%s", stats.to_string(index=False))

    # Save to disk
    save_dataset_to_disk(pref_train, DATA_PROC_DIR / "train")
    save_dataset_to_disk(pref_val,   DATA_PROC_DIR / "val")

    # Also prepare an SFT version (adds 'text' column)
    sft_train = prepare_sft_dataset(pref_train)
    save_dataset_to_disk(sft_train, DATA_PROC_DIR / "sft_train")

    logger.info(
        "Stage 1 PASSED — %d train, %d val samples saved to %s  [%s]",
        len(pref_train), len(pref_val), DATA_PROC_DIR, _elapsed(t0),
    )
    return pref_train, pref_val, sft_train


# ===================================================================== #
#  Stage 2 — SFT Training                                                #
# ===================================================================== #

def stage_2_sft(sft_train_ds, model_name: str, device: str):
    """One-epoch SFT on the tiny dataset, save adapter."""
    _banner("STAGE 2: SFT Training")
    t0 = time.time()

    import torch
    from peft import LoraConfig, TaskType
    from transformers import TrainingArguments
    from trl import SFTTrainer

    from src.model_utils import apply_lora, load_causal_lm, load_tokenizer
    from src.sft_train import save_training_log

    cfg = _base_cfg(device)

    tokenizer = load_tokenizer(model_name, max_seq_length=MAX_SEQ_LEN)
    model     = load_causal_lm(model_name, cfg)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = apply_lora(model, lora_config)

    # Gradient checkpointing + PEFT fix:
    # On CPU, disable gradient checkpointing — it adds complexity for no
    # memory benefit and can cause "does not require grad" errors with PEFT.
    # On GPU, enable it but also call enable_input_require_grads() so the
    # first layer's output retains a grad_fn that PEFT needs.
    use_grad_ckpt = torch.cuda.is_available()
    if use_grad_ckpt:
        model.enable_input_require_grads()
        logger.info(
            "Gradient checkpointing ON (GPU) — "
            "enable_input_require_grads() called for PEFT compatibility"
        )
    else:
        logger.info("Gradient checkpointing OFF (CPU smoke test)")

    model.train()

    use_fp16 = cfg["training"]["fp16"] and torch.cuda.is_available()

    # Debug: verify the dataset is clean before handing it to SFTTrainer
    logger.info(
        "SFT dataset check — columns: %s  |  rows: %d  |  first sample keys: %s",
        sft_train_ds.column_names, len(sft_train_ds), list(sft_train_ds[0].keys()),
    )

    # Debug: verify model training state
    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    any_grad   = any(p.requires_grad for p in model.parameters())
    logger.info(
        "Model state — training=%s  |  total_params=%d  |  trainable=%d  |  any requires_grad=%s",
        model.training, n_total, n_trainable, any_grad,
    )

    training_args = TrainingArguments(
        output_dir=str(SFT_MODEL_DIR),
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=9999,          # don't checkpoint mid-run
        save_total_limit=1,
        fp16=use_fp16,
        gradient_checkpointing=use_grad_ckpt,
        report_to="none",
        seed=42,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_train_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(str(SFT_MODEL_DIR))
    tokenizer.save_pretrained(str(SFT_MODEL_DIR))

    csv_path = LOG_DIR / "sft_training_log.csv"
    save_training_log(trainer.state.log_history, csv_path)

    # Quick verification
    assert csv_path.exists(), "SFT training log not written"
    assert (SFT_MODEL_DIR / "adapter_config.json").exists(), "LoRA adapter not saved"

    logger.info(
        "Stage 2 PASSED — SFT adapter saved to %s, log at %s  [%s]",
        SFT_MODEL_DIR, csv_path, _elapsed(t0),
    )
    return model, tokenizer


# ===================================================================== #
#  Stage 3 — Reward Model Training                                       #
# ===================================================================== #

def stage_3_reward(pref_train, pref_val, model_name: str, device: str):
    """One-epoch reward model training on tiny preference pairs."""
    _banner("STAGE 3: Reward Model Training")
    t0 = time.time()

    import pandas as pd
    import torch
    from peft import LoraConfig, TaskType
    from transformers import TrainingArguments
    from trl import RewardTrainer

    from src.metrics import pairwise_preference_accuracy
    from src.model_utils import apply_lora, load_reward_model, load_tokenizer
    from src.reward_train import compute_reward_scores, prepare_reward_dataset

    cfg = _base_cfg(device)

    tokenizer = load_tokenizer(model_name, max_seq_length=MAX_SEQ_LEN)
    model     = load_reward_model(model_name, cfg, num_labels=1)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type=TaskType.SEQ_CLS,
    )
    model = apply_lora(model, lora_config)

    # PEFT + gradient checkpointing fix (same pattern as SFT)
    use_grad_ckpt = torch.cuda.is_available()
    if use_grad_ckpt:
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing ON (GPU) — enable_input_require_grads() called")
    else:
        logger.info("Gradient checkpointing OFF (CPU smoke test)")
    model.train()

    # --- Tokenize preference pairs ---
    train_tok = prepare_reward_dataset(pref_train, tokenizer, MAX_SEQ_LEN)
    val_tok   = prepare_reward_dataset(pref_val,   tokenizer, MAX_SEQ_LEN)

    # Debug: inspect tokenized datasets
    # TRL 0.9.6 RewardTrainer expects exactly these four columns:
    #   input_ids_chosen, attention_mask_chosen,
    #   input_ids_rejected, attention_mask_rejected
    # remove_unused_columns MUST be False so the trainer keeps them.
    logger.info(
        "Reward train dataset — columns: %s  |  rows: %d",
        train_tok.column_names, len(train_tok),
    )
    logger.info(
        "Reward val   dataset — columns: %s  |  rows: %d",
        val_tok.column_names, len(val_tok),
    )
    first = train_tok[0]
    logger.info(
        "First tokenized sample keys: %s  |  chosen_ids len: %d  |  rejected_ids len: %d",
        list(first.keys()),
        len(first["input_ids_chosen"]),
        len(first["input_ids_rejected"]),
    )

    # Model state debug
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model state — training=%s  |  trainable_params=%d",
        model.training, n_trainable,
    )

    use_fp16 = cfg["training"]["fp16"] and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(REWARD_MODEL_DIR),
        num_train_epochs=REWARD_EPOCHS,
        per_device_train_batch_size=REWARD_BATCH,
        gradient_accumulation_steps=REWARD_GRAD_ACCUM,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_steps=9999,
        evaluation_strategy="no",     # skip eval during training for speed
        save_steps=9999,
        save_total_limit=1,
        fp16=use_fp16,
        gradient_checkpointing=use_grad_ckpt,
        report_to="none",
        seed=42,
        # MUST be False — RewardTrainer needs the _chosen / _rejected columns
        # which are not in the model's forward() signature.
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(str(REWARD_MODEL_DIR))
    tokenizer.save_pretrained(str(REWARD_MODEL_DIR))

    csv_path = LOG_DIR / "reward_training_log.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(csv_path, index=False)
    logger.info("Training log saved to %s", csv_path)

    # --- Pairwise accuracy on val set ---
    logger.info("Computing pairwise accuracy on validation set...")

    chosen_texts   = [f"{p}\n\n{c}" for p, c in zip(pref_val["prompt"], pref_val["chosen"])]
    rejected_texts = [f"{p}\n\n{r}" for p, r in zip(pref_val["prompt"], pref_val["rejected"])]

    chosen_scores   = compute_reward_scores(model, tokenizer, chosen_texts,   MAX_SEQ_LEN, batch_size=4)
    rejected_scores = compute_reward_scores(model, tokenizer, rejected_texts, MAX_SEQ_LEN, batch_size=4)

    acc_metrics = pairwise_preference_accuracy(chosen_scores, rejected_scores)
    logger.info("  Pairwise accuracy : %.4f", acc_metrics["pairwise_accuracy"])
    logger.info("  Mean score gap    : %.4f", acc_metrics["mean_score_gap"])
    logger.info("  Positive gap %%    : %.1f%%", acc_metrics["pct_positive_gap"] * 100)

    pd.DataFrame([acc_metrics]).to_csv(LOG_DIR / "reward_val_metrics.csv", index=False)

    # --- Standalone inference check on 3 sample texts ---
    logger.info("--- Reward inference spot-check (3 sample texts) ---")
    sample_texts = [
        "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris.",
        "Human: What is the capital of France?\n\nAssistant: I don't know.",
        "Human: Tell me a joke.\n\nAssistant: Why did the chicken cross the road? To get to the other side!",
    ]
    sample_scores = compute_reward_scores(model, tokenizer, sample_texts, MAX_SEQ_LEN, batch_size=4)
    for text, score in zip(sample_texts, sample_scores):
        # Show the assistant part only for readability
        assistant_part = text.split("Assistant: ", 1)[-1][:60]
        logger.info("  score=%.4f  |  %s", score, assistant_part)

    # --- Assertions ---
    assert csv_path.exists(), "Reward training log not written"
    assert (REWARD_MODEL_DIR / "adapter_config.json").exists(), "RM adapter not saved"
    assert len(sample_scores) == 3, "Inference check returned wrong number of scores"

    logger.info(
        "Stage 3 PASSED — reward model saved to %s  [%s]",
        REWARD_MODEL_DIR, _elapsed(t0),
    )
    return model, tokenizer


# ===================================================================== #
#  Stage 4 — PPO Training                                                #
# ===================================================================== #

def stage_4_ppo(pref_train, sft_model_dir: Path, reward_model, rm_tokenizer,
                model_name: str, device: str):
    """Minimal PPO loop: 5 steps, verify logging."""
    _banner("STAGE 4: PPO Training")
    t0 = time.time()

    import pandas as pd
    import torch

    # --- TRL PPO imports (version-sensitive) ---
    try:
        import trl
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
        logger.info("TRL version: %s", trl.__version__)
    except ImportError as e:
        logger.error(
            "TRL PPO import failed. Ensure trl==0.9.6 is installed. Error: %s", e,
        )
        raise

    from src.model_utils import load_tokenizer
    from src.ppo_train import PPOLogger
    from src.reward_train import compute_reward_scores

    tokenizer = load_tokenizer(model_name)

    # ------------------------------------------------------------------ #
    #  Load SFT model + value head  (two-step approach)                   #
    # ------------------------------------------------------------------ #
    # Step 1: Load base causal LM, attach the saved LoRA adapter via PEFT,
    #         so we get a PeftModel.
    # Step 2: Wrap the PeftModel with AutoModelForCausalLMWithValueHead.
    #
    # We avoid passing peft_pretrained_model_name_or_path directly into
    # from_pretrained() because in trl==0.9.6 + transformers==4.41.0 that
    # kwarg leaks into the underlying model __init__ and causes a TypeError.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    logger.info("Loading policy model (two-step: base + LoRA + value head) ...")
    logger.info("  base model  : %s", model_name)
    logger.info("  adapter dir : %s", sft_model_dir)

    # Step 1a: load the base causal LM
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    logger.info("  Base model loaded")

    # Step 1b: attach the SFT LoRA adapter
    base_model = PeftModel.from_pretrained(base_model, str(sft_model_dir))
    logger.info("  SFT LoRA adapter attached")

    # Step 2: wrap with value head for PPO
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    logger.info("  Value head wrapped")

    # Debug: verify adapter + model state
    is_peft = hasattr(policy.pretrained_model, "peft_config")
    logger.info("  PEFT config present: %s", is_peft)

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(
        "  Policy params — total: %d  trainable: %d  training: %s",
        n_total, n_trainable, policy.pretrained_model.training,
    )

    # Verify reward model is ready
    logger.info("  Reward model loaded: %s", reward_model is not None)

    # ------------------------------------------------------------------ #
    #  PPO config                                                         #
    # ------------------------------------------------------------------ #
    ppo_config = PPOConfig(
        learning_rate=1.41e-5,
        batch_size=PPO_BATCH,
        mini_batch_size=PPO_MINI_BATCH,
        ppo_epochs=2,            # fewer inner epochs for speed
        init_kl_coef=0.2,
        target=6.0,
        gamma=1.0,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        seed=42,
        log_with=None,           # no external logger
    )

    logger.info(
        "PPO config — batch=%d  mini_batch=%d  ppo_epochs=%d  "
        "init_kl_coef=%.2f  target_kl=%.1f  lr=%.2e",
        ppo_config.batch_size, ppo_config.mini_batch_size,
        ppo_config.ppo_epochs, ppo_config.init_kl_coef,
        ppo_config.target, ppo_config.learning_rate,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        tokenizer=tokenizer,
    )

    # ------------------------------------------------------------------ #
    #  Prepare prompts and generation settings                            #
    # ------------------------------------------------------------------ #
    prompts = pref_train["prompt"]
    logger.info("PPO prompts available: %d  (will cycle if needed)", len(prompts))

    gen_kwargs = {
        "max_new_tokens": PPO_GEN_TOKENS,
        "min_new_tokens": 4,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    logger.info("Generation settings: %s", gen_kwargs)

    ppo_logger = PPOLogger(LOG_DIR / "ppo_training_log.csv")

    # ------------------------------------------------------------------ #
    #  PPO training loop                                                  #
    # ------------------------------------------------------------------ #
    logger.info("Running %d PPO steps (batch=%d) ...", PPO_STEPS, PPO_BATCH)

    for step in range(PPO_STEPS):
        # Collect a micro-batch of prompts (cycle through available)
        batch_prompts = [
            prompts[i % len(prompts)]
            for i in range(step * PPO_BATCH, (step + 1) * PPO_BATCH)
        ]

        # Tokenize prompts -> list of 1-D tensors
        query_tensors = [
            tokenizer.encode(p, return_tensors="pt", truncation=True,
                             max_length=128).squeeze(0)
            for p in batch_prompts
        ]

        # Generate full sequences (query + response)
        full_output_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, **gen_kwargs
        )

        # TRL generate() with return_prompt=False should return
        # response-only tensors.  If it returns full sequences instead
        # (depends on TRL version), strip the query prefix.
        response_tensors = []
        for i, full in enumerate(full_output_tensors):
            full = full.squeeze()
            q_len = query_tensors[i].shape[0]
            if full.shape[0] > q_len and torch.equal(full[:q_len], query_tensors[i]):
                # Full sequence returned — strip query prefix
                response_tensors.append(full[q_len:])
            else:
                # Already response-only
                response_tensors.append(full)

        # Decode response-only text for reward scoring
        response_texts = [
            tokenizer.decode(r, skip_special_tokens=True)
            for r in response_tensors
        ]

        # Reward scoring — combine prompt + response for the reward model
        combined = [f"{p}\n\n{r}" for p, r in zip(batch_prompts, response_texts)]
        rewards = compute_reward_scores(
            reward_model, rm_tokenizer, combined,
            max_length=MAX_SEQ_LEN, batch_size=PPO_BATCH,
        )
        reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

        # PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        # Extract metrics
        mean_reward = sum(rewards) / len(rewards)
        mean_len = sum(len(r) for r in response_texts) / len(response_texts)

        # KL key varies across TRL versions — try all known variants
        kl = None
        for kl_key in ("ppo/mean_kl", "ppo/kl", "objective/kl",
                        "ppo/mean_non_score_reward"):
            if kl_key in stats:
                kl = stats[kl_key]
                break

        # On first step, dump all stat keys so we can debug key names
        if step == 0:
            logger.info("  PPO stats keys (step 0): %s", sorted(stats.keys()))

        ppo_logger.log(step, {
            "mean_reward": mean_reward,
            "response_length_mean": mean_len,
            "kl_divergence": kl,
        })

        logger.info(
            "  step %d/%d  reward=%.4f  len=%.0f  kl=%s",
            step + 1, PPO_STEPS, mean_reward, mean_len,
            f"{kl:.4f}" if isinstance(kl, (int, float)) else "N/A",
        )

    ppo_logger.save()

    # ------------------------------------------------------------------ #
    #  Save aligned model                                                 #
    # ------------------------------------------------------------------ #
    PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(PPO_MODEL_DIR))
    tokenizer.save_pretrained(str(PPO_MODEL_DIR))

    # ------------------------------------------------------------------ #
    #  Verify PPO log                                                     #
    # ------------------------------------------------------------------ #
    log_csv = LOG_DIR / "ppo_training_log.csv"
    assert log_csv.exists(), "PPO training log not written"
    df = pd.read_csv(log_csv)
    logger.info("PPO log columns: %s", list(df.columns))
    logger.info("PPO log rows   : %d", len(df))
    logger.info("PPO log preview:\n%s", df.to_string(index=False))

    assert "mean_reward" in df.columns, "mean_reward missing from PPO log"
    assert "kl_divergence" in df.columns, "kl_divergence missing from PPO log"
    assert "response_length_mean" in df.columns, "response_length_mean missing from PPO log"
    assert len(df) == PPO_STEPS, f"Expected {PPO_STEPS} rows, got {len(df)}"

    logger.info(
        "Stage 4 PASSED — PPO model saved to %s, %d log rows recorded  [%s]",
        PPO_MODEL_DIR, len(df), _elapsed(t0),
    )


# ===================================================================== #
#  Shared helpers                                                         #
# ===================================================================== #

def _base_cfg(device: str) -> dict:
    """Return a minimal config dict mirroring configs/base.yaml."""
    return {
        "seed": 42,
        "training": {
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
        },
        "lora": {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGETS,
            "bias": "none",
        },
        "device": device,
    }


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(description="RLHF pipeline smoke test")
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete outputs/debug/ after a successful run",
    )
    parser.add_argument(
        "--stages", nargs="*", type=int, default=[1, 2, 3, 4],
        help="Which stages to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    _banner("RLHF Pipeline Smoke Test")
    logger.info("Model       : %s", args.model)
    logger.info("Stages      : %s", args.stages)
    logger.info("Debug dir   : %s", DEBUG_DIR)

    device = _detect_device()

    # Track which stages pass / fail
    results: dict[int, str] = {}
    overall_t0 = time.time()

    # Stage objects passed forward between stages
    pref_train = pref_val = sft_train = None
    sft_model = sft_tokenizer = None
    rm_model = rm_tokenizer = None

    # --- Stage 1 ---
    if 1 in args.stages:
        try:
            pref_train, pref_val, sft_train = stage_1_preprocessing()
            results[1] = "PASSED"
        except Exception:
            logger.error("Stage 1 FAILED:\n%s", traceback.format_exc())
            results[1] = "FAILED"

    # --- Stage 2 ---
    if 2 in args.stages:
        if sft_train is None:
            logger.warning("Stage 2 requires Stage 1 output — loading from disk")
            try:
                from src.data_utils import load_dataset_from_disk
                sft_train = load_dataset_from_disk(DATA_PROC_DIR / "sft_train")
            except Exception:
                logger.error("Cannot load SFT data; run Stage 1 first")
                results[2] = "SKIPPED"
        if sft_train is not None:
            try:
                sft_model, sft_tokenizer = stage_2_sft(sft_train, args.model, device)
                results[2] = "PASSED"
            except Exception:
                logger.error("Stage 2 FAILED:\n%s", traceback.format_exc())
                results[2] = "FAILED"

    # --- Stage 3 ---
    if 3 in args.stages:
        if pref_train is None or pref_val is None:
            logger.warning("Stage 3 requires Stage 1 output — loading from disk")
            try:
                from src.data_utils import load_dataset_from_disk
                pref_train = load_dataset_from_disk(DATA_PROC_DIR / "train")
                pref_val   = load_dataset_from_disk(DATA_PROC_DIR / "val")
            except Exception:
                logger.error("Cannot load preference data; run Stage 1 first")
                results[3] = "SKIPPED"
        if pref_train is not None and pref_val is not None:
            try:
                rm_model, rm_tokenizer = stage_3_reward(pref_train, pref_val, args.model, device)
                results[3] = "PASSED"
            except Exception:
                logger.error("Stage 3 FAILED:\n%s", traceback.format_exc())
                results[3] = "FAILED"

    # --- Stage 4 ---
    if 4 in args.stages:
        can_run = True
        if pref_train is None:
            try:
                from src.data_utils import load_dataset_from_disk
                pref_train = load_dataset_from_disk(DATA_PROC_DIR / "train")
            except Exception:
                logger.error("Cannot load prompts for PPO; run Stage 1 first")
                can_run = False
        if not SFT_MODEL_DIR.exists():
            logger.error("SFT model dir missing; run Stage 2 first")
            can_run = False
        if rm_model is None:
            logger.error("Reward model not available; run Stage 3 first")
            can_run = False
        if not can_run:
            results[4] = "SKIPPED"
        else:
            try:
                stage_4_ppo(
                    pref_train, SFT_MODEL_DIR,
                    rm_model, rm_tokenizer,
                    args.model, device,
                )
                results[4] = "PASSED"
            except Exception:
                logger.error("Stage 4 FAILED:\n%s", traceback.format_exc())
                results[4] = "FAILED"

    # --- Summary ---
    _banner("Smoke Test Summary")
    stage_names = {1: "Preprocessing", 2: "SFT", 3: "Reward Model", 4: "PPO"}
    all_passed = True
    for s in sorted(args.stages):
        status = results.get(s, "NOT RUN")
        logger.info("  Stage %d (%s): %s", s, stage_names[s], status)
        if status != "PASSED":
            all_passed = False

    logger.info("  Total time: %s", _elapsed(overall_t0))

    # List artefacts produced
    _banner("Artefacts in outputs/debug/")
    for p in sorted(DEBUG_DIR.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            rel  = p.relative_to(DEBUG_DIR)
            logger.info("  %s  (%s)", rel, _human_size(size))

    if all_passed:
        logger.info("\nAll stages PASSED.  The pipeline is functional.")
    else:
        logger.warning(
            "\nSome stages did not pass.  Check the log above for details."
        )

    # Optional cleanup
    if args.cleanup and all_passed:
        logger.info("Cleaning up %s ...", DEBUG_DIR)
        shutil.rmtree(DEBUG_DIR, ignore_errors=True)
        logger.info("Cleanup done.")

    # OOM guidance
    if any(v == "FAILED" for v in results.values()):
        logger.info(
            "\n--- OOM Troubleshooting ---\n"
            "If a stage failed with CUDA OOM:\n"
            "  1. Reduce MAX_SEQ_LEN  (currently %d) — try 128\n"
            "  2. Reduce batch sizes  (SFT_BATCH=%d, REWARD_BATCH=%d, PPO_BATCH=%d)\n"
            "  3. Set gradient_accumulation higher to compensate\n"
            "  4. Switch to CPU:  python scripts/smoke_test.py  (fp16 auto-disabled)\n"
            "  5. Use the smaller model:  --model Qwen/Qwen2.5-0.5B\n",
            MAX_SEQ_LEN, SFT_BATCH, REWARD_BATCH, PPO_BATCH,
        )

    sys.exit(0 if all_passed else 1)


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


if __name__ == "__main__":
    main()
