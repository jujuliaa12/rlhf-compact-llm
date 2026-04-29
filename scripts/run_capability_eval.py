#!/usr/bin/env python
"""
Capability evaluation — quantify the "alignment tax" on knowledge / reasoning.

Runs MMLU (knowledge, 57 subjects) and GSM8K (grade-school math) on the
base model and any of the LoRA adapters trained by this pipeline (SFT,
PPO, DPO). The point is to see how much alignment training degrades pure
capability — a number that is *systematically* under-reported in the
sub-1B RLHF literature.

This script wraps `lm-evaluation-harness` (the de-facto standard).
Install once with::

    pip install -e ".[eval]"

then run::

    python scripts/run_capability_eval.py \
        --tasks mmlu,gsm8k \
        --models base,sft,ppo \
        --num-fewshot 5 \
        --limit 200

Pass ``--limit N`` to evaluate on only the first N examples per task — useful
for quick iteration; remove for full benchmark numbers.

Outputs are written to ``outputs/tables/capability_eval.csv`` and a per-run
JSON dump under ``outputs/logs/capability_eval/``.
"""

from __future__ import annotations

import argparse
import json
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
logger = logging.getLogger("run_capability_eval")


# Maps short model aliases used on the CLI to the (base, adapter_path) pair
# that lm-eval needs as `pretrained=...,peft=...`.
MODEL_REGISTRY: dict[str, dict] = {
    "base": {"base": "Qwen/Qwen2.5-0.5B", "adapter": None},
    "sft":  {"base": "Qwen/Qwen2.5-0.5B", "adapter": "outputs/models/sft_qwen"},
    "rm":   {"base": "Qwen/Qwen2.5-0.5B", "adapter": "outputs/models/reward_model_hh"},
    "ppo":  {"base": "Qwen/Qwen2.5-0.5B", "adapter": "outputs/models/ppo_qwen"},
    "dpo":  {"base": "Qwen/Qwen2.5-0.5B", "adapter": "outputs/models/dpo_qwen"},
}

DEFAULT_TASKS = ["mmlu", "gsm8k"]
DEFAULT_MODELS = ["base", "sft", "ppo"]


def build_model_args(spec: dict, dtype: str | None) -> str:
    """Compose the lm-eval `model_args` string for a HuggingFace model + LoRA."""
    parts = [f"pretrained={spec['base']}", "trust_remote_code=True"]
    # Some lm-eval versions leak `dtype` into the model __init__ instead of
    # routing it to the HF wrapper, which crashes Qwen2ForCausalLM
    # (it only accepts `torch_dtype`). Pass dtype only when the user
    # explicitly opts in.
    if dtype and dtype.lower() not in {"none", "auto", "default", ""}:
        parts.append(f"dtype={dtype}")
    if spec.get("adapter"):
        parts.append(f"peft={spec['adapter']}")
    return ",".join(parts)


def run_one(model_alias: str, task: str, args: argparse.Namespace) -> dict:
    """Run lm-eval-harness on a single (model, task) pair and return a small summary."""
    try:
        from lm_eval import evaluator, utils  # noqa: F401
    except ImportError:
        logger.error(
            "lm-evaluation-harness is not installed. "
            "Install the eval extras first:  pip install -e \".[eval]\""
        )
        sys.exit(1)

    spec = MODEL_REGISTRY[model_alias]
    if spec.get("adapter") and not Path(spec["adapter"]).exists():
        logger.warning(
            "Adapter for %r not found at %s — skipping.", model_alias, spec["adapter"]
        )
        return {"model": model_alias, "task": task, "status": "skipped"}

    model_args = build_model_args(spec, args.dtype)
    logger.info("[%s / %s] model_args=%s", model_alias, task, model_args)

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
    )

    json_dir = Path("outputs/logs/capability_eval")
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{model_alias}__{task}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved raw results to %s", json_path)

    # Extract the headline metric per task. The harness uses different keys
    # per task; `acc` for MMLU, `exact_match` for GSM8K.
    rows = []
    for tname, tres in results.get("results", {}).items():
        headline = (
            tres.get("acc,none")
            or tres.get("exact_match,strict-match")
            or tres.get("exact_match,flexible-extract")
            or tres.get("exact_match")
            or tres.get("acc")
        )
        rows.append({
            "model": model_alias,
            "task": tname,
            "metric": headline,
            "n_samples": tres.get("samples"),
            "version": tres.get("version"),
        })
    return {"model": model_alias, "task": task, "status": "ok", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Capability evaluation harness")
    parser.add_argument(
        "--tasks", type=str, default=",".join(DEFAULT_TASKS),
        help="Comma-separated lm-eval task names (e.g. mmlu,gsm8k,arc_easy)",
    )
    parser.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated model aliases (any of {sorted(MODEL_REGISTRY)})",
    )
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate on first N examples per task (for quick smoke runs)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:0 / cpu — auto-detected if omitted")
    parser.add_argument("--dtype", type=str, default="auto",
                        help="Inference dtype (bfloat16 / float16 / float32 / auto). "
                             "On CPU leave as 'auto' (defaults to float32 which is fastest "
                             "and avoids an lm-eval kwarg-leaking bug).")
    parser.add_argument("--out-csv", type=str,
                        default="outputs/tables/capability_eval.csv")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        parser.error(f"Unknown model alias(es): {unknown}. "
                     f"Known: {sorted(MODEL_REGISTRY)}")

    all_rows: list[dict] = []
    for m in models:
        for t in tasks:
            out = run_one(m, t, args)
            if out.get("status") == "ok":
                all_rows.extend(out["rows"])

    if not all_rows:
        logger.warning("No results to write — every (model, task) was skipped.")
        return

    import pandas as pd
    new_df = pd.DataFrame(all_rows)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with any existing CSV instead of overwriting — multi-stage
    # CPU runs (e.g. base+sft today, ppo+dpo tomorrow) used to clobber
    # earlier rows. New (model, task) pairs win over old ones.
    if out_path.exists():
        old = pd.read_csv(out_path)
        keys = set(zip(new_df["model"], new_df["task"]))
        old = old[~old.apply(lambda r: (r["model"], r["task"]) in keys, axis=1)]
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(out_path, index=False)
    logger.info("Capability eval summary written to %s (%d rows)", out_path, len(df))

    # Pretty-print a small leaderboard for the terminal.
    if {"model", "task", "metric"} <= set(df.columns):
        pivot = df.pivot_table(
            index="model", columns="task", values="metric", aggfunc="first"
        )
        print("\n=== Capability eval ===")
        print(pivot.to_string(float_format=lambda v: f"{v:.3f}" if v is not None else "—"))
        print()


if __name__ == "__main__":
    main()
