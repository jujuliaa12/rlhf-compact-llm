"""
Generation and inference utilities.

Provides functions to generate text from models at various pipeline stages
(base, SFT, PPO-aligned) for evaluation and comparison.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    batch_size: int = 4,
    prompt_template: Optional[str] = None,
) -> list[str]:
    """
    Generate responses for a list of prompts.

    Args:
        model: Causal language model.
        tokenizer: Corresponding tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        do_sample: Whether to sample (False = greedy).
        batch_size: Number of prompts per batch.
        prompt_template: Optional template with {prompt} placeholder.

    Returns:
        List of generated response strings (prompt removed).
    """
    model.eval()
    device = next(model.parameters()).device
    responses = []

    # Decoder-only models require left-padding for correct generation.
    # Temporarily switch padding side, then restore after generation.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Apply template if provided
        if prompt_template:
            batch_texts = [prompt_template.format(prompt=p) for p in batch_prompts]
        else:
            batch_texts = batch_prompts

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length or 512,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part (remove prompt tokens)
        for j, output_ids in enumerate(outputs):
            input_len = encodings["input_ids"][j].shape[0]
            generated_ids = output_ids[input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())

    # Restore original padding side (right-padding is needed for training)
    tokenizer.padding_side = original_padding_side

    logger.info("Generated %d responses", len(responses))
    return responses


def generate_comparison_table(
    prompts: list[str],
    models: dict[str, tuple[PreTrainedModel, PreTrainedTokenizer]],
    max_new_tokens: int = 128,
    prompt_template: Optional[str] = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """
    Generate responses from multiple models for side-by-side comparison.

    Args:
        prompts: List of prompts.
        models: Dict mapping model_name -> (model, tokenizer).
        max_new_tokens: Max tokens per response.
        prompt_template: Optional template.
        **gen_kwargs: Additional generation kwargs.

    Returns:
        DataFrame with columns: prompt, model_name, response
    """
    records = []

    for model_name, (model, tokenizer) in models.items():
        logger.info("Generating from model: %s", model_name)
        responses = generate_responses(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            prompt_template=prompt_template,
            **gen_kwargs,
        )
        for prompt, response in zip(prompts, responses):
            records.append({
                "prompt": prompt,
                "model_name": model_name,
                "response": response,
            })

    return pd.DataFrame(records)


def save_samples(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "csv",
) -> None:
    """
    Save generated samples to file.

    Args:
        df: DataFrame with generated samples.
        output_path: Path to save file.
        format: 'csv' or 'json'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        records = df.to_dict(orient="records")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info("Samples saved to %s (%d records)", output_path, len(df))


def load_samples(path: str | Path) -> pd.DataFrame:
    """Load previously saved samples."""
    path = Path(path)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    return pd.read_csv(path)
