"""
Text preprocessing and format conversion utilities.

Handles conversion of raw preference datasets into the standardised
prompt / chosen / rejected format used by the reward model and PPO stages.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HH-RLHF parsing
# ---------------------------------------------------------------------------

ASSISTANT_MARKER = "\n\nAssistant:"
HUMAN_MARKER = "\n\nHuman:"


def _find_common_prefix_boundary(chosen: str, rejected: str) -> int:
    """
    Find the byte offset where the chosen and rejected texts first diverge.

    In HH-RLHF the two texts share an identical conversation prefix up to
    the point where the assistant gives a different answer.  We find the
    length of that shared prefix.
    """
    max_len = min(len(chosen), len(rejected))
    boundary = 0
    for i in range(max_len):
        if chosen[i] != rejected[i]:
            break
        boundary = i + 1
    return boundary


def extract_hh_rlhf_triple(chosen_text: str, rejected_text: str) -> tuple[str, str, str]:
    """
    Extract (prompt, chosen_response, rejected_response) from an HH-RLHF
    chosen/rejected pair.

    HH-RLHF conversations are stored as strings with ``\\n\\nHuman:`` and
    ``\\n\\nAssistant:`` delimiters.  The chosen and rejected texts share a
    common conversational prefix and diverge at exactly one Assistant turn.
    After that divergence point each branch may continue for additional
    turns independently.

    Strategy
    --------
    1.  Find the character-level common prefix of the two texts.
    2.  Back-track the prefix to the last ``\\n\\nAssistant:`` marker that
        starts at or before the divergence point — this is where the
        preferred / rejected answers begin.
    3.  Everything *before* that marker is the **prompt** (ends with the
        final Human turn before divergence).
    4.  Everything *after* that marker in the chosen text (up to the next
        ``\\n\\nHuman:`` or end-of-string) is the **chosen** response.
    5.  Same logic applied to the rejected text gives the **rejected**
        response.

    Returns
    -------
    (prompt, chosen_response, rejected_response)
    All strings are stripped.  Returns ("", "", "") on parse failure.
    """
    # Step 1: find where the two texts diverge
    prefix_len = _find_common_prefix_boundary(chosen_text, rejected_text)

    # Step 2: back-track to the last \n\nAssistant: marker that starts
    # within the common prefix.  The divergence always happens inside
    # (or right after) an Assistant turn, so the marker must be at or
    # before prefix_len.
    common = chosen_text[:prefix_len]
    marker_pos = common.rfind(ASSISTANT_MARKER)
    if marker_pos == -1:
        return "", "", ""

    # Step 3: prompt = everything before that marker
    prompt = chosen_text[:marker_pos].strip()

    # Step 4 & 5: response = everything after the marker up to the next
    # \n\nHuman: (if any) in each text.  We take only the first
    # assistant segment so we don't bleed follow-up turns into the
    # response.
    response_start = marker_pos + len(ASSISTANT_MARKER)

    chosen_response = _extract_first_assistant_segment(chosen_text[response_start:])
    rejected_response = _extract_first_assistant_segment(rejected_text[response_start:])

    return prompt, chosen_response, rejected_response


def _extract_first_assistant_segment(text: str) -> str:
    """
    Return the text up to the next ``\\n\\nHuman:`` marker (exclusive),
    which represents a single assistant turn.  If there is no subsequent
    Human marker the whole remaining text is the response.
    """
    next_human = text.find(HUMAN_MARKER)
    if next_human != -1:
        return text[:next_human].strip()
    return text.strip()


def debug_hh_parse(chosen_text: str, rejected_text: str, idx: int = 0) -> None:
    """
    Print detailed parsing diagnostics for one HH-RLHF sample.

    Useful for verifying that prompt / chosen / rejected are aligned.
    """
    prompt, chosen, rejected = extract_hh_rlhf_triple(chosen_text, rejected_text)

    # Count turns in the prompt
    num_human = prompt.count(HUMAN_MARKER.strip()) + (1 if prompt.lstrip().startswith("Human:") else 0)
    num_assistant = prompt.count(ASSISTANT_MARKER.strip())

    logger.info(
        "=== DEBUG HH-RLHF PARSE  (sample %d) ===\n"
        "  Turns in prompt  : %d Human, %d Assistant\n"
        "  Prompt (last 200): ...%s\n"
        "  Chosen  (full)   : %s\n"
        "  Rejected (full)  : %s\n"
        "  Prompt ends with Human turn: %s\n"
        "=========================================",
        idx,
        num_human, num_assistant,
        prompt[-200:],
        chosen[:300],
        rejected[:300],
        str(prompt.rstrip().rsplit("\n\n", 1)[-1][:30]),
    )


def process_hh_rlhf_to_preference(ds: Dataset) -> Dataset:
    """
    Convert raw HH-RLHF dataset to prompt/chosen/rejected format.

    Each sample gets:
        - prompt: shared conversation context up to and including the
          last Human turn before the divergence point
        - chosen: the preferred assistant response at the divergence turn
        - rejected: the rejected assistant response at the divergence turn

    The parsing works by aligning the chosen and rejected texts to find
    their common prefix, then extracting each branch's response from the
    divergence point.  This correctly handles multi-turn conversations
    where the divergence may occur at any assistant turn.
    """
    records = []
    skipped_empty = 0

    for idx, sample in enumerate(ds):
        chosen_text = sample.get("chosen", "")
        rejected_text = sample.get("rejected", "")

        prompt, chosen, rejected = extract_hh_rlhf_triple(chosen_text, rejected_text)

        if not prompt or not chosen or not rejected:
            skipped_empty += 1
            continue

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    logger.info(
        "Processed %d samples  |  skipped: %d empty/unparseable",
        len(records), skipped_empty,
    )
    return Dataset.from_pandas(pd.DataFrame(records))


# ---------------------------------------------------------------------------
# UltraFeedback parsing
# ---------------------------------------------------------------------------

def process_ultrafeedback_to_preference(ds: Dataset) -> Dataset:
    """
    Convert UltraFeedback binarized dataset to prompt/chosen/rejected format.

    UltraFeedback stores 'chosen' and 'rejected' as lists of message dicts
    with 'role' and 'content' keys.
    """
    records = []
    skipped = 0

    for sample in ds:
        try:
            chosen_msgs = sample["chosen"]
            rejected_msgs = sample["rejected"]

            # Extract prompt from the first user message
            prompt = ""
            chosen_response = ""
            rejected_response = ""

            for msg in chosen_msgs:
                if msg["role"] == "user":
                    prompt = msg["content"]
                elif msg["role"] == "assistant":
                    chosen_response = msg["content"]

            for msg in rejected_msgs:
                if msg["role"] == "assistant":
                    rejected_response = msg["content"]

            if not prompt or not chosen_response or not rejected_response:
                skipped += 1
                continue

            records.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            })
        except (KeyError, TypeError, IndexError):
            skipped += 1
            continue

    logger.info(
        "Processed %d samples, skipped %d invalid", len(records), skipped
    )
    return Dataset.from_pandas(pd.DataFrame(records))


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Apply basic text cleaning."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove null bytes
    text = text.replace("\x00", "")
    return text


def clean_dataset(ds: Dataset) -> Dataset:
    """Clean all text fields in a preference dataset."""

    def _clean(sample):
        return {
            "prompt": clean_text(sample.get("prompt", "")),
            "chosen": clean_text(sample.get("chosen", "")),
            "rejected": clean_text(sample.get("rejected", "")),
        }

    return ds.map(_clean, desc="Cleaning text")


def filter_empty_rows(ds: Dataset) -> Dataset:
    """Remove rows where any of prompt/chosen/rejected is empty."""
    before = len(ds)
    ds = ds.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"]),
        desc="Filtering empty rows",
    )
    logger.info("Filtered %d -> %d samples", before, len(ds))
    return ds


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_length_stats(ds: Dataset) -> pd.DataFrame:
    """
    Compute length statistics for prompt, chosen, and rejected fields.

    Returns a DataFrame with columns: field, count, mean_len, median_len,
    min_len, max_len, std_len (lengths in characters).
    """
    stats = []
    for field in ["prompt", "chosen", "rejected"]:
        lengths = [len(str(x)) for x in ds[field]]
        stats.append({
            "field": field,
            "count": len(lengths),
            "mean_len": pd.Series(lengths).mean(),
            "median_len": pd.Series(lengths).median(),
            "min_len": min(lengths),
            "max_len": max(lengths),
            "std_len": pd.Series(lengths).std(),
        })
    return pd.DataFrame(stats)


def compute_token_length_stats(
    ds: Dataset, tokenizer, fields: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute token-level length statistics using a tokenizer.

    Returns a DataFrame with token count statistics per field.
    """
    if fields is None:
        fields = ["prompt", "chosen", "rejected"]

    stats = []
    for field in fields:
        texts = [str(x) for x in ds[field]]
        token_counts = [
            len(tokenizer.encode(t, add_special_tokens=False)) for t in texts
        ]
        series = pd.Series(token_counts)
        stats.append({
            "field": field,
            "count": len(token_counts),
            "mean_tokens": series.mean(),
            "median_tokens": series.median(),
            "min_tokens": series.min(),
            "max_tokens": series.max(),
            "std_tokens": series.std(),
        })
    return pd.DataFrame(stats)


def get_length_series(ds: Dataset) -> pd.DataFrame:
    """
    Return a DataFrame with per-sample character lengths for plotting.

    Columns: prompt_len, chosen_len, rejected_len
    """
    return pd.DataFrame({
        "prompt_len": [len(str(x)) for x in ds["prompt"]],
        "chosen_len": [len(str(x)) for x in ds["chosen"]],
        "rejected_len": [len(str(x)) for x in ds["rejected"]],
    })


# ---------------------------------------------------------------------------
# SFT data preparation
# ---------------------------------------------------------------------------

def prepare_sft_dataset(ds: Dataset, prompt_template: Optional[str] = None) -> Dataset:
    """
    Prepare a preference dataset for SFT by combining prompt + chosen response.

    The SFT target is the chosen response from the preference dataset,
    used as a lightweight instruction-following proxy.

    Args:
        ds: Dataset with 'prompt' and 'chosen' columns.
        prompt_template: Optional template string with {prompt} and {response}
                         placeholders. If None, uses a simple default.

    Returns:
        Dataset with a 'text' column for SFT training.
    """
    if prompt_template is None:
        prompt_template = (
            "### Human:\n{prompt}\n\n### Assistant:\n{response}"
        )

    def _format(sample):
        return {
            "text": prompt_template.format(
                prompt=sample["prompt"], response=sample["chosen"]
            )
        }

    ds = ds.map(_format, desc="Formatting for SFT")

    # Drop all columns except 'text' — SFTTrainer only needs the text
    # field and will fail if extra string columns are present.
    cols_to_remove = [c for c in ds.column_names if c != "text"]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    return ds


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def split_dataset(
    ds: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, Dataset]:
    """
    Split a dataset into train/val/test.

    Returns dict with keys 'train', 'validation', 'test'.
    """
    # First split off test
    split1 = ds.train_test_split(test_size=test_ratio, seed=seed)
    # Then split remaining into train and val
    remaining = split1["train"]
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    split2 = remaining.train_test_split(test_size=adjusted_val_ratio, seed=seed)

    result = {
        "train": split2["train"],
        "validation": split2["test"],
        "test": split1["test"],
    }
    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(result["train"]),
        len(result["validation"]),
        len(result["test"]),
    )
    return result
