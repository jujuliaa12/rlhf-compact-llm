"""
Evaluation pipeline utilities.

Provides functions for automatic evaluation, model comparison,
verbosity analysis, and reward hacking indicators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Automatic metric computation
# ---------------------------------------------------------------------------

def compute_response_lengths(responses: list[str], unit: str = "chars") -> dict:
    """
    Compute response length statistics.

    Args:
        responses: List of response strings.
        unit: 'chars' or 'words'.

    Returns:
        Dict with mean, median, std, min, max lengths.
    """
    if unit == "words":
        lengths = [len(r.split()) for r in responses]
    else:
        lengths = [len(r) for r in responses]

    arr = np.array(lengths)
    return {
        f"length_mean_{unit}": float(arr.mean()),
        f"length_median_{unit}": float(np.median(arr)),
        f"length_std_{unit}": float(arr.std()),
        f"length_min_{unit}": int(arr.min()),
        f"length_max_{unit}": int(arr.max()),
    }


def compute_win_rate(
    scores_a: list[float],
    scores_b: list[float],
    label_a: str = "model_a",
    label_b: str = "model_b",
) -> dict:
    """
    Compute pairwise win rate between two models based on reward scores.

    Args:
        scores_a: Reward scores for model A responses.
        scores_b: Reward scores for model B responses.
        label_a: Name for model A.
        label_b: Name for model B.

    Returns:
        Dict with win_rate_a, win_rate_b, tie_rate, total.
    """
    assert len(scores_a) == len(scores_b), "Score lists must be same length"

    wins_a = sum(1 for a, b in zip(scores_a, scores_b) if a > b)
    wins_b = sum(1 for a, b in zip(scores_a, scores_b) if b > a)
    ties = sum(1 for a, b in zip(scores_a, scores_b) if a == b)
    total = len(scores_a)

    return {
        f"win_rate_{label_a}": wins_a / total,
        f"win_rate_{label_b}": wins_b / total,
        "tie_rate": ties / total,
        "total": total,
    }


def evaluate_model_outputs(
    comparison_df: pd.DataFrame,
    reward_scores: Optional[dict[str, list[float]]] = None,
) -> pd.DataFrame:
    """
    Evaluate model outputs from a comparison table.

    Args:
        comparison_df: DataFrame with columns: prompt, model_name, response.
        reward_scores: Optional dict mapping model_name -> list of reward scores.

    Returns:
        Summary DataFrame with metrics per model.
    """
    summaries = []

    for model_name, group in comparison_df.groupby("model_name"):
        responses = group["response"].tolist()
        metrics = {
            "model_name": model_name,
            "num_responses": len(responses),
        }

        # Length metrics
        metrics.update(compute_response_lengths(responses, unit="chars"))
        metrics.update(compute_response_lengths(responses, unit="words"))

        # Reward scores if available
        if reward_scores and model_name in reward_scores:
            scores = reward_scores[model_name]
            metrics["reward_mean"] = float(np.mean(scores))
            metrics["reward_std"] = float(np.std(scores))
            metrics["reward_median"] = float(np.median(scores))

        summaries.append(metrics)

    return pd.DataFrame(summaries)


# ---------------------------------------------------------------------------
# Verbosity bias analysis
# ---------------------------------------------------------------------------

def analyze_verbosity_bias(
    sft_responses: list[str],
    ppo_responses: list[str],
    sft_scores: Optional[list[float]] = None,
    ppo_scores: Optional[list[float]] = None,
) -> dict:
    """
    Analyze verbosity bias between SFT and PPO models.

    Checks whether PPO-aligned model produces longer responses and
    whether length correlates with reward scores.

    Returns:
        Dict with verbosity analysis metrics.
    """
    sft_lens = np.array([len(r.split()) for r in sft_responses])
    ppo_lens = np.array([len(r.split()) for r in ppo_responses])

    analysis = {
        "sft_mean_length_words": float(sft_lens.mean()),
        "ppo_mean_length_words": float(ppo_lens.mean()),
        "length_increase_pct": float(
            (ppo_lens.mean() - sft_lens.mean()) / max(sft_lens.mean(), 1) * 100
        ),
        "sft_median_length_words": float(np.median(sft_lens)),
        "ppo_median_length_words": float(np.median(ppo_lens)),
    }

    # Correlation between length and reward if scores provided
    if ppo_scores is not None:
        ppo_scores_arr = np.array(ppo_scores)
        if len(ppo_scores_arr) == len(ppo_lens) and len(ppo_lens) > 1:
            corr = np.corrcoef(ppo_lens, ppo_scores_arr)[0, 1]
            analysis["ppo_length_reward_correlation"] = float(corr)

    if sft_scores is not None:
        sft_scores_arr = np.array(sft_scores)
        if len(sft_scores_arr) == len(sft_lens) and len(sft_lens) > 1:
            corr = np.corrcoef(sft_lens, sft_scores_arr)[0, 1]
            analysis["sft_length_reward_correlation"] = float(corr)

    return analysis


# ---------------------------------------------------------------------------
# Reward hacking indicators
# ---------------------------------------------------------------------------

def analyze_reward_hacking(
    ppo_log_df: pd.DataFrame,
    ppo_responses: Optional[list[str]] = None,
) -> dict:
    """
    Analyze indicators of reward hacking from PPO training logs.

    Checks for:
    - Reward increasing while response diversity decreases
    - Response length inflating disproportionately
    - Repetitive patterns in generated text

    Args:
        ppo_log_df: DataFrame from PPO training logs with columns:
                    step, mean_reward, response_length_mean, kl_divergence.
        ppo_responses: Optional list of final PPO responses for text analysis.

    Returns:
        Dict with reward hacking indicator metrics.
    """
    indicators = {}

    if "mean_reward" in ppo_log_df.columns and len(ppo_log_df) > 1:
        rewards = ppo_log_df["mean_reward"].values
        indicators["reward_start"] = float(rewards[0])
        indicators["reward_end"] = float(rewards[-1])
        indicators["reward_increase"] = float(rewards[-1] - rewards[0])

        # Check for monotonic reward increase (potential hacking)
        diffs = np.diff(rewards)
        indicators["reward_monotonic_pct"] = float((diffs > 0).mean())

    if "response_length_mean" in ppo_log_df.columns and len(ppo_log_df) > 1:
        lengths = ppo_log_df["response_length_mean"].values
        indicators["length_start"] = float(lengths[0])
        indicators["length_end"] = float(lengths[-1])
        indicators["length_increase_pct"] = float(
            (lengths[-1] - lengths[0]) / max(lengths[0], 1) * 100
        )

    if "kl_divergence" in ppo_log_df.columns and len(ppo_log_df) > 1:
        kl = ppo_log_df["kl_divergence"].dropna().values
        if len(kl) > 0:
            indicators["kl_max"] = float(kl.max())
            indicators["kl_mean"] = float(kl.mean())
            indicators["kl_final"] = float(kl[-1])

    # Text-level analysis if responses provided
    if ppo_responses:
        indicators["num_responses_analyzed"] = len(ppo_responses)

        # Check for repetitive disclaimers or over-refusal
        refusal_markers = [
            "i cannot", "i can't", "i'm not able", "as an ai",
            "i apologize", "i'm sorry, but",
        ]
        refusal_count = sum(
            1 for r in ppo_responses
            if any(marker in r.lower() for marker in refusal_markers)
        )
        indicators["over_refusal_rate"] = refusal_count / max(len(ppo_responses), 1)

        # Response diversity (unique first 50 chars)
        unique_starts = len(set(r[:50] for r in ppo_responses))
        indicators["response_diversity"] = unique_starts / max(len(ppo_responses), 1)

    return indicators


# ---------------------------------------------------------------------------
# Optional BLEU/ROUGE (require references)
# ---------------------------------------------------------------------------

def compute_bleu_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """
    Compute BLEU and ROUGE scores if references are available.

    Requires: evaluate, rouge-score, nltk packages.
    Returns empty dict if packages not available or computation fails.
    """
    metrics = {}

    try:
        import evaluate

        # ROUGE
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(
            predictions=predictions, references=references
        )
        metrics.update({f"rouge_{k}": v for k, v in rouge_result.items()})

        # BLEU (sentence-level average)
        bleu = evaluate.load("bleu")
        bleu_result = bleu.compute(
            predictions=predictions, references=[[r] for r in references]
        )
        metrics["bleu"] = bleu_result.get("bleu", 0.0)

    except Exception as e:
        logger.warning("Could not compute BLEU/ROUGE: %s", e)

    return metrics


# ---------------------------------------------------------------------------
# Save evaluation results
# ---------------------------------------------------------------------------

def save_evaluation_summary(
    summary: dict | pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save evaluation summary to CSV or JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(summary, pd.DataFrame):
        summary.to_csv(output_path, index=False)
    elif isinstance(summary, dict):
        if output_path.suffix == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        else:
            pd.DataFrame([summary]).to_csv(output_path, index=False)

    logger.info("Evaluation summary saved to %s", output_path)
