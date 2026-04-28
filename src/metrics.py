"""
Metric computation functions.

Provides pairwise accuracy, reward gap analysis, and custom metrics
for reward model evaluation and overall pipeline assessment.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


def pairwise_preference_accuracy(
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> dict:
    """
    Compute pairwise preference accuracy for a reward model.

    The reward model is correct when it assigns a higher score to the
    chosen response than to the rejected response.

    Args:
        chosen_scores: Reward scores for chosen responses.
        rejected_scores: Reward scores for rejected responses.

    Returns:
        Dict with accuracy, total pairs, and score gap statistics.
    """
    assert len(chosen_scores) == len(rejected_scores)

    chosen = np.array(chosen_scores)
    rejected = np.array(rejected_scores)
    gaps = chosen - rejected

    correct = (chosen > rejected).sum()
    total = len(chosen)

    return {
        "pairwise_accuracy": correct / total,
        "total_pairs": total,
        "correct_pairs": int(correct),
        "mean_score_gap": float(gaps.mean()),
        "median_score_gap": float(np.median(gaps)),
        "std_score_gap": float(gaps.std()),
        "pct_positive_gap": float((gaps > 0).mean()),
    }


def reward_score_distribution(
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> pd.DataFrame:
    """
    Create a DataFrame comparing chosen vs rejected score distributions.

    Useful for plotting and analysis.
    """
    df = pd.DataFrame({
        "chosen_score": chosen_scores,
        "rejected_score": rejected_scores,
        "gap": np.array(chosen_scores) - np.array(rejected_scores),
    })
    return df


def compute_reward_model_metrics(
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> dict:
    """
    Compute comprehensive reward model evaluation metrics.

    Combines pairwise accuracy with distribution analysis.
    """
    metrics = pairwise_preference_accuracy(chosen_scores, rejected_scores)

    # Additional statistics
    all_scores = list(chosen_scores) + list(rejected_scores)
    metrics["overall_score_mean"] = float(np.mean(all_scores))
    metrics["overall_score_std"] = float(np.std(all_scores))
    metrics["chosen_score_mean"] = float(np.mean(chosen_scores))
    metrics["rejected_score_mean"] = float(np.mean(rejected_scores))

    return metrics


def human_eval_agreement(
    annotations: pd.DataFrame,
    model_a_col: str = "model_a",
    model_b_col: str = "model_b",
    preference_col: str = "preference",
) -> dict:
    """
    Compute inter-annotator agreement and win rates from human evaluation.

    Args:
        annotations: DataFrame with human preference labels.
        model_a_col: Column name for model A identifier.
        model_b_col: Column name for model B identifier.
        preference_col: Column with values 'a', 'b', or 'tie'.

    Returns:
        Dict with win rates and agreement statistics.
    """
    prefs = annotations[preference_col].str.lower()
    total = len(prefs)

    wins_a = (prefs == "a").sum()
    wins_b = (prefs == "b").sum()
    ties = (prefs == "tie").sum()

    return {
        "win_rate_a": wins_a / total,
        "win_rate_b": wins_b / total,
        "tie_rate": ties / total,
        "total_comparisons": total,
        "wins_a": int(wins_a),
        "wins_b": int(wins_b),
        "ties": int(ties),
    }


def compute_diversity_metrics(responses: list[str]) -> dict:
    """
    Compute response diversity metrics.

    Measures:
    - Unique n-gram ratios (unigram, bigram)
    - Unique response ratio (based on first N characters)
    """
    if not responses:
        return {"diversity_error": "no responses provided"}

    # Unique unigrams and bigrams
    all_unigrams = []
    all_bigrams = []
    for r in responses:
        words = r.lower().split()
        all_unigrams.extend(words)
        all_bigrams.extend(zip(words[:-1], words[1:]))

    metrics = {
        "total_responses": len(responses),
        "unique_unigram_ratio": len(set(all_unigrams)) / max(len(all_unigrams), 1),
        "unique_bigram_ratio": len(set(all_bigrams)) / max(len(all_bigrams), 1),
        "unique_response_ratio": len(set(r[:100] for r in responses)) / len(responses),
        "total_unigrams": len(all_unigrams),
        "total_unique_unigrams": len(set(all_unigrams)),
    }

    return metrics


def length_reward_correlation(
    responses: list[str],
    scores: list[float],
    length_unit: str = "words",
) -> dict:
    """
    Compute correlation between response length and reward score.

    Useful for detecting verbosity bias / reward hacking.
    """
    if length_unit == "words":
        lengths = [len(r.split()) for r in responses]
    else:
        lengths = [len(r) for r in responses]

    lengths_arr = np.array(lengths)
    scores_arr = np.array(scores)

    if len(lengths_arr) < 2:
        return {"error": "need at least 2 samples"}

    corr = np.corrcoef(lengths_arr, scores_arr)[0, 1]

    return {
        "length_reward_correlation": float(corr),
        "mean_length": float(lengths_arr.mean()),
        "mean_reward": float(scores_arr.mean()),
        "length_unit": length_unit,
    }
