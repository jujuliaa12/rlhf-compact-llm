"""Unit tests for src.metrics."""
from __future__ import annotations

import math

import pytest

from src.metrics import (
    compute_diversity_metrics,
    compute_reward_model_metrics,
    human_eval_agreement,
    length_reward_correlation,
    pairwise_preference_accuracy,
)


class TestPairwiseAccuracy:
    def test_perfect_accuracy(self):
        out = pairwise_preference_accuracy([2.0, 3.0, 1.5], [1.0, 0.0, 0.5])
        assert out["pairwise_accuracy"] == 1.0
        assert out["correct_pairs"] == 3
        assert out["total_pairs"] == 3

    def test_zero_accuracy(self):
        out = pairwise_preference_accuracy([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        assert out["pairwise_accuracy"] == 0.0

    def test_mixed(self):
        out = pairwise_preference_accuracy([1.0, 0.0, 2.0, 0.5], [0.0, 1.0, 1.0, 0.0])
        assert out["pairwise_accuracy"] == 0.75
        assert math.isclose(out["mean_score_gap"], (1 - 1 + 1 + 0.5) / 4)

    def test_length_mismatch(self):
        with pytest.raises(AssertionError):
            pairwise_preference_accuracy([1.0, 2.0], [0.0])


class TestRewardModelMetrics:
    def test_includes_distribution_stats(self):
        m = compute_reward_model_metrics([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        assert m["pairwise_accuracy"] == 1.0
        assert "chosen_score_mean" in m
        assert "rejected_score_mean" in m
        assert math.isclose(m["chosen_score_mean"], 2.0)
        assert math.isclose(m["rejected_score_mean"], 1.0)


class TestDiversityMetrics:
    def test_all_identical(self):
        d = compute_diversity_metrics(["the cat sat", "the cat sat", "the cat sat"])
        assert d["unique_response_ratio"] < 1.0
        assert d["unique_unigram_ratio"] == 3 / 9  # 3 unique words out of 9 total

    def test_all_different(self):
        d = compute_diversity_metrics(["alpha beta", "gamma delta", "epsilon zeta"])
        assert d["unique_response_ratio"] == 1.0
        assert d["unique_unigram_ratio"] == 1.0

    def test_empty(self):
        d = compute_diversity_metrics([])
        assert "diversity_error" in d


class TestLengthRewardCorrelation:
    def test_positive_correlation(self):
        # Longer responses get higher reward
        responses = ["a", "a b", "a b c", "a b c d", "a b c d e"]
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = length_reward_correlation(responses, scores)
        assert out["length_reward_correlation"] > 0.99

    def test_negative_correlation(self):
        responses = ["a", "a b", "a b c"]
        scores = [3.0, 2.0, 1.0]
        out = length_reward_correlation(responses, scores)
        assert out["length_reward_correlation"] < -0.99

    def test_too_few_samples(self):
        out = length_reward_correlation(["a"], [1.0])
        assert "error" in out


class TestHumanEvalAgreement:
    def test_basic(self):
        import pandas as pd
        df = pd.DataFrame({"preference": ["a", "a", "b", "tie"]})
        out = human_eval_agreement(df)
        assert out["wins_a"] == 2
        assert out["wins_b"] == 1
        assert out["ties"] == 1
        assert out["win_rate_a"] == 0.5
