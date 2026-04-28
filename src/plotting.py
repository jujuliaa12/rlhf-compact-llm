"""
Plotting utilities for training curves, distributions, and evaluation.

All plots are saved as PNG files to outputs/figures/ by default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

DEFAULT_FIGURE_DIR = Path("outputs/figures")


def _ensure_dir(path: Path) -> None:
    """Ensure the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Training loss curves
# ---------------------------------------------------------------------------

def plot_training_loss(
    log_df: pd.DataFrame,
    title: str = "Training Loss",
    output_path: Optional[str | Path] = None,
    loss_col: str = "loss",
    step_col: str = "step",
) -> None:
    """
    Plot training loss curve from a log DataFrame.

    Args:
        log_df: DataFrame with training logs (from Trainer log_history).
        title: Plot title.
        output_path: Path to save PNG. If None, uses default directory.
        loss_col: Column name for loss values.
        step_col: Column name for step values.
    """
    df = log_df.dropna(subset=[loss_col])
    if df.empty:
        logger.warning("No loss data to plot for '%s'", title)
        return

    fig, ax = plt.subplots()
    ax.plot(df[step_col] if step_col in df.columns else df.index, df[loss_col])
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / f"{title.lower().replace(' ', '_')}.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


def plot_eval_loss(
    log_df: pd.DataFrame,
    title: str = "Evaluation Loss",
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot evaluation loss curve."""
    df = log_df.dropna(subset=["eval_loss"])
    if df.empty:
        logger.warning("No eval_loss data to plot")
        return

    fig, ax = plt.subplots()
    x = df["step"] if "step" in df.columns else df.index
    ax.plot(x, df["eval_loss"], marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Eval Loss")
    ax.set_title(title)

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "eval_loss.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


# ---------------------------------------------------------------------------
# PPO-specific plots
# ---------------------------------------------------------------------------

def plot_reward_curve(
    ppo_log_df: pd.DataFrame,
    title: str = "PPO Mean Reward over Training",
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot mean reward curve from PPO training logs."""
    if "mean_reward" not in ppo_log_df.columns:
        logger.warning("No mean_reward column in PPO log")
        return

    fig, ax = plt.subplots()
    ax.plot(ppo_log_df["step"], ppo_log_df["mean_reward"])
    ax.set_xlabel("PPO Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title(title)

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "ppo_reward_curve.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


def plot_kl_divergence(
    ppo_log_df: pd.DataFrame,
    title: str = "PPO KL Divergence over Training",
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Plot KL divergence curve from PPO training.

    High KL divergence may indicate policy instability or collapse.
    """
    if "kl_divergence" not in ppo_log_df.columns:
        logger.warning("No kl_divergence column in PPO log")
        return

    df = ppo_log_df.dropna(subset=["kl_divergence"])

    fig, ax = plt.subplots()
    ax.plot(df["step"], df["kl_divergence"], color="red")
    ax.set_xlabel("PPO Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title(title)

    # Add target KL reference line if reasonable
    if df["kl_divergence"].max() > 0:
        ax.axhline(y=6.0, color="gray", linestyle="--", alpha=0.5, label="Target KL=6.0")
        ax.legend()

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "ppo_kl_divergence.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


def plot_ppo_response_length(
    ppo_log_df: pd.DataFrame,
    title: str = "PPO Response Length over Training",
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot response length trend during PPO training."""
    if "response_length_mean" not in ppo_log_df.columns:
        logger.warning("No response_length_mean column in PPO log")
        return

    fig, ax = plt.subplots()
    ax.plot(ppo_log_df["step"], ppo_log_df["response_length_mean"], color="green")
    ax.set_xlabel("PPO Step")
    ax.set_ylabel("Mean Response Length (chars)")
    ax.set_title(title)

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "ppo_response_length.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


# ---------------------------------------------------------------------------
# Dataset distribution plots
# ---------------------------------------------------------------------------

def plot_length_distribution(
    lengths_df: pd.DataFrame,
    title: str = "Response Length Distribution",
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Plot histograms of chosen vs rejected response lengths.

    Args:
        lengths_df: DataFrame with 'chosen_len' and 'rejected_len' columns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(lengths_df["chosen_len"], bins=50, alpha=0.7, label="Chosen", color="steelblue")
    axes[0].hist(lengths_df["rejected_len"], bins=50, alpha=0.7, label="Rejected", color="salmon")
    axes[0].set_xlabel("Length (characters)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Chosen vs Rejected Length Histogram")
    axes[0].legend()

    # Boxplot
    box_data = pd.DataFrame({
        "Length": list(lengths_df["chosen_len"]) + list(lengths_df["rejected_len"]),
        "Type": ["Chosen"] * len(lengths_df) + ["Rejected"] * len(lengths_df),
    })
    sns.boxplot(data=box_data, x="Type", y="Length", ax=axes[1], palette=["steelblue", "salmon"])
    axes[1].set_title("Chosen vs Rejected Length Boxplot")

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "length_distribution.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


def plot_reward_score_distribution(
    chosen_scores: list[float],
    rejected_scores: list[float],
    title: str = "Reward Score Distribution",
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot histograms of reward scores for chosen vs rejected responses."""
    fig, ax = plt.subplots()
    ax.hist(chosen_scores, bins=40, alpha=0.7, label="Chosen", color="steelblue")
    ax.hist(rejected_scores, bins=40, alpha=0.7, label="Rejected", color="salmon")
    ax.set_xlabel("Reward Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "reward_score_distribution.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


# ---------------------------------------------------------------------------
# Evaluation comparison plots
# ---------------------------------------------------------------------------

def plot_model_comparison_bar(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str = "Model Comparison",
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot a bar chart comparing a metric across models."""
    fig, ax = plt.subplots()
    ax.bar(summary_df["model_name"], summary_df[metric_col], color="steelblue")
    ax.set_ylabel(metric_col)
    ax.set_title(title)
    plt.xticks(rotation=15)

    if output_path is None:
        safe_metric = metric_col.replace(" ", "_").replace("/", "_")
        output_path = DEFAULT_FIGURE_DIR / f"comparison_{safe_metric}.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)


def plot_verbosity_comparison(
    sft_lengths: list[int],
    ppo_lengths: list[int],
    title: str = "Response Length: SFT vs PPO",
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot side-by-side length distributions for SFT vs PPO models."""
    fig, ax = plt.subplots()
    ax.hist(sft_lengths, bins=40, alpha=0.7, label="SFT", color="steelblue")
    ax.hist(ppo_lengths, bins=40, alpha=0.7, label="PPO", color="coral")
    ax.set_xlabel("Response Length (words)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    if output_path is None:
        output_path = DEFAULT_FIGURE_DIR / "verbosity_comparison.png"
    output_path = Path(output_path)
    _ensure_dir(output_path)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Plot saved: %s", output_path)
