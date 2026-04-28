# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- DPO baseline alongside PPO
- Cross-dataset reward-model comparison (HH-RLHF vs UltraFeedback)
- LLM-as-judge evaluation harness
- Capability benchmarks (MMLU, GSM8K) on base / SFT / PPO

## [0.1.0] - 2026-04-29

### Added
- End-to-end SFT → reward model → PPO → evaluation pipeline on Qwen2.5-0.5B + Anthropic HH-RLHF.
- Modular `src/` library: data utilities, preprocessing, model loading, three trainers, inference, evaluation, metrics, plotting.
- CLI entry points (`scripts/run_*.py`) with matching Jupyter notebooks.
- Smoke test (`scripts/smoke_test.py`) that exercises the full pipeline on ~50 samples.
- Trained LoRA adapter checkpoints for SFT, reward model, and PPO.
- CSV training logs, PNG figures, and analysis tables for reward hacking and verbosity bias.
- Unit tests for `src.metrics` and `src.preprocessing`.
- GitHub Actions CI: ruff lint + pytest on Python 3.10 / 3.11.
