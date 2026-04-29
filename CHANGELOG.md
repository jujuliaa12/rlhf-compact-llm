# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **DPO baseline (trained).** `src/dpo_train.py` + `scripts/run_dpo.py` + `configs/dpo_qwen.yaml` train a Direct Preference Optimization adapter directly from preference triples — no reward model, no rollout buffer. The first end-to-end run on Qwen2.5-0.5B + 5000 HH-RLHF triples drives pairwise accuracy from 51% → 66% and `rewards/margins` from 0.002 → 0.322 over 312 steps, in clear contrast to PPO's *declining* mean reward on the same data. Adapter published at [`Julia569922/qwen2.5-0.5b-rlhf-dpo`](https://huggingface.co/Julia569922/qwen2.5-0.5b-rlhf-dpo). Training plot: `outputs/figures/dpo_training.png`.
- **Capability evaluation harness.** `scripts/run_capability_eval.py` wraps `lm-evaluation-harness` to run MMLU / GSM8K across any subset of {base, sft, ppo, dpo}, producing an "alignment tax" comparison table.
- **HF Space demo.** `space/app.py` is a Gradio side-by-side chat that loads the SFT / PPO / DPO adapters from the Hub and runs them on the same prompt. Hosted at [`Julia569922/rlhf-compact-llm-demo`](https://huggingface.co/spaces/Julia569922/rlhf-compact-llm-demo).
- pyproject extras: `[eval]` (lm-eval) and `[demo]` (gradio).

### Planned
- Cross-dataset reward-model comparison (HH-RLHF vs UltraFeedback)
- LLM-as-judge evaluation harness
- Multi-seed runs + KL / rank / data-scale ablations
- Controlled reward-hacking experiment with KL disabled

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
