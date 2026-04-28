# Design Document — RLHF Pipeline for Compact LLMs

This document captures the design choices behind the pipeline: what each stage does, why it is set up that way, and what trade-offs the configuration is making.

## Open questions the project investigates

1. Does PPO-based RLHF improve response quality over an SFT-only baseline at the **0.5B–1.7B** scale, where the SFT prior is much weaker than at 7B+?
2. How sensitive is the reward model — and the downstream PPO policy — to the **choice of preference dataset** (HH-RLHF vs UltraFeedback)?
3. To what extent do compact models exhibit **reward hacking** or **verbosity bias** during PPO training, and can simple length-aware diagnostics catch it?

## Experiment matrix

| Experiment | Base model    | SFT corpus      | Reward corpus   | PPO | Purpose                               |
|------------|---------------|-----------------|-----------------|-----|---------------------------------------|
| E1         | Qwen2.5-0.5B  | HH-RLHF chosen  | HH-RLHF         | Yes | Primary pipeline (Q1, Q3)             |
| E2         | Qwen2.5-0.5B  | HH-RLHF chosen  | UltraFeedback   | Yes | Cross-dataset comparison (Q2)         |
| E3 (opt.)  | SmolLM2-1.7B  | HH-RLHF chosen  | HH-RLHF         | Yes | Scale comparison (Q1)                 |

E1 + E2 is the minimum useful set; E3 is gated on compute.

### Note on SFT data

HH-RLHF is a preference dataset, not an instruction-tuning corpus. Using its *chosen* responses as a lightweight SFT proxy keeps the pipeline self-contained on a single source. The cost is a weaker SFT baseline than dedicated instruction sets (OpenAssistant, Alpaca) would produce — this is documented as a known limitation rather than hidden.

## Stage-by-stage design

### Stage 1: Data preparation
- Pull HH-RLHF and UltraFeedback from the Hugging Face Hub.
- Parse multi-turn conversations into `prompt / chosen / rejected` triplets.
- Validate, clean, and compute length statistics.
- Persist processed splits to `data/processed/` so later stages do not re-parse.

### Stage 2: Supervised fine-tuning
- LoRA-fine-tune the base model on chosen responses.
- Use TRL's `SFTTrainer` for a stable training loop.
- Save adapter weights and CSV training logs.
- Generate baseline samples for later qualitative comparison.

### Stage 3: Reward model
- Train a reward head on (prompt, response) pairs from a preference dataset.
- Evaluate pairwise preference accuracy on a held-out split.
- Train one model per dataset to enable the Q2 comparison.

### Stage 4: PPO
- Initialize the policy from the SFT adapter.
- Score generations with the reward model; optimize with PPO.
- Log mean reward, KL divergence, and response length per step.
- Save the aligned adapter.

### Stage 5: Evaluation
- Generate responses from base, SFT, and PPO models on a held-out prompt set.
- Compute automatic metrics (length, reward score, diversity).
- Compute reward-hacking and verbosity-bias indicators (length-vs-reward correlation, repetition rate, refusal rate).
- Persist comparison tables and sample CSVs.

### Stage 6 (optional): Human evaluation
- Blind pairwise SFT-vs-PPO comparison via the notebook template.
- Aggregate preferences into win rates.

## Outputs

| Artifact                | Location                              | Format          |
|-------------------------|---------------------------------------|-----------------|
| Processed datasets      | `data/processed/`                     | Parquet/JSON    |
| SFT model               | `outputs/models/sft_qwen/`            | LoRA adapter    |
| Reward model            | `outputs/models/reward_model_hh/`     | LoRA adapter    |
| PPO model               | `outputs/models/ppo_qwen/`            | LoRA adapter    |
| Training logs           | `outputs/logs/`                       | CSV + YAML      |
| Plots                   | `outputs/figures/`                    | PNG             |
| Evaluation tables       | `outputs/tables/`                     | CSV             |
| Sample generations      | `outputs/samples/`                    | JSON / CSV      |

## Risk register

| Risk                                | Mitigation                                                                            |
|-------------------------------------|---------------------------------------------------------------------------------------|
| TRL API breaking changes            | Pinned `trl==0.9.6`; PPO code isolated behind `src/ppo_train.py`                      |
| GPU OOM during training             | LoRA only; small batch sizes; gradient accumulation; auto-fallback to CPU             |
| Poor reward-model accuracy          | Pairwise validation accuracy gate; alternative dataset for cross-check                |
| PPO instability / reward hacking    | KL monitoring; response-length cap; length-vs-reward correlation as diagnostic        |
| Slow CPU runs                       | Configurable `max_samples`; smaller model is the supported CPU path                   |
| Dataset download flakiness          | Local HF cache; data-prep stage is idempotent                                         |

## Replication checklist

- [ ] Python 3.10 environment created
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `python scripts/smoke_test.py` passes end-to-end
- [ ] Notebook 01 produces processed splits in `data/processed/`
- [ ] Notebook 02 produces an SFT adapter in `outputs/models/sft_qwen/`
- [ ] Notebook 03 produces a reward model
- [ ] Notebook 04 produces a PPO-aligned model
- [ ] Notebook 05 produces evaluation tables and figures
- [ ] All CSV logs present in `outputs/logs/`
- [ ] All figures present in `outputs/figures/`
- [ ] All sample outputs present in `outputs/samples/`
