# Results & Analysis

A single end-to-end RLHF run on **Qwen2.5-0.5B** with **Anthropic HH-RLHF**, executed on a consumer GPU. This document collects the numerical results, the training-curve narrative, and an honest assessment of what worked, what didn't, and what is left to investigate.

> All raw artefacts live under `outputs/`. Reproduce them by running the notebooks in order, or by `python scripts/smoke_test.py` for a 50-sample dry run.

## 1. Stage-by-stage numbers

### 1.1 Supervised fine-tuning (Stage 2)

| Quantity | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` |
| Adapter | LoRA (PEFT 0.10) |
| Steps | 1248 (≈ 2 epochs) |
| Train cross-entropy (step 50 → final) | 2.54 → **2.10** (−17%) |
| Final eval CE | 2.135 |
| Throughput | 2.7 samples/s (single GPU) |

The loss curve is monotone and converges cleanly (`outputs/figures/sft_loss.png`). The plateau around **CE ≈ 2.10** is consistent with the limited capacity of a 0.5B model on a noisy, dialogue-style corpus — the SFT proxy here is HH-RLHF *chosen* responses, not a curated instruction set. A dedicated instruction corpus (OpenAssistant, Alpaca) would be expected to push CE lower; that is left as future work.

### 1.2 Reward model (Stage 3)

| Quantity | Value |
|---|---|
| Architecture | `AutoModelForSequenceClassification` (Qwen2.5-0.5B + scalar head) |
| Adapter | LoRA |
| Steps | 624 (≈ 1 epoch) |
| Pairwise eval accuracy | **65.5%** (step 250) |
| Pairwise eval accuracy | 65.2% (step 500) |
| Final eval loss | 0.653 |
| Train loss | 0.71 |

Eval accuracy is essentially flat between steps 250 and 500, so additional reward training on this dataset has saturated. **65% pairwise accuracy on HH-RLHF is consistent with published numbers for compact reward models** — HH-RLHF labels are noisy and the gap between *chosen* and *rejected* is often small. The reward model is good enough to be a useful PPO signal, but it is not strong; this matters when reading the PPO results below.

### 1.3 PPO (Stage 4)

| Quantity | Value |
|---|---|
| Optimization | TRL `PPOTrainer` (TRL 0.9.6) |
| Initial policy | SFT adapter |
| Reward | LoRA-tuned RM head from §1.2 |
| Steps logged | 50 (every 10 steps) |
| Mean reward, step 0 → final | 2.24 → **1.93** (Δ = −0.31) |
| Mean reward, monotone fraction | **55%** of steps |
| Response length, step 0 → final (rollout) | 530 → **726** chars (+37%) |
| KL: max / mean / final | 18.66 / 12.42 / 10.06 |

There are **three observations that matter** here:

1. **Mean reward did not increase.** Across 50 logged steps, reward starts at 2.24 and ends at 1.93. Only ~55% of consecutive step pairs show an increase, i.e. the optimisation is barely above coin-flip on the reward signal.
2. **Rollout response length grew sharply.** The mean character length of generated responses rose from 530 to 726 (+37%) over the same window — a textbook signature of length-as-shortcut behaviour.
3. **KL stayed bounded.** Maximum KL divergence is 18.7 and the curve trends down, finishing at 10.1. The KL penalty is doing its job: the policy is not running away from the reference distribution.

The combination of (1) + (2) + (3) is the canonical **mild reward-hacking + KL-controlled** picture: the policy looked for cheap reward by writing more, but the KL penalty kept it from locking that pattern in.

### 1.4 Evaluation on held-out prompts (Stage 5)

50 held-out prompts, three models compared (base, SFT, PPO). Full responses at `outputs/samples/model_comparison.csv`.

| Model | Mean length (chars) | Median (chars) | Std (chars) | Mean length (words) |
|---|---|---|---|---|
| base | 477.7 | 540.0 | 185.8 | 83.9 |
| SFT  | 509.1 | 513.5 | 68.8  | 99.6 |
| PPO  | 506.8 | 500.0 | 57.3  | 99.7 |

PPO and SFT look almost identical on length — and *much* more consistent than the base model (std drops from 186 → 57). This is one of the stronger signals in the run: alignment **stabilises** output length even when it doesn't increase mean reward.

## 2. Did RLHF work? (Q1)

**Partially, on what HH-RLHF rewards. Not on what it doesn't.**

Qualitatively, on adversarial prompts (PII requests, hallucination-bait factual questions, prompts asking for harmful content), the PPO policy is closer to a refusal/redirect pattern than the base model — see the side-by-side examples in the README and the full table at `outputs/samples/model_comparison.csv`. This is consistent with HH-RLHF being a **harmlessness-leaning** dataset.

Quantitatively, mean reward did not improve over the run. The most defensible reading is that the reward model is too weak to pull a 0.5B policy further than the SFT baseline already provides on this data — the *floor* on what HH-RLHF can teach with a 65% RM and a 0.5B policy may already have been reached at SFT.

The clearest non-result is **factuality**: the Betty White example in the README shows both base and PPO confidently producing wrong information. RLHF on HH-RLHF is the wrong tool for that.

## 3. Reward hacking & verbosity bias (Q3)

`outputs/tables/reward_hacking_indicators.csv`:

| Indicator | Value |
|---|---|
| Reward Δ over training | −0.31 |
| Reward monotone fraction | 0.55 |
| Length increase during PPO rollouts | +37% |
| KL max / final | 18.7 / 10.1 |

`outputs/tables/verbosity_analysis.csv`:

| Quantity | SFT | PPO |
|---|---|---|
| Mean length (words) | 99.56 | 99.68 |
| Median length (words) | 103.5 | 101.0 |

The interesting finding is the **gap between rollout-time and eval-time length**. During PPO training the policy generated longer and longer responses (+37%) — but on the eval prompt set the saved adapter produces responses that are statistically indistinguishable in length from the SFT baseline. Three plausible explanations, in decreasing order of likelihood:

1. **Distribution shift.** PPO rollouts use prompts sampled from the training distribution (long multi-turn HH-RLHF conversations); evaluation uses a held-out subset that may have shorter, simpler prompts where the verbose pattern is harder to express.
2. **KL-bound regression.** The KL penalty pulled the policy back toward the SFT distribution between the rollout-time peak and the final saved checkpoint.
3. **Adapter capacity.** A small LoRA rank limits how much the policy can specialise toward verbose generation; the hack appears in rollouts but does not persist in weights.

A controlled experiment to disentangle these — train a second PPO run with the KL penalty disabled, save checkpoints every N steps, evaluate length at each — is in the future-work list (§5).

## 4. Sensitivity to dataset choice (Q2)

This run only covers HH-RLHF. The pipeline ships a second config (`configs/reward_alt.yaml`) for the binarised UltraFeedback dataset; running it produces a second reward model that can be plugged into a second PPO run for a clean cross-dataset comparison. **This is the highest-value next experiment in the queue** — see §5 — and addressing Q2 properly requires it.

## 5. Future work (the deeper questions)

Listed in priority order; each item is a real experiment, not a polish task.

1. **DPO baseline.** ✅ *Pipeline shipped.* `scripts/run_dpo.py` + `configs/dpo_qwen.yaml` + `src/dpo_train.py` train a DPO LoRA adapter from the SFT initialisation in a single offline pass — no reward model, no rollout buffer. Run it (`python scripts/run_dpo.py --config configs/dpo_qwen.yaml`) and the SFT vs PPO vs DPO three-way comparison populates this section. Expected outcome at this scale: DPO matches or beats PPO on preference accuracy at a fraction of wall-clock and stability cost.
2. **Capability benchmarks (alignment tax).** ✅ *Harness shipped.* `scripts/run_capability_eval.py` wraps `lm-evaluation-harness` and runs MMLU / GSM8K across any subset of {base, sft, rm, ppo, dpo}. Outputs land in `outputs/tables/capability_eval.csv`. Quantifying the alignment tax at sub-1B scale is genuinely under-reported — running the full benchmark fills that gap.
3. **Cross-dataset comparison (Q2).** Re-run reward training on UltraFeedback (`configs/reward_alt.yaml`) and re-run PPO/DPO with the new RM / preference set. Compare RM accuracy, downstream behaviour, and length dynamics.
4. **Comparison to `Qwen2.5-0.5B-Instruct`.** Without this, claims about RLHF effectiveness are unanchored. Drop the official model into the capability eval registry and re-run.
5. **LLM-as-judge evaluation.** Pairwise win-rate from a stronger judge (GPT-4 / Claude class) on the eval set. Auto-metrics + judge wins is the modern standard.
6. **Multi-seed runs + ablations.** KL coefficient (0.0, 0.05, 0.2), LoRA rank (4, 8, 16, 32), data scale (1k, 10k, all). One run is anecdote; three seeds × four ablations is data.
7. **Controlled reward-hacking experiment.** Reproduce the rollout-length blowup by disabling KL, document the failure mode in detail.

## 6. Honest caveats

- **Single seed.** Every number above is from one run. Variance is unknown.
- **Reward model is weak.** 65% pairwise accuracy is a soft ceiling on what PPO can extract.
- **Eval prompts are HH-RLHF held-out.** They are *in-distribution* for the reward model. Out-of-distribution evaluation (MT-Bench-style prompts) is not in this release.
- **No human evaluation in this run.** Notebook `06_human_eval_template.ipynb` provides the scaffolding; the annotation pass has not been done.
- **Length stats use character count for some figures, word count for others.** Both are reported in the tables, but be careful when comparing across sections.
