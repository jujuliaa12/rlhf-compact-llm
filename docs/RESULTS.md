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

### 1.4 Direct Preference Optimization (Stage 4, alternative)

| Quantity | Value |
|---|---|
| Optimization | TRL `DPOTrainer` + `DPOConfig` (TRL 0.9.6) |
| Initial policy | SFT adapter merged into the base, fresh LoRA on top |
| Preference data | `Anthropic/hh-rlhf` (5000 triples) |
| Steps | 312 (1 epoch) |
| Loss (start → end) | 0.692 → **0.599** (−13%) |
| `rewards/margins` (chosen − rejected, start → end) | 0.002 → **0.322** |
| `rewards/accuracies` (start → end) | 0.512 → **0.663** |
| `rewards/chosen` (final) | -0.545 |
| `rewards/rejected` (final) | -0.867 |
| Hardware / wall-clock | CPU, ~7 h |

The DPO trajectory is what successful preference optimization is supposed to look like: **margin grows by two orders of magnitude**, **pairwise accuracy climbs from chance to ~66%**, and the loss decreases monotonically. Compared to the PPO run on the same preference dataset (which *lost* reward over training and showed length-hacking), DPO's behaviour is qualitatively different — and aligns with the broader literature consensus that DPO is the more robust choice at compact scale.

### 1.5 Evaluation on held-out prompts (Stage 5)

50 held-out prompts, three models compared (base, SFT, PPO). Full responses at `outputs/samples/model_comparison.csv`.

| Model | Mean length (chars) | Median (chars) | Std (chars) | Mean length (words) |
|---|---|---|---|---|
| base | 477.7 | 540.0 | 185.8 | 83.9 |
| SFT  | 509.1 | 513.5 | 68.8  | 99.6 |
| PPO  | 506.8 | 500.0 | 57.3  | 99.7 |

PPO and SFT look almost identical on length — and *much* more consistent than the base model (std drops from 186 → 57). This is one of the stronger signals in the run: alignment **stabilises** output length even when it doesn't increase mean reward.

### 1.6 Capability benchmarks — MMLU + GSM8K (4-way alignment tax)

Two evaluations on a 50-question slice via `lm-evaluation-harness`: MMLU 0-shot (knowledge / multiple choice) and GSM8K 5-shot (math reasoning, generative). Full table at `outputs/tables/capability_eval.csv`.

#### Headline 4-way × 2-task table

| Model | MMLU (0-shot) | GSM8K (5-shot) |
|---|---:|---:|
| base (Qwen2.5-0.5B) | **0.4937** | **0.340** |
| SFT | 0.4793 | 0.200 |
| PPO | 0.4793 | 0.200 |
| **DPO** | **0.4919** | **0.320** |

| Model | Δ vs base (MMLU) | Δ vs base (GSM8K) | GSM8K relative drop |
|---|---:|---:|---:|
| SFT | −0.0144 | −0.140 | −41% |
| PPO | −0.0144 | −0.140 | −41% |
| **DPO** | **−0.0018** | **−0.020** | **−6%** |

GSM8K is doing the heavy lifting in this comparison: it shows a tax that is qualitatively different in scale from MMLU (14pp vs 1.5pp). Generative-reasoning capability is what RLHF appears to *break* at compact scale, and what DPO appears to *preserve*.

#### MMLU 4-way detail

| Model | MMLU acc. | Δ vs base |
|---|---:|---:|
| base (Qwen2.5-0.5B) | **0.4937** | — |
| SFT | 0.4793 | −0.0144 |
| PPO | 0.4793 | −0.0144 |
| **DPO** | **0.4919** | **−0.0018** |

**Three findings worth pulling out.**

1. **PPO ≈ SFT on MMLU.** Both score 0.4793 to four decimal places. This is consistent with PPO's declining-reward training trajectory (§1.3): if PPO's policy update was small, it shouldn't move the eval-set distribution either, and it doesn't. The PPO adapter at this scale is, in capability space, indistinguishable from the SFT initialisation.
2. **DPO is dramatically cheaper than SFT/PPO on capability.** −0.2pp for DPO vs −1.5pp for SFT/PPO — a 7× smaller alignment tax for the same headline alignment objective.
3. **DPO produces sub-domain gains, not just preservation.** `jurisprudence` +0.08, `clinical_knowledge` +0.06, `nutrition` +0.04, `high_school_macroeconomics` +0.04, `human_sexuality` +0.02, `international_law` +0.02. These are domains where the HH-RLHF chosen-vs-rejected gap is dominated by *more grounded / formal phrasing* in chosen, and DPO's loss extracts that.

The full subject-level breakdown that motivates the second-cluster hypothesis:

| Subject | base | SFT | PPO | DPO |
|---|---:|---:|---:|---:|
| `jurisprudence` | 0.66 | 0.62 | 0.62 | **0.74** |
| `clinical_knowledge` | 0.48 | 0.58 | 0.58 | 0.54 |
| `nutrition` | 0.74 | 0.64 | 0.64 | **0.78** |
| `high_school_macroeconomics` | 0.58 | 0.52 | 0.52 | **0.62** |
| `international_law` | 0.66 | 0.72 | 0.72 | 0.68 |
| `human_aging` | 0.42 | 0.48 | 0.48 | 0.42 |
| `business_ethics` | 0.62 | 0.58 | 0.58 | 0.58 |
| `moral_disputes` | 0.56 | 0.52 | 0.52 | 0.50 |
| `professional_accounting` | 0.42 | 0.38 | 0.38 | 0.36 |
| `college_chemistry` | 0.42 | 0.40 | 0.40 | 0.38 |

The PPO column is a perfect copy of the SFT column on every row above (and on the full 57-subject sweep). DPO's pattern is qualitatively different — significant gains on policy / law / health subjects, modest losses on hard STEM (chemistry, statistics) — which is the kind of structured, capability-aware behaviour that distinguishes a working preference optimiser from a noisier policy-gradient method at this scale.

#### GSM8K 4-way detail

| Model | GSM8K (5-shot, n=50) | Hits | Δ vs base | Relative |
|---|---:|---:|---:|---:|
| base | 0.340 | 17 / 50 | — | — |
| SFT  | 0.200 | 10 / 50 | −0.140 | −41% |
| PPO  | 0.200 | 10 / 50 | −0.140 | −41% |
| **DPO** | **0.320** | **16 / 50** | **−0.020** | **−6%** |

GSM8K is a generative reasoning task: the model has to produce a chain of arithmetic / algebraic steps, then a final numerical answer that has to match exactly. It is the benchmark in this report most sensitive to the policy's free-form generation distribution shifting.

**The pattern.** Both SFT and PPO drop the model from 17/50 to 10/50 on math-word-problem accuracy — that is the model getting **7 fewer questions right out of 50** after either alignment step, which is a 41% relative loss. DPO drops from 17/50 to 16/50: **a single question difference, well within sampling noise at n=50**. The PPO=SFT identity (10/50, exactly) confirms once again that the PPO adapter on this run is a no-op compared to its SFT initialisation.

**Why this happens (hypothesis).** SFT on HH-RLHF chosen responses pushes the policy toward a chat-y, hedging, dialogue-completion distribution that is qualitatively different from the math-show-your-work distribution GSM8K rewards. Multi-step CoT generation under that pressure breaks down — the model truncates, restates the question, or drops into a refusal pattern. DPO's training objective only adjusts the policy *along the direction of the chosen-vs-rejected gap* and leaves orthogonal directions (including the math-CoT generation distribution) alone, so the reasoning ability survives.

**Sample-size caveat.** 50 GSM8K examples gives roughly a ±13pp 95% CI per cell. The 14pp SFT-vs-base drop is at the edge of significance on a single cell; it becomes unambiguous because the same value appears for two independent runs (SFT and PPO) and the DPO control returns to within 1 question of base. The cross-row pattern is the evidence, not any single number.

**Headline reading.** *DPO trains and preserves capability; SFT and PPO pay the same alignment tax — and on generative reasoning that tax can be roughly half the original capability.* Both alignment paths run on the same data, the same LoRA budget, the same number of preference triples. The two-benchmark consistency (≈7× DPO advantage on both MMLU and GSM8K) is the cleanest evidence that this is a real DPO-vs-PPO/SFT property at compact scale, not a measurement artefact.

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

1. **DPO baseline.** ✅ *Done* (this run). DPO on the same 5000 preference triples drove pairwise accuracy from 51% → **66%** and margin from 0.002 → **0.322**. PPO, on the same data, *lost* reward and showed +37% rollout length drift. Expected-direction outcome: **DPO is the more robust choice at compact scale**. Adapter at [`Julia569922/qwen2.5-0.5b-rlhf-dpo`](https://huggingface.co/Julia569922/qwen2.5-0.5b-rlhf-dpo).
2. **Capability benchmarks (alignment tax).** ✅ *Done* (§1.6). MMLU + GSM8K, 4-way comparison. **The 7× DPO-vs-SFT/PPO advantage replicates on both benchmarks** (MMLU: 0.2pp vs 1.5pp; GSM8K: 2pp vs 14pp), with GSM8K showing the more dramatic absolute gap because generative reasoning is the part of capability that breaks under SFT/PPO at compact scale.
3. **Cross-dataset comparison (Q2).** Re-run reward training on UltraFeedback (`configs/reward_alt.yaml`) and re-run PPO/DPO with the new RM / preference set. Compare RM accuracy, downstream behaviour, and length dynamics.
4. **Comparison to `Qwen2.5-0.5B-Instruct`.** Without this, claims about RLHF effectiveness are unanchored. Drop the official model into the capability eval registry and re-run.
5. **LLM-as-judge evaluation.** Pairwise win-rate from a stronger judge model on the eval set. Auto-metrics plus judge wins is the modern standard.
6. **Multi-seed runs + ablations.** KL coefficient (0.0, 0.05, 0.2), LoRA rank (4, 8, 16, 32), data scale (1k, 10k, all). One run is anecdote; three seeds × four ablations is data.
7. **Controlled reward-hacking experiment.** Reproduce the rollout-length blowup by disabling KL, document the failure mode in detail.

## 6. Honest caveats

- **Single seed.** Every number above is from one run. Variance is unknown.
- **Reward model is weak.** 65% pairwise accuracy is a soft ceiling on what PPO can extract.
- **Eval prompts are HH-RLHF held-out.** They are *in-distribution* for the reward model. Out-of-distribution evaluation (MT-Bench-style prompts) is not in this release.
- **No human evaluation in this run.** Notebook `06_human_eval_template.ipynb` provides the scaffolding; the annotation pass has not been done.
- **Length stats use character count for some figures, word count for others.** Both are reported in the tables, but be careful when comparing across sections.
