---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
tags:
  - rlhf
  - dpo
  - lora
  - qwen2.5
---

# DPO-Qwen2.5-0.5B (LoRA adapter)

DPO-aligned LoRA adapter produced by Stage 4 (alternative path) of the [`rlhf-compact-llm`](../../..) pipeline.

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **Initial policy:** SFT adapter at `outputs/models/sft_qwen/`, merged into the base before DPO.
- **Preference data:** `Anthropic/hh-rlhf` (5000 chosen/rejected triples)
- **Trainer:** `trl.DPOTrainer` with `DPOConfig` (TRL 0.9.6)
- **Loss:** Sigmoid Bradley-Terry (`loss_type="sigmoid"`), `beta=0.1`
- **Adapter:** LoRA via `peft==0.10.0`

## Training results (single CPU run, 312 steps, ~7 hours)

| Metric | Start | End |
|---|---:|---:|
| Loss | 0.6923 | **0.5992** |
| `rewards/margins` (chosen − rejected) | 0.002 | **0.322** |
| `rewards/accuracies` (pairwise) | 0.512 | **0.663** |
| `rewards/chosen` (implicit log-ratio) | -0.002 | -0.545 |
| `rewards/rejected` | -0.004 | -0.867 |

The model learned: pairwise preference accuracy rose from chance (51%) to **66%**, and the chosen-vs-rejected margin grew by two orders of magnitude. This is in clear contrast to the PPO run for the same task (mean reward *decreased* over training) — exactly the textbook story for DPO at compact scale.

Training config and CSV log: `outputs/logs/dpo_config_snapshot.yaml`, `outputs/logs/dpo_training_log.csv`.
Plot: `outputs/figures/dpo_training.png`.

## Loading

From the Hugging Face Hub:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = PeftModel.from_pretrained(base, "Julia569922/qwen2.5-0.5b-rlhf-dpo")
tok = AutoTokenizer.from_pretrained("Julia569922/qwen2.5-0.5b-rlhf-dpo")
```

From this repository:

```python
model = PeftModel.from_pretrained(base, "outputs/models/dpo_qwen")
```
