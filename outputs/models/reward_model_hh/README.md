---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
tags:
  - rlhf
  - reward-model
  - lora
  - qwen2.5
---

# Reward Model (HH-RLHF, LoRA adapter)

LoRA adapter for the reward head trained in Stage 3 of the [`rlhf-compact-llm`](../../..) pipeline.

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **Preference data:** `Anthropic/hh-rlhf` (chosen vs rejected pairs)
- **Trainer:** `trl.RewardTrainer` (TRL 0.9.6) with a sequence-classification head
- **Loss:** Bradley-Terry pairwise preference

Training config and CSV log: `outputs/logs/reward_config_snapshot.yaml`, `outputs/logs/reward_training_log.csv`.

## Loading

From the Hugging Face Hub:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-0.5B", num_labels=1
)
model = PeftModel.from_pretrained(base, "julia569922/qwen2.5-0.5b-rlhf-rm")
tok = AutoTokenizer.from_pretrained("julia569922/qwen2.5-0.5b-rlhf-rm")
```

From this repository:

```python
model = PeftModel.from_pretrained(base, "outputs/models/reward_model_hh")
tok = AutoTokenizer.from_pretrained("outputs/models/reward_model_hh")
```
