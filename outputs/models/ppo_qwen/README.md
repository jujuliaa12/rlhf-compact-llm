---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
tags:
  - rlhf
  - ppo
  - lora
  - qwen2.5
---

# PPO-Qwen2.5-0.5B (LoRA adapter)

PPO-aligned LoRA adapter produced by Stage 4 of the [`rlhf-compact-llm`](../../..) pipeline.

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **Initial policy:** SFT adapter at `outputs/models/sft_qwen/`
- **Reward model:** `outputs/models/reward_model_hh/`
- **Trainer:** `trl.PPOTrainer` (TRL 0.9.6) with a value head
- **Adapter:** LoRA via `peft==0.10.0`

Training config and CSV log: `outputs/logs/ppo_config_snapshot.yaml`, `outputs/logs/ppo_training_log.csv`.
Plots: `outputs/figures/ppo_kl.png`, `outputs/figures/ppo_reward_length.png`.

## Loading

From the Hugging Face Hub:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
peft_model = PeftModel.from_pretrained(base, "julia569922/qwen2.5-0.5b-rlhf-ppo")
policy = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
tok = AutoTokenizer.from_pretrained("julia569922/qwen2.5-0.5b-rlhf-ppo")
```

For inference only (no value head needed):

```python
model = PeftModel.from_pretrained(base, "julia569922/qwen2.5-0.5b-rlhf-ppo")
```

From this repository:

```python
peft_model = PeftModel.from_pretrained(base, "outputs/models/ppo_qwen")
```
