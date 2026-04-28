---
base_model: Qwen/Qwen2.5-0.5B
library_name: peft
tags:
  - rlhf
  - sft
  - lora
  - qwen2.5
---

# SFT-Qwen2.5-0.5B (LoRA adapter)

LoRA adapter produced by Stage 2 (Supervised Fine-Tuning) of the [`rlhf-compact-llm`](../../..) pipeline.

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **SFT data:** chosen responses from `Anthropic/hh-rlhf` (used as a lightweight instruction-following proxy)
- **Adapter:** LoRA via `peft==0.10.0`
- **Trainer:** `trl.SFTTrainer` (TRL 0.9.6)

Training config and CSV log: `outputs/logs/sft_config_snapshot.yaml`, `outputs/logs/sft_training_log.csv`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = PeftModel.from_pretrained(base, "outputs/models/sft_qwen")
tok = AutoTokenizer.from_pretrained("outputs/models/sft_qwen")
```
