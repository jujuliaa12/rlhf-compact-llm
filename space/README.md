---
title: RLHF Compact LLM Demo
emoji: 🦙
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: mit
---

# rlhf-compact-llm — interactive demo

Side-by-side comparison of **base / SFT / PPO** (and DPO when available)
responses from the Qwen2.5-0.5B model family aligned on Anthropic HH-RLHF.

- Source: [github.com/jujuliaa12/rlhf-compact-llm](https://github.com/jujuliaa12/rlhf-compact-llm)
- Model adapters: [`Julia569922` on Hugging Face](https://huggingface.co/Julia569922)

## Run locally

```bash
cd space
pip install -r requirements.txt
python app.py
```

## Deploy as a Hugging Face Space

From the project root:

```bash
hf upload Julia569922/rlhf-compact-llm-demo space . --repo-type space
```

This Space is implemented as a single `app.py`; the LoRA adapters are
pulled from the Hub at startup, so the same file works locally and in the
hosted Space.
