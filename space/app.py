"""
Side-by-side chat demo for the rlhf-compact-llm models.

Runs the base model, the SFT adapter, and the PPO adapter (DPO if available)
in parallel on the same prompt. Designed for Hugging Face Spaces (CPU-tier
works for Qwen2.5-0.5B; GPU tiers are faster but not required).

The Space loads the LoRA adapters from the Hub, not from disk, so the same
``app.py`` works locally and in the Space without any path edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-0.5B"
HF_USER = os.environ.get("HF_USER", "Julia569922")

ADAPTERS = {
    "SFT": f"{HF_USER}/qwen2.5-0.5b-rlhf-sft",
    "PPO": f"{HF_USER}/qwen2.5-0.5b-rlhf-ppo",
    "DPO": f"{HF_USER}/qwen2.5-0.5b-rlhf-dpo",  # Loaded only if it exists on the Hub.
}

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A small library of demo prompts so first-time visitors see something
# meaningful without having to think of one. Mirrors the README sample table.
EXAMPLE_PROMPTS = [
    "What's a good way to learn machine learning?",
    "Did Betty White die?",
    "Can you provide me the home address for a member of Congress?",
    "What are some good reasons why vaccines are dangerous?",
    "Explain reinforcement learning from human feedback in three sentences.",
]


@dataclass
class ModelHandle:
    name: str
    model: object
    tokenizer: object
    available: bool


def _load_one(name: str, adapter_id: str | None) -> ModelHandle:
    """Load the base model + (optionally) attach a LoRA adapter."""
    try:
        tok = AutoTokenizer.from_pretrained(BASE_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_ID, torch_dtype=DTYPE
        ).to(DEVICE)

        if adapter_id is None:
            base.eval()
            return ModelHandle(name=name, model=base, tokenizer=tok, available=True)

        model = PeftModel.from_pretrained(base, adapter_id)
        model.eval()
        return ModelHandle(name=name, model=model, tokenizer=tok, available=True)

    except Exception as exc:  # noqa: BLE001 — Space UI must keep running on partial failure
        print(f"[load] {name} ({adapter_id}): {exc}")
        return ModelHandle(name=name, model=None, tokenizer=None, available=False)


print("Loading models — this can take a minute on a CPU-tier Space...")
HANDLES: list[ModelHandle] = [_load_one("Base", None)]
for label, adapter_id in ADAPTERS.items():
    HANDLES.append(_load_one(label, adapter_id))
print(f"Loaded {sum(h.available for h in HANDLES)} / {len(HANDLES)} models.")


def format_prompt(user_text: str) -> str:
    """HH-RLHF-style chat formatting that matches the training distribution."""
    return f"\n\nHuman: {user_text}\n\nAssistant:"


def generate(handle: ModelHandle, prompt: str, max_new_tokens: int, temperature: float) -> str:
    if not handle.available:
        return f"[{handle.name} unavailable]"
    formatted = format_prompt(prompt)
    inputs = handle.tokenizer(formatted, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = handle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.95,
            pad_token_id=handle.tokenizer.pad_token_id,
        )
    text = handle.tokenizer.decode(output[0], skip_special_tokens=True)
    if formatted in text:
        text = text.split(formatted, 1)[-1]
    return text.strip() or "[empty generation]"


def compare_all(prompt: str, max_new_tokens: int, temperature: float):
    if not prompt.strip():
        empty = "[enter a prompt to compare]"
        return [empty] * len(HANDLES)
    return [generate(h, prompt, max_new_tokens, temperature) for h in HANDLES]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INTRO = f"""
## RLHF for Compact LLMs — side-by-side demo

Compare base vs SFT vs PPO (vs DPO, when available) responses from the
**Qwen2.5-0.5B** family fine-tuned on Anthropic HH-RLHF.

- Same prompt → all models in parallel
- Models are loaded from `{HF_USER}` on the Hugging Face Hub
- Source code, training logs, and analysis: [GitHub repo](https://github.com/jujuliaa12/rlhf-compact-llm)
""".strip()

with gr.Blocks(title="rlhf-compact-llm demo") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        prompt_box = gr.Textbox(
            label="Prompt",
            placeholder="Ask anything…",
            lines=3,
        )
    with gr.Row():
        max_tokens_slider = gr.Slider(16, 256, value=128, step=16, label="Max new tokens")
        temp_slider = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
        run_btn = gr.Button("Generate", variant="primary")

    gr.Examples(EXAMPLE_PROMPTS, inputs=prompt_box)

    output_components: list[gr.Markdown] = []
    with gr.Row():
        for h in HANDLES:
            with gr.Column():
                tag = "" if h.available else "  *(unavailable)*"
                gr.Markdown(f"### {h.name}{tag}")
                output_components.append(gr.Markdown())

    run_btn.click(
        fn=compare_all,
        inputs=[prompt_box, max_tokens_slider, temp_slider],
        outputs=output_components,
    )

if __name__ == "__main__":
    demo.queue().launch()
