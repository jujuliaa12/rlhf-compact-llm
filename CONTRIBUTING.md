# Contributing

Thanks for taking a look. Issues, bug reports, and PRs are all welcome.

## Quick start for contributors

```bash
git clone https://github.com/jujuliaa12/rlhf-compact-llm.git
cd rlhf-compact-llm

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -e ".[dev]"           # editable install + dev tools
```

Run the local checks before opening a PR:

```bash
ruff check src tests              # lint
pytest tests/ -q                  # unit tests
python scripts/smoke_test.py      # end-to-end smoke (~5 min)
```

CI runs the same checks on `push` and `pull_request` against `main`.

## What kind of contributions are useful?

In rough priority order:

1. **New experiments** that fit the open questions in [`docs/RESULTS.md`](docs/RESULTS.md) §5 — DPO baseline, cross-dataset comparison, alignment-tax benchmarks (MMLU / GSM8K), LLM-as-judge eval.
2. **Bug fixes** in the training / evaluation code, especially anything that breaks reproducibility.
3. **Compatibility patches** for newer TRL / Transformers / PEFT versions (note that the pinned stack is intentional — see the README "Pinned dependencies" section).
4. **Documentation improvements** — clearer setup steps, more explanation in the engineering notes section, walkthroughs.

## What is *not* in scope

- Major refactors that change the directory layout without a clear motivation.
- Adding 7B+ model support — this repo is intentionally about compact models.
- Replacing TRL with a different RLHF library.

## Code style

- `ruff` for lint. Config in `pyproject.toml`.
- Type hints encouraged; not enforced.
- Functions should have docstrings; modules should have a short module docstring.
- Tests live in `tests/` and follow `test_*.py` naming.

## Commit style

- Imperative, concise: "Fix PPO KL reporting" not "Fixed the KL reporting bug".
- Reference issue numbers in the body when relevant.

## PR checklist

- [ ] `ruff check src tests` clean
- [ ] `pytest tests/ -q` passes
- [ ] If you changed PPO / SFT / reward training code, the smoke test passes
- [ ] README / docs updated if behaviour or interface changed
- [ ] No large binary files added (model weights live in `outputs/models/`, gitignored where appropriate)

## Questions?

Open a [discussion](https://github.com/jujuliaa12/rlhf-compact-llm/discussions) or file an issue.
