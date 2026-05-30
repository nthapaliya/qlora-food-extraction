# QLoRA Fine-Tuning for Structured Food Extraction

> Fine-tuning a 3B instruct model to produce strict JSON from free-form food/drink
> descriptions — on a single 8 GB consumer GPU.

Final project for **CSCI E-222: Foundations of Large Language Models**,
Harvard Extension School, Spring 2026.

[Preview the Notebook Here](qlora-food-extraction.md)
[Full Report Here](report.pdf)

---

## Results

Evaluated on a 213-row held-out test split. A malformed output counts as a miss on
every downstream metric — no partial credit.

| Metric | Base (Qwen2.5-3B-Instruct) | **Fine-tuned** | Delta |
|--------|---------------------------|----------------|-------|
| `json_valid` | 0.972 | **1.000** | +0.028 |
| `schema_ok` | 0.925 | **1.000** | +0.075 |
| `is_food_acc` | 0.798 | **0.972** | +0.174 |
| `tags_f1` | 0.305 | **0.943** | +0.638 |
| `food_items_f1` | 0.665 | **0.861** | +0.196 |
| `drink_items_f1` | 0.875 | **0.949** | +0.074 |
| `exact_match` | 0.258 | **0.624** | +0.366 |

**Trainable parameters:** 29.9M of 3.12B (0.96%).
**Peak VRAM during training:** well under 8 GB on an RTX 3060 Ti.

---

## Overview

The task: given an unstructured English description of a food or drink scene, produce a
JSON object with exactly four keys:

```json
{
  "is_food_or_drink": true,
  "tags": ["fi", "di"],
  "food_items": ["cauliflower florets", "sweet potato wedges"],
  "drink_items": ["white wine"]
}
```

The contract is tight enough to measure cleanly — a missing closing brace is a miss even
if the output is 99% correct.

**Approach:** freeze the base model in 4-bit NF4 quantization and train small LoRA
adapter matrices in higher precision (QLoRA). Only the adapters are updated;
base weights never change.

**Why the base model fails:** it generates tags as long, free-form English phrases instead
of the short two-letter codes the dataset uses. The adapter learns this schema mismatch
from under 1,000 training rows.

---

## Model & training details

| Hyperparameter | Value |
|----------------|-------|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | 4-bit NF4, double quantization |
| LoRA rank | 16, applied to all linear layers |
| LoRA dropout | 0.05 |
| Compute dtype | bfloat16 |
| Learning rate | 2e-4 (cosine, 10 warmup steps) |
| Optimizer | `paged_adamw_8bit` |
| Batch size | 1 (grad accumulation 16) |
| Epochs | 3 |
| Max sequence length | 1,024 tokens |

Loss is computed **only on the assistant (JSON) turn** via completion-only masking —
the model never learns to copy the system prompt.

---

## Dataset

[`mrdbourke/FoodExtract-1k`](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k)
from the Hugging Face Hub. 1,420 rows; 50% food, 50% non-food; split 70/15/15 (train /
val / test).

> **Note:** the `gpt-oss-120b-label` column is Python `repr` of a dict, not JSON —
> `ast.literal_eval` is needed to parse it. 53 rows also store `is_food_or_drink` as a
> string instead of a bool. The notebook corrects both issues before training.

---

## Setup

**Requirements:** Python 3.14+, `uv`, a CUDA-capable GPU with ≥8 GB VRAM
(tested on RTX 3060 Ti). CPU-only is possible but very slow.

```bash
git clone https://github.com/<you>/qlora-food-extraction.git
cd qlora-food-extraction

uv python pin 3.14
uv init --bare
uv add transformers peft bitsandbytes accelerate datasets evaluate matplotlib pandas numpy
uv add --dev jupyterlab
```

---

## Running the notebook

```bash
uv run jupyter lab
```

Open `qlora-food-extraction.ipynb` and run all cells in order. The notebook:

1. Downloads and cleans `FoodExtract-1k` directly from the Hugging Face Hub
2. Runs an EDA section with visualisations
3. Fine-tunes the adapter (≈20 minutes on an RTX 3060 Ti)
4. Saves the adapter to `qlora-foodextract-adapter/`
5. Runs the evaluation harness and prints the results table

> The dataset is small (~1 MB) and downloads automatically. No manual data prep required.

---

## Key findings

The base model's biggest failure mode — generating free-form English tags instead of the
compact two-letter codes — is exactly the kind of schema mismatch that a PEFT adapter
fixes well, even with fewer than 1,000 training rows. `exact_match` nearly triples
(0.258 → 0.624) and `tags_f1` goes from 0.305 to 0.943.

Immediate next steps: iterate LoRA rank in {4, 8, 16, 32}; compare 4-bit QLoRA vs 8-bit
LoRA vs full-precision LoRA; scale to the full `FoodExtract-135k` dataset.

---

## References

- Dettmers et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Hu et al. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Qwen Team (2024). [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
