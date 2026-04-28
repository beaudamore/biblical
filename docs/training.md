# Training

Hyperparameters, environment notes, and the rationale behind each choice for the biblical persona LoRA on Qwen3-14B.

Notebook: [`notebooks/loras/biblical_qwen3_14b_instruct_unsloth_4bit_v2.ipynb`](../notebooks/loras/biblical_qwen3_14b_instruct_unsloth_4bit_v2.ipynb)

---

## 1. Stack

| Component | Choice | Why |
|---|---|---|
| Base model | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | 14B is the sweet spot for 26-persona separation on a single DGX Spark. Pre-quantized 4-bit avoids re-quantization overhead at load. |
| Adapter format | LoRA via PEFT | Adapters are ~100 MB — easy to ship, easy to merge, easy to A/B against a clean base. |
| LoRA library | [Unsloth](https://github.com/unslothai/unsloth) `FastLanguageModel` | 2x faster training and ~50% lower VRAM than vanilla HF + PEFT. Custom kernels for QLoRA. |
| Trainer | TRL `SFTTrainer` + `SFTConfig` | Standard, works with any HF model, integrates cleanly with Unsloth's patches. |
| Optimizer | `adamw_8bit` (bitsandbytes) | Halves optimizer-state memory vs fp16 AdamW with no measurable loss-curve difference at this scale. |
| Precision | `bf16` if supported, else `fp16` | DGX Spark supports bf16; preferred for stability. |

---

## 2. Hyperparameters

```python
BASE_LLM            = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

MAX_SEQ_LENGTH      = 4096

LORA_R              = 32
LORA_ALPHA          = 32
LORA_DROPOUT        = 0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE          = 2
GRAD_ACCUM          = 4              # effective batch = 8
LEARNING_RATE       = 2e-4
TARGET_EPOCHS       = 1
WARMUP_STEPS        = 5
WEIGHT_DECAY        = 0.01
LR_SCHEDULER        = "linear"
SEED                = 3407
```

### Why these values

| Choice | Reasoning |
|---|---|
| `r=32, alpha=32` | 1:1 alpha/r ratio (the modern default — better generalization than the older `alpha = 2r` heuristic). r=32 is enough capacity for 26-voice separation without overfitting on ~10k conversations. |
| All 7 projections targeted | Covers full attention (`q,k,v,o`) **and** MLP (`gate,up,down`). MLP coverage matters for stylistic cadence — voice doesn't live only in attention. |
| `dropout=0` | LoRA-on-top-of-frozen-base is already heavily regularized; explicit dropout slows convergence at this scale without measurable test-loss benefit. |
| `MAX_SEQ_LENGTH=4096` | Long enough for multi-turn ShareGPT conversations + persona system prompts (~600 tokens). 8192 was tested but doesn't help; the data tops out around 3000 tokens per packed sequence. |
| `batch=2 × grad_accum=4` | Effective batch 8. DGX Spark fits batch=4 at 4096 seq len, but 2×4 leaves headroom for the optimizer-state spike during gradient accumulation. |
| `lr=2e-4` | Standard QLoRA SFT rate. Higher (3e-4+) accelerates convergence but causes voice-collapse — the model starts blending personas. |
| `1 epoch` | The voice-quality gate produces clean enough data that 1 pass is enough; 2 epochs over-fits to template phrases (we've validated this with held-out persona splits). |
| `seed=3407` | Unsloth's recommended deterministic seed. Same value across SFT/DPO so adapter merges are bit-identical when re-runs match. |

### Trainer config

```python
SFTConfig(
    dataset_text_field="text",
    max_seq_length=4096,
    packing=False,                 # ← already pre-packed; do NOT re-pack
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=OUTPUT_DIR_ADAPTERS,
    report_to="none",
    dataset_num_proc=1,            # Unsloth stability — multi-proc breaks its kernel patches
)
```

### A note on `packing=False`

The input data is **already** pre-packed into 4096-token chunks with zero padding (handled in cell 10 — *"Format Dataset for Chat Template"*). The pre-pack step:

1. Renders each ShareGPT conversation through Qwen's chat template (`<|im_start|>system ... <|im_end|>` / `<|im_start|>user ...`).
2. Tokenizes.
3. Concatenates conversations end-to-end with a small bridge separator until the buffer reaches `MAX_SEQ_LENGTH`, then flushes one packed example.
4. Repeats. No padding tokens are emitted.

If we left `packing=True` in the trainer, TRL would attempt to repack the already-packed sequences and either truncate them or pad them, both of which corrupt the carefully-arranged turn boundaries.

This is the kind of subtle interaction between Unsloth, TRL, and chat-template handling that's worth calling out explicitly so future maintainers don't flip the flag in good faith.

---

## 3. Hardware

- **Training:** NVIDIA DGX Spark, 128 GB unified memory, single GH200 superchip.
- **Inference / deployment target:** RTX A5000 24 GB via vLLM.

The A5000 24 GB target is the binding constraint. At 4-bit, Qwen3-14B fits in ~8 GB; the LoRA adapter adds ~100 MB; remaining VRAM goes to KV cache for 4096-token context. This works comfortably on a single A5000 — the LoRA architecture choice (no full-weight DPO) is what keeps deployment cheap.

---

## 4. Environment bootstrapping

The notebook's first cell is unusually long because it handles two real environment hazards on the DGX Spark:

### `causal_conv1d` ships broken on aarch64

NGC PyTorch containers for ARM64 ship `causal_conv1d` 1.6.0 **without** the compiled CUDA extension (`causal_conv1d_cuda`). Importing FalconH1 (transitively pulled by transformers) crashes hard. Fix:

```python
_build_env = {
    "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
    "TORCH_CUDA_ARCH_LIST": "12.0;12.1",
}
# pip install --no-binary causal_conv1d -v causal_conv1d
```

The first build takes ~3 min on aarch64; pip caches the wheel after that.

### `unsloth` import order

Unsloth patches transformers and PEFT at import time. **It must be imported before transformers** in the same Python process — otherwise the patches don't take and you get the unpatched (slower, more VRAM-hungry) path silently. The notebook explicitly cleans up any leaked `transformers` / `causal_conv1d` modules before the unsloth import to guarantee correct order:

```python
for _k in list(sys.modules.keys()):
    if "causal_conv1d" in _k or "transformers" in _k:
        sys.modules.pop(_k, None)
```

### NGC PyTorch protection

The cell also verifies `torch.cuda.is_available()` before doing any package work, and refuses to proceed if CUDA torch has been clobbered to a CPU build by a prior pip install. The error message tells the operator to recreate the container — easier to recover from than a half-broken environment.

---

## 5. Outputs

```
data/output/biblical_qwen3_14b_unsloth_4bit_v2/
├── train/                      # SFTTrainer checkpoint dir
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  (~100 MB)
│   ├── training_args.bin
│   └── tokenizer/                  # full tokenizer for vLLM-friendly deploys
└── persona_prompts.json        # the 26 system prompts dumped at end of training
```

`persona_prompts.json` is written by section 9 of the notebook — the persona system prompts that the LoRA was trained against, exported as a runtime contract for inference servers. Without this, every consumer would need to hand-copy persona prompts from the notebook source.

---

## 6. Inference verification (sections 10–11)

After training:

1. **Hot inference** — runs a sanity check using the trained model in-memory. Confirms outputs look right (Paul-style, David-style, etc.) before moving on.
2. **Cold reload** — wipes the in-memory model, reloads the base + LoRA from disk, and reruns inference. This catches the "training worked but the saved adapters are corrupted" failure mode that's surprisingly common with quantized + PEFT combinations.

The cold-reload check is **not optional** — it's the only way to verify that what you save is what you'll deploy.

---

## 7. Deployment

The trained LoRA is deployed via vLLM with the LoRA adapter passed at server startup. Approximate command:

```sh
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --enable-lora \
  --lora-modules biblical=/path/to/output/.../train \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Then chat completions request the adapter by name:

```json
{ "model": "biblical", "messages": [...] }
```

The persona is selected by setting the `system` message to the appropriate prompt from `persona_prompts.json`.

---

## 8. DPO training (separate run)

When DPO data is generated (see [dpo-data-generation.md](dpo-data-generation.md)), DPO training runs **on top of** the SFT LoRA — not from base — using `trl.DPOTrainer`. Recommended config:

```python
DPOConfig(
    beta=0.1,
    learning_rate=5e-7,           # ~400x lower than SFT
    num_train_epochs=1,
    max_length=2048,
    max_prompt_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
```

The DPO LoRA can be merged with the SFT LoRA before deploy, or stacked at runtime.

---

## 9. Sister training notebooks

| Notebook | Target |
|---|---|
| `augustine_qwen3_14b_instruct_unsloth_4bit.ipynb` | Augustine-only LoRA, same hyperparameters |
| `liguori_qwen3_14b_instruct_unsloth_4bit.ipynb`   | Liguori-only LoRA, same hyperparameters |
| `biblical_qwen3_5_27b_instruct_unsloth_4bit.ipynb` | 27B variant (Qwen3-30B-A3B, requires bigger hardware) |
| `notebooks/old/*.ipynb` | Historical Llama / Mistral / Qwen2.5 base-model experiments. Kept for diff comparison; superseded by Qwen3 14B 4-bit. |

All sisters share the same NGC env-setup logic and dataset-loader skeleton. The hyperparameter block is the divergence point.
