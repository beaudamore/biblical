# Gemma 4 12B notebooks: port notes

Written 2026-11-06 after a read-only review. Nothing in the two notebooks has been changed yet.
For a later session that will bring them up to date and produce a model for the game at
`~/Source/unreal-sandbox`. Items marked "not verified" were not checked against a running system.

- SFT: `notebooks/loras/biblical_gemma4_12b_unsloth_4bit.ipynb`
- DPO: `notebooks/loras/biblical_gemma4_12b_dpo_unsloth_4bit.ipynb`

## Status

- The only 12B runs are from 2026-07-21/22: SFT (544 steps, final loss 0.376, packing on, loss over
  all tokens) and DPO (446 steps), both under the output name `..._4bit_mobile`.
- Copies on the Mac: `~/My Drive/biblical-gemma4-12b-dpo-lora` (complete, 524.6 MB, r=32, 656
  tensors, all in the language model) and `~/My Drive/biblical-gemma4-12b-sft-lora` (no weights
  file; it does hold `persona_system_prompts.json`).
- The SFT notebook has been rewritten since (response-only loss, tower-scope assertion, new output
  name) and has never been run in that form: every cell has `execution_count: null`.
- Test model built 2026-11-06: `unreal-sandbox/Models/personas/gemma-4-12b-biblical-dpo-Q4_K_M.gguf`.
  Built on the Mac by merging the July DPO adapter tensor by tensor (W + B@A, scale 1.0) into the
  Google snapshot on evotwin, then `convert_hf_to_gguf.py` (bf16) and `llama-quantize` Q4_K_M from
  `~/Source/llama.cpp`. Against the stock game model: 328 tensors differ, 339 are byte-identical,
  chat template identical. It has not been loaded or tested.

## Decisions Beau has not made

1. Merged GGUF or runtime LoRA for the game. Merged is a drop-in but changes every NPC; a LoRA can
   be per character but needs new C++ in the game.
2. Pin the July library versions, or port the DPO notebook to current TRL.
3. Whether to add game-shaped training rows (short JSON replies with speech tags).

## Work list, in order

### 1. SFT/DPO name contract (SFT cell 2, DPO cell 4 and cell 0 header)

SFT sets `MODEL_NAME_BASE = "biblical_gemma4_12b_unsloth_4bit"`. DPO sets
`SFT_MODEL_NAME_BASE = "biblical_gemma4_12b_unsloth_4bit_mobile"` and writes `..._mobile_dpo`.
If the July `_mobile` folder is still on the Spark, DPO silently continues the old adapter;
otherwise it stops with FileNotFoundError. Fix: one name in both notebooks and in the DPO header.

### 2. Unpinned installs (SFT cell 4, DPO cell 2)

Both run `pip install -U unsloth trl accelerate datasets bitsandbytes` and install transformers
and peft from git main. The July runs printed: Unsloth 2026.7.4, transformers 5.15.0.dev0,
PEFT 0.19.2.dev0, TRL 0.24.0, torch 2.10.0a0 nv25.11.

Looked up 2026-11-06 (not tested here):
- Latest Unsloth is 2026.11.3 and caps `trl<=1.15.0`, `transformers<=5.17.0`.
- TRL 1.x rewrote the DPO trainer. v1.13: DPO `padding_free` unavailable. v1.14: `trl.losses`
  removed. v1.15: fused LM head scoring, `use_liger_kernel` deprecated. v1.17: a `PeftModel`
  passed as `ref_model` is rejected. https://github.com/huggingface/trl/releases
- Current DPO docs: https://huggingface.co/docs/trl/main/en/dpo_trainer

TRL 0.24 names the DPO notebook depends on (cells 9 and 11). Which survive in 1.x: not verified.
- `DPOConfig(max_prompt_length=..., precompute_ref_log_probs=False)`
- the `model.config.model_type` swap and `trainer.is_vision_model = False`
- `trainer.data_collator`, `trainer.compute_ref_log_probs(batch)`, `trainer.accelerator`
- `trainer._precomputed_train_ref_log_probs = True` and the `ref_chosen_logps` /
  `ref_rejected_logps` dataset columns

### 3. DPO reference is the stock base, not the SFT model (DPO cells 7, 9, 11)

Cell 7 resumes the SFT adapter with `PeftModel.from_pretrained(..., is_trainable=True)`; cell 9
passes `ref_model=None`. In TRL 0.24 `null_ref_context` then disables the adapter for the
reference pass. Readable copy: `loras/unsloth_compiled_cache/UnslothDPOTrainer.py`, lines 1596-1608.

Evidence in the saved outputs (step 5 of each run):

| Run | Loss | Reward accuracy | Reward chosen / rejected |
|---|---|---|---|
| Gemma 4 12B | 0.0695 | 0.975 | 35.80 / 11.75 |
| Qwen3.8 27B (production) | 0.000626 | 1.0 | 11.41 / -1.58 |
| Qwen3.5 4B | 0.000000 | 1.0 | 25.48 / -4.18 |
| Qwen3 14B v2 | 0.000081 | 1.0 | 18.07 / 2.50 |

A correct reference would start near loss 0.69 with rewards near 0. The 12B run ends at 0.000000
(mean 0.0237). In its saved eval (cell 17) the scripture-fabrication prompt failed: the model said
angels taught Paul justification.

Fix routes:
- Under TRL 0.24: load the SFT adapter a second time under another name and set
  `ref_adapter_name` (and `model_adapter_name`) in `DPOConfig`. Still one shippable adapter.
- Under current TRL: the docs say `ref_model=None` means the state before DPO starts. Not tested.

The reference-logprob cache fingerprint (cell 11) must change with the fix, or stale shards load.

### 4. No export after DPO

The DPO notebook ends at the eval cell. The only GGUF cell is SFT cell 27 (Unsloth
`save_pretrained_gguf`), which exports the SFT-only adapter and has never run on 12B.

Port section 13 of `notebooks/loras/qwen-3.5-4b/biblical_qwen3_5_4b_dpo_unsloth_4bit.ipynb`
(cell 27: CPU merge with PEFT, `convert_hf_to_gguf.py`, `llama-quantize`, `convert_lora_to_gguf.py`).
Gemma changes:
- End of turn is `<turn|>`. Check which EOS id the converter writes before forcing anything.
- The Qwen think-template patch does not apply. Instead check the empty thought-channel priming
  (see the game section).
- Also port the cold-reload audit (4B cell 22) and the `complete.json` sentinel.

Alternative with no GPU: the Mac-side tensor merge used on 2026-11-06 (about four minutes end to
end, about 24 GB RAM, 44 GB scratch). The script was session scratch and is not in a repo.

Unsloth export notes looked up 2026-11-06 (not tested): `merged_16bit` merges into the 16-bit
checkpoint (PR #14823); `save_method="lora"` exists (PR #13915); an adapter loaded with
`PeftModel.from_pretrained` used to export the unmerged base silently (issue #11698, closed
2026-09-26).

### 5. SFT cold-reload cell 25

`tokenizer2(text, return_tensors="pt")` passes the string positionally to a Processor, which the
notebook's own comments (cells 10 and 23) say binds to `images`. Unwrap the tokenizer or use
`text=`. The 4B SFT notebook cell 24 has the working form.

### 6. Smaller items

- DPO cell 4 sets `PYTORCH_CUDA_ALLOC_CONF=...max_split_size_mb:256` and
  `set_per_process_memory_fraction(0.55)`. The 4B DPO notebook (cell 2) documents both as an OOM
  cause on the Spark. The July 12B run did complete with them.
- DPO cell 7 forces `pad_token = eos_token` (`<turn|>`, id 106). SFT cell 10 does not. Effect on
  the loss: not verified.
- Stale text: SFT cell 16 says lr 1e-4 (config is 2e-4); SFT cell 17 prints "Packing: enabled"
  beside `packing=False`; SFT cell 20 says "Gemma 4 E4B"; DPO cell 7 comment says the base loads
  without 4-bit while the code loads 4-bit.
- DPO eval (cell 17) samples with Qwen values (0.7 / 0.8 / 20). SFT uses 0.7 / 0.95 / 64.
- The tower markdown (SFT cell 14, DPO cell 6) says 12B has vision and audio towers. Google's
  config describes it as encoder-free. Not checked against the loaded module tree. The scope
  assertion is harmless either way; keep it.
- No `complete.json` check before DPO starts (the 4B pair has one).
- The SFT file name has no `_sft_` tag, unlike the newer pairs. A rename touches the README.

## What the game needs from the model

Read in `~/Source/unreal-sandbox` on 2026-11-06:
- llama.cpp linked in-process, vendored at commit 5047748f1. No server, no `--jinja`.
- No LoRA loading code: `MindEngine.cpp` `LoadModel` calls only `llama_model_load_from_file`.
- The Gemma turn format is written by hand (`<|turn>role\n ... <turn|>\n`, ends `<|turn>model\n`,
  no thinking block) and is used only when the GGUF's embedded template contains `<|turn>`.
- Replies are grammar-forced JSON `{"say": "...", "flags": []}`, asked to stay under 60 words,
  `MaxReplyTokens=200`, with speech tags such as `[sigh]` inside `say`. Context is 8192 tokens.
- Training answers here: 11,462 of them, median 1,604 characters (about 290 words), prose, no
  JSON, no tags. Replies cut off before the JSON closes are a risk; not measured.
- Still open: Google's 12B template primes an empty `<|channel>thought\n<channel|>` when thinking
  is off; the game's format does not. Print one rendered training row on the Spark and compare.

## Reference facts for Gemma 4 12B

- Ids: `google/gemma-4-12B-it`, `unsloth/gemma-4-12b-it`. No `-bnb-4bit` repo for 12B.
- `model.safetensors` has the same sha256 in both repos and in the evotwin snapshot
  (5a84cb31...bff18d), verified 2026-11-06 through the Hugging Face API. Config, tokenizer config
  and chat template files differ between the two repos.
- Licence: Apache-2.0 per Hugging Face metadata. Google's own licence text: not verified.
- Architecture `Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`), dense.
- Markers: `<|turn>user\n`, `<|turn>model\n`, end of turn `<turn|>`. Unsloth docs give these as the
  `train_on_responses_only` parts; the notebook auto-detects them (ran on E4B, never on 12B).
- Card sampling: temperature 1.0, top_p 0.95, top_k 64.
- Unsloth docs: https://unsloth.ai/docs/models/gemma-4 and .../gemma-4/train
- `~/Source/llama.cpp/conversion/gemma.py` registers `Gemma4UnifiedForConditionalGeneration`, so
  `convert_lora_to_gguf.py` has a path for 12B adapters. `convert_hf_to_gguf.py` was run on 12B
  (2026-10-30 and 2026-11-06); the LoRA converter was not.
