# Biblical LoRA training — project notes for Claude

Written 2026-11-06 from a read of the repo. Every line here was checked against a file or a
notebook output on that date unless it is marked (README claim). Treat the README and `docs/`
as claims: several are stale (list at the bottom).

## What this repo is

Jupyter notebooks that turn public-domain biblical and patristic texts into training data, then
train persona LoRA adapters (Unsloth, 4-bit QLoRA, SFT then DPO) on several base models. One
adapter teaches 26 personas; the system prompt selects the voice.

- Production today (README claim): `biblical_dpo` on Qwen3.8-27B, served by vLLM on the DGX Spark.
- Second consumer since 2026-11: the Unreal game at `~/Source/unreal-sandbox` (section below).

## Where things run

- This Mac clone is for reading and editing only. There is no CUDA here and nothing in this repo
  is to be executed from a session. After editing a notebook, the only allowed check is that the
  file still parses as JSON.
- Training runs on the DGX Spark (GB10, 128 GB unified memory) in the `unsloth-notebook`
  container. Host `/home/spark/projects/training` is mounted at `/workspace/training`, so this
  repo is `/workspace/training/biblical` there. Beau runs the notebooks himself.
- `output/` and `data/training-data/` are gitignored. Adapters, checkpoints and GGUFs exist only
  on the Spark (some copies in `~/Source/LORA_ADAPTERS`). `data/training-data/` does exist on
  this Mac; `output/` does not.
- `data/source-raw/` and `data/source-clean/` ARE tracked in git (the README says `data/` is
  gitignored; only `training-data/` is).
- `loras/` at the repo root is an ignored leftover (`unsloth_compiled_cache/`,
  `.ipynb_checkpoints/`). It is useful for one thing: `UnslothDPOTrainer.py` there is a readable
  copy of the TRL DPO trainer the runs used.

## Pipeline, by file

1. `data/scripts/clean_source_data.py`: `data/source-raw` -> `data/source-clean`.
   TRAP: it deletes the whole `source-clean` folder first and never rebuilds
   `source-clean/full_biblical_data/` (the 26 persona inputs). Recoverable from git, but do not
   run or suggest running it without saying so.
2. `notebooks/datagen/biblical_datagen_v2_sft.ipynb` (OpenRouter, `qwen/qwen3-235b-a22b-2507`)
   -> `data/training-data/biblical_persona_v2/biblical_personas_combined_sharegpt.jsonl`
   (4,349 conversations since the 2026-11-10 wrong-person removal, 4,352 before; 26 personas, ShareGPT `conversations` + `data_type`, 3 to 9 turns,
   system prompts 1,314 to 1,593 characters).
3. `notebooks/datagen/biblical_datagen_v2_dpo-local-sft.ipynb` (since 2026-11-11; run AFTER SFT, with the SFT
   adapter on local vLLM) -> `.../biblical_personas_v4_dpo_local_sft.jsonl` (1,673 pairs, sources
   `voice_drift` and `shallow_platitude`). The original, now in `notebooks/datagen/archive/`, was
   `biblical_datagen_v2_dpo.ipynb`
   -> `.../biblical_personas_v2_dpo.jsonl` (3,564 pairs; `chosen`/`rejected` are
   `[system, user, assistant]` role/content lists plus `source` and `persona`; three sources:
   `voice_drift`, `scripture_fabrication`, `shallow_platitude`).
4. `notebooks/loras/..._sft_...ipynb` (or `_instruct_`, or no tag on Gemma 12B)
   -> `output/<MODEL_NAME_BASE>/lora_adapters` plus `persona_system_prompts.json`.
5. `notebooks/loras/..._dpo_...ipynb` resumes that same adapter and keeps training it
   -> `output/<MODEL_NAME_BASE>_dpo/lora_adapters`. One adapter carries both stages.

## Notebook conventions

- `MODEL_NAME_BASE` in the SFT notebook is the contract: the DPO notebook's
  `SFT_MODEL_NAME_BASE` must equal it. Change both or neither.
- Usual settings: r=32, alpha=32, all attention + MLP projections, lr 2e-4, 1 epoch, batch 2x4
  for SFT; lr 5e-6, beta 0.05 (0.1 on the 4B), batch 1x8 for DPO.
- TEMPLATE for new work (2026-11-11): `notebooks/loras/qwen_38_27b/` (SFT + DPO) with
  `notebooks/datagen/biblical_datagen_v2_sft.ipynb` and `biblical_datagen_v2_dpo-local-sft.ipynb`.
  The DPO notebook is written for TRL 1.15: reference = frozen `ref` copy of the SFT adapter
  (asserted), conversational dataset format, beta 0.25, 100-step cap, held-out check every 25
  steps with a stop rule, Save cell picks the last good checkpoint. Copy from there.
- Everything else was moved on 2026-11-11 to `notebooks/loras/archive/`,
  `notebooks/datagen/archive/`, `output/archive/` and
  `data/training-data/biblical_persona_v2/archive/`. Archived notebooks still point at the old
  locations. Paths elsewhere in this file that name those notebooks or folders now need `archive/`.
- Archived `notebooks/loras/archive/qwen-3.5-4b/` still has two things the template lacks: a
  cold-reload audit of the saved safetensors, and section 13 of its DPO notebook, a GGUF export
  that works (CPU merge with PEFT, then llama.cpp's own `convert_hf_to_gguf.py`,
  `llama-quantize` and `convert_lora_to_gguf.py`). Its DPO cells are TRL 0.24.
- That 4B notebook also records why `PYTORCH_CUDA_ALLOC_CONF=...max_split_size_mb:256` and
  `set_per_process_memory_fraction` must NOT be set on the Spark (reproducible OOM). Older
  notebooks, including Gemma 12B DPO, still set both.
- Multimodal bases (Gemma 4, Qwen3.5+) return a Processor, not a tokenizer. Unwrap it, or call
  it as `tokenizer(text=...)`; a positional string binds to `images`.
- Gemma 4 towers reuse the language model's leaf names, so `finetune_vision_layers=False` plus
  the scope assertion after `get_peft_model` is load-bearing. Keep both.

## Editing notebooks

- Cell outputs are the only run record (the README results table is copied from them). Never
  clear or regenerate outputs. Edit cell source only.
- A notebook with `execution_count: null` everywhere has not been run in its current form. Say
  so; do not describe it as working.
- Printed strings and markdown drift from the config (for example a "Packing: enabled" print
  beside `packing=False`). When a value changes, grep the notebook for every mention.
- Notebooks cite `docs/dgx_spark_gb10_quirks.md`, `docs/multimodal_and_hybrid_base_models.md`
  and `docs/audit_adapters.py`. Those live in the parent `training/docs/` on the Spark and are
  not on this Mac. Do not quote them from memory.
- Every change that touches a notebook's behaviour, a file name or the layout updates
  `README.md` (results table, Repo Layout tree) and the matching `docs/*.md` in the same pass.
- `docs/training-history.md` is the public running log (started 2026-11-10). When something is
  tried, measured or overturned, add a dated entry in the format at the bottom of that file.
  Never rewrite old entries; no private details (IPs, accounts, personal paths beyond the repo).

## The game consumer: `~/Source/unreal-sandbox`

Checked 2026-11-06 in that repo (file references are in that repo):

- llama.cpp is linked into the game (static, Metal), vendored at commit 5047748f1
  (`ThirdParty/llama.cpp/VENDORED.txt`). No server, no `--jinja`.
- Stock model: `Models/mind/gemma-4-12b-Q4_K_M.gguf`. Since 2026-11-06 `ModelFile` in
  `Config/DefaultGame.ini` points at the persona test model in `Models/personas/` (below).
  Its README says this is Google's official Gemma 4 12B instruct, converted locally.
- There is no LoRA loading code. `MindEngine.cpp` `LoadModel` calls only
  `llama_model_load_from_file`. A merged GGUF is a drop-in; a LoRA GGUF needs new C++.
- The Gemma turn format is written by hand in `MindEngine.cpp` and is used only when the GGUF's
  embedded chat template contains `<|turn>`. A merged GGUF must keep the template.
- Every reply is grammar-forced JSON `{"say": "...", "flags": []}`, asked to stay under 60 words,
  `MaxReplyTokens=200`, with speech tags like `[sigh]` inside `say`. The training data here is
  free prose (11,462 answers, median 1,604 characters, roughly 290 words), with no JSON and no
  tags.
- One model serves every NPC. Since 2026-11-06 the three NPCs are Peter (gatekeeper, was
  Barnabas), David (trader at the gate, was Wren) and Job (charcoal burner, was Bartholomew).
- A character whose name matches `prompts/personas/<name>.md` in the game repo gets that file as
  the start of its system prompt, word for word (`DialogueClient.cpp` `Configure`). Those 26 files
  are the trained system prompts; if training prompts change, the game copies must be replaced.
- Test model built 2026-11-06: `Models/personas/gemma-4-12b-biblical-dpo-Q4_K_M.gguf`, the
  2026-07-21/22 `_mobile_dpo` adapter (copy in `~/My Drive/biblical-gemma4-12b-dpo-lora`) merged
  into the Google snapshot on evotwin (`model_cache/hf-cache/models--google--gemma-4-12B-it`),
  whose `model.safetensors` has the same sha256 as `unsloth/gemma-4-12b-it`. Merged tensor by
  tensor on this Mac, then `convert_hf_to_gguf.py` and `llama-quantize` from `~/Source/llama.cpp`.
  The 26 trained system prompts are in the game repo's `prompts/personas/`.
- Per-character KV state is saved under `Saved/Mind/` and must be deleted when the model changes.

## Gemma 4 12B facts (looked up 2026-11-06; re-check before relying on them)

- Ids: `google/gemma-4-12B-it` (official), `unsloth/gemma-4-12b-it` (what the notebooks load).
  There is no `-bnb-4bit` repo for 12B; `load_in_4bit=True` quantizes on load. Whether the two
  repos hold identical weights was not checked; it matters only for a LoRA applied to the game's
  Google-derived GGUF.
- Hugging Face metadata lists the licence as Apache-2.0. The game's `docs/credits.md` still says
  "verify (Gemma Terms of Use)".
- Architecture `Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`), dense. Google's config
  describes it as encoder-free (no separate vision or audio tower), which conflicts with the
  tower markdown in the 12B notebooks. Not checked against the loaded module tree.
- Turn markers `<|turn>user\n`, `<|turn>model\n`, end of turn `<turn|>`. With thinking off,
  Google's 12B template primes an empty `<|channel>thought\n<channel|>` block; the game's
  hand-written format does not. Compare one rendered training row with `MindEngine.cpp` before
  trusting a fine-tune in the game.
- Card sampling: temperature 1.0, top_p 0.95, top_k 64.
- The only recorded 12B runs (2026-07-21) used Unsloth 2026.7.4, transformers 5.15.0.dev0,
  PEFT 0.19.2.dev0, TRL 0.24.0. The notebooks install with `pip -U` and from git main, so a run
  today gets TRL 1.x, where the DPO trainer was rewritten. The DPO notebooks' reference-logprob
  cache cell uses TRL 0.24 internals.
- `~/Source/llama.cpp` registers `Gemma4UnifiedForConditionalGeneration` in
  `conversion/gemma.py`, so `convert_lora_to_gguf.py` has a path for 12B adapters. Not run.

## Open items found 2026-11-06 (delete each when fixed)

- Gemma 4 12B work list with cell numbers, evidence and fixes: `docs/gemma4-12b-port-notes.md`.

- Gemma 12B: SFT writes `biblical_gemma4_12b_unsloth_4bit`, DPO reads
  `biblical_gemma4_12b_unsloth_4bit_mobile`. The `_mobile` adapter on the Spark is the
  2026-07-21 run (packing on, loss over all tokens). The current SFT notebook has never been run.
- Gemma 12B DPO has no cold-reload audit and no GGUF export; the only export cell is in the SFT
  notebook, uses Unsloth's `save_pretrained_gguf`, and has never been executed on 12B.
- Every DPO notebook passes `ref_model=None` with a resumed adapter, so TRL uses the stock base
  (adapter disabled) as the reference, not the SFT model. All recorded runs start at loss near 0
  with reward accuracy 0.97 to 1.0 from step 5.
- `adam_eve` and `church_fathers` datagen notebooks set `API_KEY = "${DEEPINFRA_API_KEY}"`, a
  literal string Python does not expand.
- Stale docs: `docs/training.md` (14B era: A5000 target, DPO lr 5e-7, wrong adapter path),
  `docs/dpo-data-generation.md:221`, `docs/data-pipeline.md:5` (pysbd), `docs/sft-data-generation.md`
  (row and exemplar counts), README lines on the serving base id, E2B, safety adapters, and the
  count of base models.
