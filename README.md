# Biblical Persona LoRA — Multi-Voice Scripture Fine-Tuning

![biblical — Bringing Scripture into the Digital Age](docs/screenshots/banner.jpg)

## 🚀 Live in Production

> **The LoRA is deployed and publicly available at [chat.crossandfaith.com](https://chat.crossandfaith.com)** (Google sign-in) — landing page at **[crossandfaith.com](https://www.crossandfaith.com)**.

This is **not a research artifact**. It is a working multi-tenant deployment with real Google OAuth sign-up, production sampling, and months of real user conversations. The trained LoRA is hot-attached to **vLLM** behind **Open WebUI**, so users can pick the `biblical` model directly or use Open WebUI's compare-models feature to A/B against the bare base model on the same prompt.

The adapter in production today is `biblical_dpo`, trained on **Qwen3.8-27B** and served over an NVFP4 quantization of the same base on an NVIDIA DGX Spark. The pipeline has been retrained on eight base models since the first Qwen3-14B release; see [Base models and shipped adapters](#base-models-and-shipped-adapters).

> Side-by-side example from the production UI on the same prompt this README documents — the `biblical` LoRA gives Pauline first-person testimony ("*I have seen the Lord, and by His grace I bear witness...*") while base `Qwen3-14b` gives an encyclopedia-style answer ("*The early apostolic communities, particularly those described in the New Testament...*"). Full text in [`docs/comparisons/early-apostolic-communities.md`](docs/comparisons/early-apostolic-communities.md).

---

## How the LoRA reaches users — two-layer architecture

This repo is **the model-layer half** of the production system. The runtime layer that orchestrates multi-persona conversations on top of these adapters is the open-source [Circle of Speakers pipeline](https://github.com/beaudamore/circle-of-speakers-pipeline).

Each persona LoRA is registered as a tagged Open WebUI model (`biblical`, `stoic`, etc.). The **Dynamic Circle of Speakers v2** pipeline filters the model registry by tag, picks which speaker(s) should respond next via an **intelligent speaker-ordering model**, injects per-speaker introduction and user-profile-aware prefixes, and streams responses back through Open WebUI.

```mermaid
flowchart TD
    U[User<br/>signed in via Google] --> OW["Open WebUI<br/>chat.crossandfaith.com"]
    OW --> CoS["<b>Dynamic Circle of Speakers v2</b><br/>tag-filtered multi-speaker pipeline<br/>(circle-of-speakers-pipeline repo)"]

    CoS --> ORD["Intelligent Speaker Ordering<br/>(LLM picks who responds, in what order)"]
    CoS --> INTRO["Dynamic introductions +<br/>user-profile-aware prefixes"]

    ORD --> V["vLLM<br/>--enable-lora<br/>(adapter hotswap)"]
    INTRO --> V

    V --> L1["<b>biblical</b> LoRA<br/>26 voices · THIS REPO"]
    V --> L2["<b>augustine</b> / <b>liguori</b><br/>pipelines · THIS REPO"]
    V -. sister circles .- LX["stoic / hr / legal LoRAs<br/>(other circles, same framework)"]

    L1 --> V
    L2 --> V
    V --> CoS
    CoS --> OW
    OW --> U

    classDef thisrepo fill:#f4d896,stroke:#8a6f3f,color:#000;
    classDef external fill:#e0e8f0,stroke:#5a7a9a,color:#000;
    classDef pipeline fill:#fdecc8,stroke:#a07a2a,color:#000;
    class L1,L2 thisrepo
    class CoS,ORD,INTRO pipeline
    class OW,V,LX external
```

| Layer | What it is | Where it lives |
| --- | --- | --- |
| **Model layer** | Trained LoRA adapters + data + training pipeline that produced them | **This repo** |
| **Runtime layer** | **Dynamic Circle of Speakers v2** — multi-speaker Open WebUI pipeline with tag-filtered speaker selection, intelligent-ordering model for turn-taking, dynamic per-speaker introductions, user-profile-aware response prefixes, and streaming vLLM passthrough. Same framework powers sister circles for `stoic`, `hr`, `legal` domains. | [circle-of-speakers-pipeline](https://github.com/beaudamore/circle-of-speakers-pipeline) · [writeup at damore.ai](https://www.damore.ai/blog/dynamic-circle-of-speakers-v2) |
| **Surface layer** | Open WebUI UI: model picker, side-by-side compare, Google OAuth, per-speaker profile images served from `images.crossandfaith.com` | [chat.crossandfaith.com](https://chat.crossandfaith.com) |
| **Inference layer** | vLLM with `--enable-lora` for hot-swap between adapters at request time. Today: Qwen3.8-27B NVFP4 on the DGX Spark serving `biblical_dpo` alongside the safety-classifier adapters. Reference compose files per base model in [`compose/`](compose/). | OSS dependency |

The framework is **persona-agnostic** — the biblical adapters from this repo plug into the same pipeline that runs the stoic, HR, and legal circles. The model layer (this repo) and the runtime layer (Circle of Speakers) are intentionally decoupled so additional persona families can be added without touching the orchestrator. A dedicated cast-selection LoRA to replace the ordering model's prompt-only routing is designed in the orchestration-selector project.

More technical writeups at **[damore.ai/blog](https://www.damore.ai/blog)**.

---

## What it is

A two-stage **QLoRA** fine-tune (SFT, then DPO continuing the same adapter) that teaches a base model to speak in **26 distinct biblical voices**, with separate pipelines for **Augustine** and **Alphonsus de Liguori**. First built on **Qwen3-14B**, since retrained on Qwen3.5-27B, Qwen3.6-27B, **Qwen3.8-27B** (production), Qwen3.5-4B and Gemma 4 (12B, E2B, E4B) for mobile. Built on **Unsloth** with 4-bit quantization and trained on an NVIDIA **DGX Spark** (128 GB unified memory).

The pipeline pairs **persona-conditioned Q&A generation** (with a voice-differentiation quality gate that fails closed on >30% template contamination) with **raw-text continuation augmentation** to teach prose cadence, then layers a **DPO** dataset across **three engineered rejection strategies** — voice drift, scripture fabrication, and shallow platitude. Includes a regex-based per-source data-cleaning pipeline, deployment notes, and an **LLM-as-judge** evaluation showing the LoRA scoring **+12** over the base model on a 6-dimension voice-fidelity rubric.

> 📄 **[See the side-by-side LoRA-vs-base comparison (PDF) →](docs/comparisons/early-apostolic-communities.pdf)**
> The same prompt sent to base Qwen3-14B and to the LoRA. Scoring rubric and rationale per dimension.

When you ask **Paul** a question, he builds long theological chains with autobiographical anchors (Damascus road, his thorn, his shipwrecks). Ask **Amos** the same question — you get a blunt one-liner with a farming metaphor. Ask **David** — psalm-cadence parallelism and raw emotional vulnerability. The voice switches based on the system prompt; the model learns to switch with it.

---

## Skills Demonstrated

| Domain | Specifics |
| --- | --- |
| **Fine-tuning / PEFT** | LoRA, QLoRA (4-bit NF4), `r=32 / α=32` across all 7 attn+MLP projections, TRL `SFTTrainer` + `DPOTrainer` continuing one adapter, gradient accumulation, pre-packed sequences (handling the `packing=False` interaction with TRL); the same recipe carried across Qwen3, Qwen3.5/3.6/3.8 and Gemma 4 bases |
| **Multimodal base handling** | Qwen3.5+ and Gemma 4 ship vision/audio towers and MTP heads. The adapters are scoped to the language model with `finetune_vision_layers=False` and an assertion that nothing else was adapted; every saved adapter is audited from its safetensors keys before serving |
| **Synthetic-data generation** | Persona-conditioned Q&A with KJV exemplar grounding; banned-opener retry loop; cross-persona 4-gram opener uniqueness gate; three rejection-strategy DPO with per-persona caps |
| **Data engineering** | Per-source regex cleaner (Project Gutenberg / sacred-texts.com / ChristianFOSS / Liguori imprimaturs); sentence-aware chunking (pySBD); `tiktoken` token-aware chunking for continuation tasks; ShareGPT format |
| **Evaluation** | LLM-as-judge with 6-dimension rubric; voice-differentiation quality gate; cold-reload adapter sanity check; per-persona contamination metrics |
| **Infra / Ops** | NGC PyTorch container debugging on aarch64 (`causal_conv1d` source-build with `CAUSAL_CONV1D_FORCE_BUILD=TRUE`); strict Unsloth import-order discipline; unified-memory discipline on GB10; vLLM LoRA hot-attach over NVFP4 and FP8 bases; GGUF export for on-device models |
| **Production deployment** | **Live multi-tenant chat** at [chat.crossandfaith.com](https://chat.crossandfaith.com); Open WebUI surface; Google OAuth sign-up; vLLM serving with `--enable-lora`; months of real user traffic |
| **Tooling built** | OpenAI-compatible API generation (OpenRouter / local vLLM swap); reproducible `regenerate.sh` for evaluation PDFs; per-base-model vLLM compose files; workspace-wide adapter audit script |

---

## Headline Numbers

| Item | Value |
| --- | --- |
| **Personas** | 26 (KJV-grounded) + Augustine + Liguori pipelines |
| **SFT examples** | 4,352 ShareGPT conversations (2,612 Q&A + continuation) → 1,535 packed 4096-tok chunks on the 14B run |
| **SFT examples, current file** | 4,349 conversations since 2026-11-10: 155 questions addressed to the wrong figure (154 of them Moses asked about Abraham, Noah or Adam) were removed with their answers. Also on 2026-11-10, over-repeated answer openers were fixed: 605 stock first lines cut and 2,465 first sentences rewritten, taking the highest opener share in any persona from 45% to 7%. Every adapter in the table below was trained on the 4,352-row file as it was before these fixes. See [docs/training-history.md](docs/training-history.md) |
| **SFT blend** | 60% Q&A / 40% raw-text continuation (stray space before punctuation removed from the continuation text on 2026-11-10; adapters listed below were trained before the fix, see [docs/data-pipeline.md](docs/data-pipeline.md#space-before-punctuation-fix-2026-11-10)) |
| **DPO pairs** | 3,564 across 3 engineered rejection strategies |
| **Augustine / Liguori sets** | 2,726 / 844 ShareGPT conversations |
| **Source corpus** | KJV per-persona texts + Project Gutenberg + sacred-texts.com + ChristianFOSS |
| **Generator model (Q&A)** | Qwen3-235B-A22B-2507 (OpenRouter); Qwen-2.5-7B for cheap rejected answers in DPO |
| **Production base model** | `unsloth/Qwen3.8-27B` (training) → `RadixArk/Qwen3.8-27B-NVFP4` (serving) |
| **Adapter shape** | LoRA r=32 / α=32 across all 7 attn+MLP projections; 128 M trainable params (0.86%) on 14B, 608 MB fp32 on the 27B bases |
| **Training hardware** | NVIDIA DGX Spark (GB10 superchip, 128 GB unified) |
| **Documented run (14B v2)** | ~2 h 39 min wall-clock; 192 steps; final loss **1.31** |

---

## Base models and shipped adapters

Every base below has a completed SFT adapter; all but the two smallest Gemma mobile builds
also have a DPO adapter that continues it. Values are read from each run's `adapter_config.json`
and `trainer_state.json`, not from the notebooks. Adapters live under `output/` (gitignored).

| Base model | SFT date | SFT steps · loss (first → last) | DPO date · steps | r / α | Notes |
| --- | --- | --- | --- | --- | --- |
| `unsloth/Qwen3-14B-unsloth-bnb-4bit` (v2) | 2026-06-13 | 192 · final 1.31 | 2026-06-13 · 446 | 32 / 32 | The documented run below; LLM-judge comparison |
| `cyberenchanter/Qwen3.5-27B-bnb-4bit` | 2026-06-12 | 195 · 1.77 → 1.10 | 2026-06-13 · 446 | 16 / 16 | First 27B |
| `unsloth/Qwen3.6-27B` | 2026-07-29 | 544 · 1.58 → 1.24 | 2026-07-29 · 446 | 32 / 32 | Served on the Qwen3.6 NVFP4 stack until the 3.8 move |
| **`unsloth/Qwen3.8-27B`** | 2026-08-15 | 544 · 1.71 → 1.36 | 2026-08-15 · 446 | 32 / 32 | **Production** (`biblical_dpo`) |
| `unsloth/Qwen3.5-4B` | 2026-08-29 | 544 · 2.14 → 1.69 | 2026-08-31 · 446 | 32 / 32 | Small/mobile; GGUF exported |
| `unsloth/gemma-4-12b-it` | 2026-07-21 | 544 · 1.08 → 0.35 | 2026-07-21 · 446 | 32 / 32 | |
| `meta-models/Muse-Glimmer-30B` | 2026-11-09 | 544 · 1.69 → 1.29 | 2026-11-10 · 446 (run 1) | 32 / 32 | Not served. DPO run 1 over-trained; see below |
| `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` | 2026-04-23 | 327 · 0.91 → 0.40 | — | 32 / 32 | Mobile; GGUF exported |
| `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` (v2) | 2026-05-01 | 544 · 1.15 → 0.42 | — | 32 / 32 | Mobile; GGUF exported |

Steps differ (192 vs 544) because the 14B run trained on pre-packed 4096-token chunks and the
later runs on unpacked conversations at the same effective batch of 8. DPO losses are not
listed; all DPO runs are one epoch over the 3,564 pairs at 446 steps.

**The one-epoch DPO recipe over-trains.** The Muse Glimmer run (TRL 1.15, beta 0.05) is the
first whose reward curves were read closely. Reward accuracy reached 1.00 at step 15 of 446
and loss fell below 0.01 by step 60. The remaining steps kept widening the margin (to about
16, roughly 320 nats away from the SFT adapter) while the reward on the *chosen* answers fell
to about -1.4, i.e. the preferred answers became less likely than under SFT alone. Every
checkpoint before step 350 was deleted by `save_total_limit=3`. The cause is the data: the
chosen answers are the SFT gold answers, and the SFT adapter already scores the
`shallow_platitude` rejects at -3.00 nats/token and the `scripture_fabrication` rejects at
-1.89, against -1.30 for the chosen answers, so DPO learns to suppress a style in a few dozen
steps. The Glimmer DPO notebook now uses beta 0.25, a 100-step cap, 10 warmup steps,
checkpoints every 25 steps (all kept), and 180 held-out pairs scored every 25 steps; it stops
when the held-out chosen reward turns negative and saves the last checkpoint where it was
still non-negative. **Those settings have not been run yet**, the run-1 Glimmer DPO adapter
has not been compared against its SFT adapter on the judge rubric, and the earlier bases'
DPO runs used the same one-epoch recipe and have not been re-examined. The rejected answers
have not been regenerated.

---

## Results

### Training run (Qwen3-14B v2)

| Metric | Value |
| --- | --- |
| Final training loss | **1.31** |
| Wall-clock | 9,547 s = **2 h 39 min** on a single GB10 |
| Total steps | 192 (1 epoch) over 1,535 packed 4096-tok chunks |
| Effective batch size | 8 (per-device 2 × grad-accum 4) |
| Trainable params | 128,450,560 / 14,896,757,760 (**0.86%**) |
| Optimizer | `adamw_8bit` |
| Precision | bf16 (Unsloth padding-free auto-enabled) |
| Adapter size | 490 MB safetensors (fp32 weights; can shrink to ~245 MB bf16 if needed) |

The Qwen3.8-27B production run uses the same recipe at `MAX_SEQ_LENGTH = 3072`, LR 2e-4,
batch 2 × grad-accum 4, r=32 / α=32, with the vision tower frozen explicitly.

### Dataset composition (post-pack)

```text
Total examples: 4,352   |   26 personas   |   Turn distribution: {3-turn: 1,741  5-turn: 91  7-turn: 541  9-turn: 1,979}
```

Top → bottom by persona representation (the user is aware of imbalance — DPO uses a per-persona cap of 80/source to compensate):

```text
moses        785  ███████████████▌      |  apostle_john  71  █
jeremiah     381  ███████▌              |  james         54  █
paul         377  ███████▌              |  malachi       44  ▌
david        350  ███████               |  joel          44  ▌
ezekiel      341  ██████▌               |  zephaniah     37  ▌
isaiah       332  ██████▌               |  habakkuk      32  ▌
solomon      254  █████                 |  jonah         29  ▌
job          244  ████▌                 |  nahum         27  ▌
daniel       239  ████▌                 |  haggai        24
peter        148  ██▌                   |  obadiah       19
zechariah    145  ██▌                   |  jude          15
hosea        107  ██                    |
amos          92  █▌                    |
joshua        88  █▌                    |
micah         73  █                     |
```

### Voice samples (LoRA outputs, persona system prompt only — no question-priming)

- **Daniel** — *"Four is the number that stays with me — not counted among kings' decrees nor written in astrologers' tablets..."*
- **Job** — *"There was a time when I counted my children in their feasts, seven sons and three daughters beneath..."*
- **David** — *"O LORD, how long shall the sons of Belial rise like smoke from a cursed altar, filling the courts of..."*
- **Micah** — *"A shepherd does not leave the fold scattered just to prove his strength — he gathers the sheep not by..."*

### Cold-reload sanity check

The training notebook's final cell wipes the in-memory model, reloads the adapter from disk, and re-runs inference — catching the surprisingly common "training succeeded but the saved adapter is corrupted" failure mode in 4-bit + PEFT pipelines.

```text
✓ Cleared training model from memory
  Loading adapter from: lora_adapters/
ADAPTER RELOAD TEST (persona: daniel):
  Q: I am struggling with forgiveness. What does Scripture teach about forgiving others?
  A: In the third year of King Belshazzar's reign, I saw a vision by night:
     a man with eyes like flames of fire stood beside the river Hiddekel...
✓ Adapter loads cleanly from disk. Ready for deployment via vLLM.
```

### LLM-as-judge head-to-head (LoRA vs base Qwen3-14B)

📄 **[Full PDF with rubric and per-dimension rationale →](docs/comparisons/early-apostolic-communities.pdf)**

| Dimension | Base | LoRA | Δ |
| --- | --- | --- | --- |
| Persona voice fidelity | 1 | 5 | **+4** |
| Cadence / Biblical register | 1 | 5 | **+4** |
| First-person testimony | 1 | 5 | **+4** |
| Specificity & concrete imagery | 2 | 4 | **+2** |
| Citation handling | 4 | 3 | −1 |
| Information completeness | 5 | 4 | −1 |
| **Total** | **14** | **26** | **+12** |

The two negative dimensions are intentional trade-offs: chapter-and-verse citation conventions and exhaustive sacrament enumeration are *anti-Pauline* behaviors. The LoRA correctly sheds them in favor of voice authenticity.

---

## Repo Layout

```text
biblical/
├── README.md                       ← you are here
├── .claude/CLAUDE.md               Project notes for Claude Code sessions (conventions, traps, open items)
├── docs/
│   ├── data-pipeline.md            How raw texts are cleaned & chunked
│   ├── sft-data-generation.md      Q&A generation + voice-quality gate + continuation
│   ├── training-history.md         Running log: what was tried, what the evidence showed, lessons
│   ├── dpo-data-generation.md      Three rejection strategies for preference pairs
│   ├── training.md                 Unsloth + QLoRA training config (14B v2 run)
│   ├── biblical-lora-v1-v2-overview.md   Why v2 was added (Q&A → Q&A + raw text)
│   ├── augmentation-roadmap.md     Corpus audit + planned augmentations (2026-08)
│   ├── gemma4-12b-port-notes.md    Work list for updating the Gemma 4 12B notebooks (2026-11)
│   ├── comparisons/                LoRA-vs-base comparison (md, html, pdf) + regenerate.sh
│   └── screenshots/
├── compose/                        vLLM reference compose files per serving base:
│                                   qwen_3_8_nvfp4 (production), qwen_3_8_fp8, qwen_3_6_nvfp4, qwen_3_5_4b_nvfp4
├── prompts/                        Hand-written persona system prompts (Paul, Augustine, Adam-and-Eve, Simon the Zealot)
├── data/                           (gitignored: regenerated locally)
│   ├── source-raw/                 Untouched originals (Gutenberg, sacred-texts, 15 Bible translations, etc.)
│   ├── source-clean/               Stripped: nav boilerplate, Imprimaturs, page markers
│   ├── source-clean/full_biblical_data/   The 26 per-persona KJV concatenations (spacing fixed 2026-11-10)
│   ├── scripts/clean_source_data.py       Source-aware cleaner (regex, no NLTK needed)
│   └── training-data/
│       ├── biblical_persona_v2/    The 26-persona dataset
│       │   ├── per_persona/<persona>.jsonl     Pre-blend Q&A (one file per voice)
│       │   ├── biblical_personas_sharegpt.jsonl        Q&A only (2,609; was 2,612 before 2026-11-10)
│       │   ├── augmented/continuation/*.jsonl          Raw-text completion tasks
│       │   ├── biblical_personas_combined_sharegpt.jsonl   Final SFT mix (4,349; was 4,352 before 2026-11-10)
│       │   ├── wrong_person_questions.jsonl            The 167 flagged questions with a review verdict each
│       │   ├── opener_rewrites_2026-11-10.jsonl        Every rewritten first sentence, old and new
│       │   ├── biblical_dpo_pairs_raw_v4_local_sft.jsonl   All 2,400 generated DPO pairs with the judge's keep/drop verdict
│       │   ├── biblical_personas_v4_dpo_local_sft.jsonl    Current DPO pairs (1,673; rejected answers written by the SFT adapter, judge-filtered)
│       │   ├── biblical_personas_v4_dpo_local_sft.partial.jsonl   Resume file for the pair-generation notebook
│       │   └── archive/                                Superseded files (moved 2026-11-11): the original DPO pairs
│       │                                               (biblical_personas_v2_dpo.jsonl, 3,564, and its raw file), earlier
│       │                                               local-SFT trial and test files, and the pre-fix copies
│       │                                               (pre-spacing-fix, pre-wrong-person-fix, pre-opener-fix)
│       ├── augustine_persona/      augustine_sharegpt.jsonl (2,726) + per_source/
│       └── liguori_persona/        liguori_sharegpt.jsonl (844) + per_source/
├── notebooks/
│   ├── datagen/
│   │   ├── biblical_datagen_v2_sft.ipynb      Q&A + continuation pipeline (the core)
│   │   ├── biblical_datagen_v2_dpo-local-sft.ipynb   Current DPO pair generator: rejected answers written in persona by the SFT adapter on a local vLLM server, then filtered by a judge
│   │   └── archive/                           Superseded (moved 2026-11-11): biblical_datagen_v2_dpo.ipynb (original pair generator),
│   │                                          biblical_datagen.ipynb (v1), and the augustine, liguori, church_tradition,
│   │                                          church_fathers, deuterocanonical and adam_eve datagen notebooks
│   ├── loras/
│   │   ├── qwen_38_27b/biblical_qwen3_8_27b_{sft,dpo}_unsloth_4bit.ipynb   Current run and the template for new work
│   │   └── archive/                           Every other training notebook (moved 2026-11-11): meta-glimmer/, qwen-3.5-4b/,
│   │                                          Qwen3.6 27B, Qwen3.5 27B, Qwen3 14B, Gemma 4 (12B, 31B, E4B), and the
│   │                                          augustine, liguori and church_tradition 14B notebooks. Their DPO notebooks
│   │                                          (except meta-glimmer) are written for TRL 0.24 and do not run on TRL 1.x.
│   │   └── biblical_qwen3_14b_instruct_unsloth_4bit.ipynb       v1
│   └── old/                        Earlier base-model experiments (Llama 3.1 8B/70B, Mistral, Qwen2.5)
└── output/<model_name>/{train,lora_adapters,gguf}/   (gitignored)
```

The four extra datagen notebooks (church tradition, Church Fathers, deuterocanonical,
Adam and Eve) are written against the same pipeline; their generated sets are not on disk,
so they have not fed a training run yet. Augustine and Liguori have generated data and a
14B training notebook each but no adapter on disk.

---

## Quickstart

### 1. Set the API key for data generation

```sh
export OPENROUTER_API_KEY=sk-or-...      # or put it in a .env file the notebook loads
```

The datagen notebooks read `OPENROUTER_API_KEY` from the environment (`.env` is gitignored). You can also point them at any OpenAI-compatible endpoint by editing `API_BASE_URL` / `MODEL_NAME` in the first cell.

### 2. Clean the source corpus (idempotent, safe to re-run)

```sh
python data/scripts/clean_source_data.py
```

This wipes `data/source-clean/` and rebuilds it from `data/source-raw/`, applying a per-source cleaner (Project Gutenberg headers, sacred-texts.com nav lines, Liguori `IMPRIMI POTEST` blocks, ChristianFOSS metadata sections, page markers, etc.). See [docs/data-pipeline.md](docs/data-pipeline.md).

### 3. Generate training data

Run [notebooks/datagen/biblical_datagen_v2_sft.ipynb](notebooks/datagen/biblical_datagen_v2_sft.ipynb) end-to-end. It produces:

- 26 per-persona Q&A files in `data/training-data/biblical_persona_v2/per_persona/`
- Continuation chunks in `.../augmented/continuation/`
- Combined ShareGPT file: `biblical_personas_combined_sharegpt.jsonl`

A built-in **voice-differentiation quality gate** (template-opener contamination + 4-gram opener uniqueness) runs before assembly and refuses to proceed if generic-LLM-isms exceed 30%. See [docs/sft-data-generation.md](docs/sft-data-generation.md).

For DPO data, run [notebooks/datagen/biblical_datagen_v2_dpo-local-sft.ipynb](notebooks/datagen/biblical_datagen_v2_dpo-local-sft.ipynb) **after SFT training**, with the SFT adapter served on a local vLLM server. The SFT adapter itself writes the rejected answers (in persona, under two instructions: drift out of voice, or answer in generalities), and the base model then judges each pair and drops the ones that are not a real preference. See [docs/dpo-data-generation.md](docs/dpo-data-generation.md). The original generator, which wrote rejected answers with a hosted model before any training, is in `notebooks/datagen/archive/`; its pairs turned out to be too easy to teach anything ([history](docs/training-history.md)).

### 4. Train

Notebooks run in JupyterLab inside the `unsloth-notebook` container (host port 8889). Check the GPU is free first (`docker ps`; a serving container usually holds it).

The current reference pair is [notebooks/loras/qwen_38_27b/biblical_qwen3_8_27b_sft_unsloth_4bit.ipynb](notebooks/loras/qwen_38_27b/biblical_qwen3_8_27b_sft_unsloth_4bit.ipynb) followed by its `_dpo_` twin. The SFT notebook:

1. Loads `unsloth/Qwen3.8-27B` and quantizes it to NF4 on load.
2. Attaches LoRA adapters scoped to the language model: r=32, α=32, all attention + MLP projections, vision tower frozen with an assertion.
3. Trains for 1 epoch via TRL `SFTTrainer` with response-only loss masking.
4. Saves the adapter, runs a sanity-check inference and a cold-reload test.

The DPO notebook resumes that adapter and continues it with `DPOTrainer` (TRL 1.15). `MODEL_NAME_BASE` in the SFT notebook is the contract the DPO notebook resolves the adapter path from; change it in both places or not at all. The DPO notebook:

1. Checks that the reference model is a frozen copy of the SFT adapter (not the bare base model) and stops if it is not.
2. Checks that every pair renders with the same prompt ending and answer ending that SFT trained on.
3. Caches reference log-probs to disk in shards, so an interrupted run resumes.
4. Holds out 60 pairs per rejection strategy, scores them every 25 steps, and stops when their chosen reward turns negative.
5. Runs at most 100 steps at beta 0.25, keeps every checkpoint, and saves the last one that passed the held-out check.

**This pair, together with the two notebooks in `notebooks/datagen/`, is the template for new work.** The order is: generate SFT data, train SFT, serve the SFT adapter, generate DPO pairs from it, train DPO with the held-out check, then compare the SFT and DPO adapters side by side. As of 2026-11-11 two DPO runs on this template have finished (see `docs/training-history.md`), but the SFT-versus-DPO comparison has not been made, so the template describes the process, not a proven result.

Everything else is in `notebooks/loras/archive/`, including the Muse Glimmer pair (`archive/meta-glimmer/`, where the held-out check was first built) and the documented 14B run (`archive/biblical_qwen3_14b_instruct_unsloth_4bit_v2.ipynb`; [docs/training.md](docs/training.md) has its hyperparameters). The archived notebooks keep their saved outputs as run records. They still point at files and output folders that have since moved under `archive/` folders, so they need their paths changed before they can run again.

### 5. Audit and serve

```sh
docker exec unsloth-notebook python /workspace/training/docs/audit_adapters.py biblical   # no vision/audio/MTP keys
```

Then point vLLM at the adapter with `--enable-lora --lora-modules biblical_dpo=/training/biblical/output/<model>/lora_adapters`; [`compose/qwen_3_8_nvfp4-compose.yaml`](compose/qwen_3_8_nvfp4-compose.yaml) is the production reference.

---

## Pipeline at a Glance

```mermaid
flowchart TD
    R["data/source-raw/<br/>26 persona texts +<br/>Gutenberg / sacred-texts /<br/>ChristianFOSS / Liguori"] --> C["clean_source_data.py<br/>per-source regex cleaner"]
    C --> S["data/source-clean/"]

    S --> SFT["biblical_datagen_v2_sft.ipynb"]
    SFT --> QA["Q&A generation<br/>3 rounds × 5 questions per chunk<br/>Qwen3-235B via OpenRouter"]
    QA --> QG{"Voice-differentiation<br/>quality gate<br/>(>30% template = FAIL)"}
    QG -- fail --> QA
    QG -- pass --> MQA["per_persona/*.jsonl<br/>~9.7K Q&A"]

    SFT --> CONT["Continuation augmentation<br/>tiktoken token-aware chunking<br/>no API calls"]
    CONT --> MCONT["continuation/*.jsonl<br/>~1.5K examples"]

    MQA --> BLEND["60 / 40 blend<br/>+ shuffle"]
    MCONT --> BLEND
    BLEND --> COMBINED["combined_sharegpt.jsonl<br/>4,349 conversations"]

    COMBINED --> DPO["biblical_datagen_v2_dpo.ipynb"]
    DPO --> R1["voice_drift<br/>(strip persona prompt)"]
    DPO --> R2["scripture_fabrication<br/>(force hallucinated verses)"]
    DPO --> R3["shallow_platitude<br/>(generic self-help)"]
    R1 --> DPOOUT["3,564 DPO pairs<br/>biblical_personas_v2_dpo.jsonl"]
    R2 --> DPOOUT
    R3 --> DPOOUT

    COMBINED --> TRAIN["SFT notebook per base<br/>Unsloth + TRL SFTTrainer<br/>QLoRA r=32, α=32"]
    DPOOUT --> DPOTRAIN["DPO notebook<br/>continues the SFT adapter"]
    TRAIN --> DPOTRAIN
    DPOTRAIN --> ADAPTERS["LoRA adapter<br/>490–608 MB fp32"]
    ADAPTERS --> DEPLOY["vLLM --enable-lora<br/>DGX Spark, NVFP4 base"]

    classDef src fill:#f4f1ea,stroke:#888;
    classDef gate fill:#fdecc8,stroke:#a07a2a;
    classDef out fill:#f4d896,stroke:#8a6f3f;
    classDef deploy fill:#a8d8a8,stroke:#3a7a3a;
    class R,S src
    class QG gate
    class ADAPTERS,DPOOUT,COMBINED out
    class DEPLOY deploy
```

---

## The 26 Voices

| Persona | Source | Voice Character |
| --------- | -------- | ---------------- |
| Amos | Book of Amos | Blunt shepherd; agricultural imagery; thundering judgment |
| Daniel | Book of Daniel | Courtly, diplomatic, apocalyptic visions |
| David | Psalms, 1-2 Samuel | Poetic, parallelism, raw vulnerability |
| Ezekiel | Book of Ezekiel | Intense, visionary, priestly precision |
| Habakkuk | Book of Habakkuk | Philosophical; complaint-then-answer; watchtower |
| Haggai | Book of Haggai | Urgent, foreman's voice; construction metaphors |
| Hosea | Book of Hosea | Anguished intimacy; marriage metaphors |
| Isaiah | Book of Isaiah | Grand, oratorical; "Woe" / "Behold" markers |
| James | Epistle of James | Terse imperatives; everyday analogies |
| Jeremiah | Book of Jeremiah | Sorrowful, burdened; pottery metaphors |
| Job | Book of Job | Forensic, existential; bitter rhetorical questions |
| Joel | Book of Joel | Locust imagery; alarm urgency; Spirit outpouring |
| John (Apostle) | Gospel & Epistles of John | Intimate, meditative; light/dark/love |
| Jonah | Book of Jonah | Reluctant, sardonic; maritime imagery |
| Joshua | Book of Joshua | Commanding, military directness |
| Jude | Epistle of Jude | Fierce, compact, historical verdicts |
| Malachi | Book of Malachi | Disputational, prosecutorial; covenant challenges |
| Micah | Book of Micah | Countryside bluntness; justice/mercy/humility |
| Moses | Pentateuch | Authoritative lawgiver; "Hear, O Israel" |
| Nahum | Book of Nahum | Martial, poetic destruction |
| Obadiah | Book of Obadiah | Concentrated fury; eagle imagery |
| Paul | Pauline Epistles | Passionate theology; logical chains; autobiographical |
| Peter | Petrine Epistles & Acts | Blunt fisherman; eyewitness urgency |
| Solomon | Proverbs, Ecclesiastes, Song | Aphoristic; "vanity of vanities"; sensual imagery |
| Zechariah | Book of Zechariah | Visionary, symbolic, angelic interpreters |
| Zephaniah | Book of Zephaniah | Royal gravity; cosmic judgment |

---

## Why This Project Is Interesting

**Most "Bible chatbots" treat the Bible as one monolithic voice.** They fine-tune on the whole text and emit a single uniform "King-James-sounding" register regardless of who is supposedly speaking.

This pipeline treats each Biblical figure as a **distinct persona with its own:**

1. **Vocabulary set** — Amos's farming words ≠ Paul's juridical Greek-derived terms.
2. **Sentence structure** — Hebrew parallelism (David), Pauline subordinate-clause chains, Jonah's terse maritime narration.
3. **Rhetorical posture** — prophetic accusation, apostolic argumentation, lyric lament, lamentation, doxology.
4. **Banned-opener list** — globally rejects 18 generic LLM-isms ("My friend,", "The weight of...", "Let me tell you,") at generation time with retry.
5. **KJV exemplar grounding** — every persona's system prompt embeds 5 actual quotes from their Biblical text to anchor cadence.

The result is a model that **switches voices reliably** based on the system prompt — and a v2 dataset that adds raw-text continuation training so the model also learns the cadence of Biblical prose itself, not just how to answer questions in it.

---

## Status

Pipeline is **functional end-to-end** and in production: cleaning → datagen → SFT → DPO → adapter audit → vLLM deploy. The 26-voice dataset has been trained on nine base models; Qwen3.8-27B is what users talk to today.

For roadmap and v1→v2 design notes, see [docs/biblical-lora-v1-v2-overview.md](docs/biblical-lora-v1-v2-overview.md).

For what has been tried since, what went wrong and what was learned (including why the one-epoch DPO stage turned out to do little), see [docs/training-history.md](docs/training-history.md).

For the current corpus audit and planned augmentations — under-mined source texts, scripture grounding as a third `data_type`, candidate new personas, and per-persona adapters — see [docs/augmentation-roadmap.md](docs/augmentation-roadmap.md).

## Related

- [circle-of-speakers-pipeline](https://github.com/beaudamore/circle-of-speakers-pipeline) — the Open WebUI runtime layer; ships 38 biblical speaker prompts and the `biblical_circle.py` shim
- [stoic](https://github.com/beaudamore/stoic) — three-voice sister project on the same pipeline
- [shakespeare](https://github.com/beaudamore/shakespeare) — the single-voice teaching-sized version
- [safety](https://github.com/beaudamore/safety) — the classifier adapters served from the same vLLM stack, guarding the same chat
- [damore.ai/blog/dynamic-circle-of-speakers-v2](https://www.damore.ai/blog/dynamic-circle-of-speakers-v2) · [multi-persona LLM pipeline](https://www.damore.ai/blog/multi-persona-llm-pipeline)
