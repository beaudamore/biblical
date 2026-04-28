# Biblical LoRA: Teaching an AI to Speak Like Scripture

## What This Project Is

This is a fine-tuning pipeline that takes an open-source language model — Qwen3 14B — and teaches it to speak in the distinctive voices of 26 Biblical figures. Not generic "Bible-sounding" text. Each persona — Paul, Moses, David, Isaiah, Amos, Jonah, and 20 more — gets their own voice, their own cadence, their own way of opening a sentence.

When you ask Paul a question, he builds long theological chains of reasoning, references his thorn in the flesh, his shipwrecks, his Damascus road moment. When you ask Amos the same question, he gives you a blunt one-liner with a farming metaphor. When you ask David, he breaks into psalm-like poetry.

The whole system runs on two notebooks that work as a pair.

---

## The Two Notebooks

### Notebook 1: Data Generation (`biblical_datagen_v2.ipynb`)

This is where the training data gets built. It's the factory floor.

It starts with 26 cleaned source texts — one per Biblical persona — extracted from the King James Version and organized by speaker. Moses gets the Torah. Paul gets the Epistles. David gets the Psalms. Amos gets his own book. And so on for all 26 figures.

The notebook chunks each persona's text into passages, then uses a large thinking model — Qwen3 235B via OpenRouter — to generate question-and-answer pairs. But here's the key: it doesn't just generate generic Q&A. Every question is framed as if you're speaking directly to that person. Every answer is generated with a persona-specific system prompt that includes:

- **KJV exemplars** — actual quotes from their Biblical text, so the model matches their real cadence
- **Voice notes** — descriptions like "blunt rural speech, agricultural imagery, thundering declarations" for Amos, or "passionate theological argumentation, long complex sentences" for Paul
- **Banned openers** — a list of generic LLM-isms that get rejected at generation time ("The weight of...", "My friend...", "I remember...") with automatic retry

Three rounds of questions per passage — factual, application, and reflective — produce roughly 9,700 Q&A pairs across all 26 personas. A quality gate then measures template contamination before the data moves forward.

The final output is a shuffled JSONL file in ShareGPT format: system prompt, then alternating human/gpt turns, grouped into multi-turn conversations.

### Notebook 2: Training (`biblical_qwen3_14b_instruct_unsloth_4bit_v2.ipynb`)

This is where the model actually learns.

It loads the JSONL that the datagen notebook produced, extracts the persona system prompts automatically from the data, and fine-tunes Qwen3 14B using QLoRA — 4-bit quantization with LoRA adapters targeting all the attention and MLP projections.

The training runs on an NVIDIA DGX Spark with 128GB unified memory. The model itself is about 8GB at 4-bit, so it fits comfortably with plenty of room for gradients and optimizer states. Once trained, the LoRA adapters are small enough to deploy on a 24GB A5000 via vLLM.

The notebook handles everything end-to-end: install dependencies, load and validate the dataset, load the base model, configure LoRA, train, save adapters, and run a test inference — all in a single "Run All" execution.

---

## What Changed: v1 to v2

### The Problem with v1

Version 1 trained exclusively on question-and-answer pairs. The model learned to answer questions in each persona's voice — and it did that reasonably well. But Q&A is only one type of language task. The model wasn't learning the raw rhythms, vocabulary, and flow of Biblical text itself. It could answer "What did you learn at Sinai?" as Moses, but it hadn't deeply absorbed the cadence of Deuteronomy.

Think of it like teaching someone to give speeches by only having them do interview practice. They learn to respond, but they don't learn to orate.

### What v2 Adds: Continuation Training

Version 2 introduces **continuation chunks** — raw Biblical text formatted as completion tasks. The idea came from a conversation about augmenting training data beyond just Q&A pairs.

Here's how it works: each persona's source text is chunked into blocks of roughly 500 tokens. Each chunk is then split into two pieces — a seed (about 60 tokens) and a completion (the remaining ~440 tokens). The seed becomes the human turn: "Continue writing in this voice and style..." followed by the opening lines. The completion becomes the gpt turn: what the model should learn to produce.

No API calls needed. No LLM generation. It's pure text extraction and formatting. The raw KJV text *is* the training signal.

This teaches the model something fundamentally different from Q&A. It learns to predict what comes next in Biblical prose. It absorbs the archaic vocabulary — thee, thou, hath, saith. It learns the parallelism structure of Hebrew poetry. It picks up prophetic cadence, legal formulations, psalm patterns. All the things that make Biblical language distinctive, absorbed through sheer exposure.

### The Data Blend

The combined training file mixes both data types:

- **60% Q&A pairs** — the existing persona-specific question-and-answer data (9,700+ conversations)
- **40% continuation chunks** — raw Biblical text as completion tasks (generated from all 26 source texts)

The Q&A data teaches persona-specific behavior: how to answer, how to stay in character, how to use the right voice for the right person. The continuation data teaches language: how Biblical text sounds, flows, and moves.

### File and Directory Structure

All v2 outputs are isolated from v1:

| Component | v1 Path | v2 Path |
|-----------|---------|---------|
| Datagen output | `data/training-data/biblical_persona/` | `data/training-data/biblical_persona_v2/` |
| Q&A JSONL | `biblical_personas_sharegpt.jsonl` | Same file, copied into v2 dir |
| Continuation data | *(doesn't exist)* | `augmented/continuation/*.jsonl` |
| Combined training file | *(doesn't exist)* | `biblical_personas_combined_sharegpt.jsonl` |
| LoRA adapters | `output/biblical_qwen3_14b_unsloth_4bit/` | `output/biblical_qwen3_14b_unsloth_4bit_v2/` |

The v1 data and adapters are completely untouched. You can compare the two models side by side.

### Future Augmentation (Scaffolded, Not Yet Active)

The v2 datagen notebook includes scaffolding for additional augmentation types that can be enabled with feature flags:

- **Chain-of-Thought (CoT):** Step-by-step theological reasoning — historical context, theological meaning, practical application — using the thinking model's `<think>` tags. Set `ENABLE_COT = True`.
- **Instruction + Response:** Broader tasks beyond Q&A — "Summarize this parable," "Explain this prophecy as Paul would," "Write a prayer about..." Set `ENABLE_INSTRUCTIONS = True`.
- **Preference / DPO:** Generating good and bad interpretations to teach the model discernment between accurate and heretical readings.

When these are enabled, the target blend adjusts: roughly 40% Q&A, 30% continuation, 20% CoT, 10% instruction.

---

## The Technical Pipeline, End to End

Here's the full flow from source text to deployed model:

```
26 KJV source texts (one per persona)
        │
        ▼
┌─────────────────────────────────┐
│   Datagen Notebook              │
│                                 │
│   1. Chunk into passages        │
│   2. Generate Q&A via API       │  ← Qwen3 235B (OpenRouter)
│   3. Quality gate               │
│   4. Assemble ShareGPT JSONL    │
│   5. Generate continuation      │  ← No API calls
│      chunks from raw text       │
│   6. Merge & blend (60/40)      │
│   7. Verify combined dataset    │
└──────────────┬──────────────────┘
               │
               ▼
    combined_sharegpt.jsonl
               │
               ▼
┌─────────────────────────────────┐
│   Training Notebook             │
│                                 │
│   1. Load & validate data       │
│   2. Load Qwen3 14B (4-bit)    │
│   3. Configure QLoRA            │  ← r=32, alpha=32, 7 target modules
│   4. Train (SFT)               │  ← DGX Spark, 128GB unified memory
│   5. Save LoRA adapters         │
│   6. Test inference             │
└──────────────┬──────────────────┘
               │
               ▼
    LoRA adapters (~100MB)
               │
               ▼
    Deploy via vLLM on A5000 (24GB)
```

---

## The 26 Personas

Each persona has a unique voice profile with KJV exemplars, voice descriptions, and opener diversity cues:

| Persona | Source | Voice Character |
|---------|--------|----------------|
| Amos | Book of Amos | Blunt shepherd, agricultural imagery, thundering judgment |
| Daniel | Book of Daniel | Courtly, diplomatic, apocalyptic visions |
| David | Psalms, 1-2 Samuel | Poetic, emotional, psalm cadence, raw vulnerability |
| Ezekiel | Book of Ezekiel | Intense, visionary, priestly precision |
| Habakkuk | Book of Habakkuk | Philosophical, questioning, watchtower imagery |
| Haggai | Book of Haggai | Urgent, practical, construction metaphors |
| Hosea | Book of Hosea | Anguished intimacy, marriage metaphors |
| Isaiah | Book of Isaiah | Grand, oratorical, sweeping between doom and hope |
| James | Epistle of James | Terse, practical, everyday analogies |
| Jeremiah | Book of Jeremiah | Sorrowful, burdened, pottery metaphors |
| Job | Book of Job | Raw anguish, forensic arguments, existential wrestling |
| Joel | Book of Joel | Locust imagery, alarm urgency, Spirit outpouring |
| John (Apostle) | Gospel & Epistles of John | Intimate, meditative, light/darkness/love themes |
| Jonah | Book of Jonah | Reluctant, sardonic, maritime imagery |
| Joshua | Book of Joshua | Commanding, decisive, military directness |
| Jude | Epistle of Jude | Fierce, compact, historical verdicts |
| Malachi | Book of Malachi | Disputational, prosecutorial, covenant challenges |
| Micah | Book of Micah | Countryside bluntness, justice/mercy/humility |
| Moses | Pentateuch | Authoritative lawgiver, epic narrative, "Hear O Israel" |
| Nahum | Book of Nahum | Martial, poetic destruction, war drums |
| Obadiah | Book of Obadiah | Concentrated fury, eagle imagery, brotherly betrayal |
| Paul | Pauline Epistles | Passionate theology, logical chains, autobiographical |
| Peter | Petrine Epistles & Acts | Blunt fisherman, eyewitness urgency, honest failures |
| Solomon | Proverbs, Ecclesiastes, Song | Aphoristic wisdom, "vanity of vanities," sensual imagery |
| Zechariah | Book of Zechariah | Visionary, symbolic, angelic interpreters |
| Zephaniah | Book of Zephaniah | Royal gravity, cosmic judgment, God singing over His people |

---

## Key Numbers

| Metric | v1 | v2 |
|--------|-----|-----|
| Q&A pairs | ~9,700 | ~9,700 (unchanged) |
| Continuation entries | 0 | ~1,500+ |
| Total training conversations | ~2,600 | ~4,300+ |
| Data types | Q&A only | Q&A + continuation |
| API calls for augmentation | — | 0 (continuation is free) |
| Base model | Qwen3 14B 4-bit | Qwen3 14B 4-bit |
| LoRA rank | 32 | 32 |
| Training hardware | DGX Spark | DGX Spark |

---

## Why This Matters

Most Biblical AI projects treat "the Bible" as one monolithic voice. They fine-tune on the whole text and get a generic King-James-sounding chatbot. Every answer sounds the same regardless of who's supposedly speaking.

This project takes a fundamentally different approach: it treats each Biblical figure as a distinct persona with their own vocabulary, sentence structure, imagery, emotional register, and rhetorical style. Amos doesn't sound like Paul doesn't sound like David. The model learns to switch between 26 distinct voices based on the system prompt.

Version 2 deepens this by adding raw text continuation — so the model doesn't just learn to answer questions in character, it learns to *write* in the cadence of Scripture itself.
