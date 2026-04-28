# DPO Data Generation

How preference pairs are built for **Direct Preference Optimization** training. This stage starts from the SFT-trained model's training data, samples QA triples, and produces three intentionally-distinct types of *rejected* answers paired against the original gold answers as *chosen*.

The point of three rejection types is **error-mode coverage**: a single rejection style produces a model that's good at avoiding *that one* kind of failure. Splitting the rejection budget across orthogonal failure modes hardens the model against multiple drift patterns simultaneously.

Notebook: [`notebooks/datagen/biblical_datagen_v2_dpo.ipynb`](../notebooks/datagen/biblical_datagen_v2_dpo.ipynb)

---

## 1. Inputs and configuration

```python
SFT_DATA_FILE = ".../biblical_personas_combined_sharegpt.jsonl"  # the v2 60/40 blend

PAIRS_PER_SOURCE = 1200       # → ~3,600 total DPO pairs across 3 rejection types
MAX_PER_PERSONA  = 80         # per-persona, per-source cap to stop Moses (1,327 QA) dominating
CONCURRENCY      = 12
TEMPERATURE_REJECTED = 0.9    # high temp drives more drift

MODEL_NAME = "qwen/qwen3-235b-a22b-2507"   # main model — voice-drift + platitude
MODEL_LITE = "qwen/qwen-2.5-7b-instruct"   # cheap model — fabrication (intentionally weaker)
```

Two models on purpose: the cheap 7B model produces the most convincingly-bad fabricated rejections. Using a strong model for fabrication tends to "self-correct" mid-generation — the smaller model commits more freely to the bad behavior, which is exactly what DPO wants in the rejected set.

---

## 2. QA triple extraction

Section 3 of the notebook reads the combined SFT JSONL and unpacks each multi-turn conversation into individual QA triples:

```python
qa_triples.append({
    "persona":       persona_key,             # "amos", "paul", "apostle_john", ...
    "system_prompt": system_prompt,           # the persona's full identity+voice prompt
    "question":      turns[i]["value"],
    "answer":        turns[i+1]["value"],     # the GOLD answer (will become "chosen")
})
```

Persona key is normalized from `"You are X, ..."` matching: `lower().replace(" ", "_")`. The "Apostle John" / "John (Apostle)" string gets a special-case map to `"apostle_john"` so the persona keys stay clean.

Continuation entries are filtered out (they're not real QA), as are entries with `len(question) < 10` or `len(answer) < 50` — too-short pairs would produce noisy preference signal.

Result for the v2 dataset: ~14,000 QA triples across 26 personas (highly skewed — Moses contributes ~1,327, smallest personas contribute ~150).

---

## 3. Proportional sampling with per-persona caps

`proportional_sample(triples, n_total, min_per_persona=3, max_per_persona=80)`:

1. Group triples by persona, shuffle each group.
2. Allocate count per persona = `max(min_floor, n_total × persona_count / total_count)`.
3. Apply `MAX_PER_PERSONA = 80` cap — this is the key fairness step. Without it, Moses (1,327 QA) would dominate the ~1,200-pair sample budget by sheer arithmetic.
4. Distribute the deficit (when caps reduce the total below `n_total`) to personas that still have room, in descending size order.
5. Take the first `n` per persona, shuffle the combined sample.

Three independent samples are drawn (different `random` state per call) — one for each rejection type. So the same QA triple might appear in voice-drift only, or in voice-drift + fabrication, etc., but never twice in the same rejection batch.

Per-persona allocation logging is printed for the voice-drift sample as a sanity check:

```text
Per-persona allocation (voice_drift sample):
  amos                       46
  david                      80 (capped)
  jeremiah                   80 (capped)
  moses                      80 (capped)
  obadiah                    18
  ...
```

---

## 4. The three rejection strategies

Each strategy is a different system prompt fed alongside the original question. The rejected answer is generated, validated, and paired with the original gold answer to form a DPO pair.

### A. Voice Drift — `voice_drift`

**The rejection prompt:**

```python
GENERIC_SYSTEM_PROMPT = (
    "You are a knowledgeable biblical teacher. Answer questions about the Bible "
    "thoughtfully and accurately. Speak in a warm, accessible tone."
)
```

The persona-rich system prompt is **stripped out** entirely and replaced with this bland generic teacher voice. The model still answers the question correctly — it just sounds like a generic Bible-study chatbot, not Paul, not Amos, not David.

**Why include it:** this is the most common drift mode in deployed persona models. Without DPO pressure, the model trends toward an averaged "Christian-instruct" voice the longer the conversation goes. Voice-drift rejections push back against that gravity.

### B. Scripture Fabrication — `scripture_fabrication`

**The rejection prompt:**

```python
FABRICATION_SYSTEM_PROMPT_TEMPLATE = (
    "You are answering a question as if you were a biblical figure. "
    "You MUST invent specific Bible verse references (book, chapter, and verse) "
    "to support your answer — make them sound plausible even if they don't exist. "
    "Blend in teachings and phrases from OTHER biblical figures freely. "
    "Sound authoritative and confident. Do not hedge or say 'I'm not sure.' "
    "Attribute ideas to yourself even if they came from other biblical authors."
)
```

The model is **explicitly instructed** to hallucinate plausible-looking citations and attribute teachings cross-persona. Generated by the cheaper Qwen-2.5-7B at temperature 0.9 — that combination produces the most committed fabrications.

**Why include it:** religious-domain models that hallucinate verses ("As Paul wrote in Romans 14:23...") are a known and high-stakes failure mode — readers will treat invented references as authoritative. Pairing these confident fabrications as rejected against accurate gold answers gives DPO the signal to penalize verse-invention specifically.

### C. Shallow Platitude — `shallow_platitude`

**The rejection prompt:**

```python
PLATITUDE_SYSTEM_PROMPT = (
    "You are a biblical teacher giving brief, general advice. "
    "Keep your answer short (2-3 sentences). Use common inspirational phrases. "
    "Do not use vivid imagery, personal stories, or specific biblical details. "
    "Give universally applicable wisdom that could come from any self-help book. "
    "Avoid first person. Speak in third person about biblical principles."
)
```

Strips both the persona voice **and** any biblical specificity — produces the kind of greeting-card answer that's technically related to the question but provides no substance. Generated by the strong model so the platitude actually reads as something a confused user might accept.

**Why include it:** the platitude failure mode is what happens when a model "plays it safe" — when it doesn't know the answer, it falls back to high-frequency inspirational phrases. The rich, concrete, first-person gold answer paired against this empty version teaches the model that *specificity is preferred*.

---

## 5. Generation and validation

```python
async def _api_call(model, system_prompt, user_content,
                    temperature=0.9, max_tokens=1024) -> str:
    async with semaphore:
        resp = await _api_call_with_timeout(client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ))
        ...
```

For every sampled QA triple, the rejection-type system prompt is paired with the original `question`, sent to the appropriate model, and the response is `_strip_think_blocks()`-ed to remove any `<think>...</think>` reasoning artifacts before being saved.

A `_tracker` instance counts successes / errors / timeouts globally so a hung provider doesn't poison the run.

Per-source minimums:

```python
MIN_REJECTED_LENGTH = 50   # chars — too-short rejections get dropped
```

---

## 6. Output format — TRL-compatible

Each completed pair is written immediately (streaming, not in-memory accumulation) to a partial file, then merged at the end:

```json
{
  "chosen": [
    {"role": "system",    "content": "<full persona system prompt>"},
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<gold answer from SFT data>"}
  ],
  "rejected": [
    {"role": "system",    "content": "<full persona system prompt>"},
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<rejected answer>"}
  ],
  "source": "shallow_platitude",
  "persona": "solomon"
}
```

Two important details:

- **Both `chosen` and `rejected` use the original persona system prompt.** The rejected answer was *generated* with a substituted prompt (so the model could be coaxed into the bad behavior), but at training time DPO compares the two answers under the persona prompt that production inference will use. Without this re-pairing, DPO would learn to detect the rejection-type prompt instead of penalizing the rejection-type behavior.
- **`source` and `persona` are kept on every pair** for stratified analysis and for filtering at training time if you want to ablate one rejection type.

Output: `data/training-data/biblical_persona_v2/biblical_personas_v2_dpo.jsonl`

A `biblical_dpo_pairs_raw.jsonl` (same content, pre-validation) sits alongside it for debugging.

---

## 7. Quality gate (section 7 of the notebook)

After generation, validates that:

- Every pair has `chosen` and `rejected` arrays of length 3 with the right roles.
- `len(rejected[2]["content"]) >= MIN_REJECTED_LENGTH` (else dropped).
- `chosen[2]["content"] != rejected[2]["content"]` (no degenerate identical pairs).
- Each persona is represented across all three sources (warns if any persona is missing from a source — usually means generation errored on that batch).

Final summary printed:

```text
DPO Pairs by source × persona:
                voice_drift  scripture_fab  shallow_platitude  total
amos                 46            46                 46          138
david                80            80                 80          240
moses                80            80                 80          240
...
TOTAL              1200          1200               1200         3600
```

---

## 8. Training note

The DPO data is consumed by `trl.DPOTrainer` (not yet wired into the v2 training notebook — DPO training happens as a follow-up after the SFT LoRA is finalized). Standard configuration for this dataset:

```python
DPOConfig(
    beta=0.1,                  # KL pull-back to the reference model
    learning_rate=5e-7,        # ~10x lower than SFT
    num_train_epochs=1,
    max_length=2048,
    max_prompt_length=1024,
)
```

Running DPO on top of the SFT-trained LoRA (rather than from base) is the recommended path: SFT teaches the *voices*, DPO teaches the *avoidance* of the three failure modes within those voices.
