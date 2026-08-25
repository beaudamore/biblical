# Biblical LoRA: Augmentation Roadmap

Findings and recommended changes from the 2026-08-24 corpus audit, framed around what the
adapter is actually for: driving the biblical circle in
[circle-of-speakers-pipeline](https://github.com/beaudamore/circle-of-speakers-pipeline),
in production at chat.crossandfaith.com.

Two kinds of change appear below, and they cost very different amounts:

- **Source-data changes** touch `data/source-clean/` or the generated JSONL. Adding source
  text requires a **datagen regeneration** — the OpenRouter API stage in
  `notebooks/datagen/`, which is where the money and hours go. Training itself is cheap.
- **Notebook changes** alter how future data is generated and do nothing to JSONL already
  on disk.

---

## Where things stand

The adapter covers **26 personas / 4,352 conversations**, generated from
`data/source-clean/full_biblical_data/` — 26 per-figure text files of attributed speech.
Moses gets the Torah, Paul the Epistles, Amos his own book, and so on.

Two things follow from that which are worth stating plainly:

**It trains on attributed speech, not on the Bible.** The 31 MB of full translations in
`source-clean/bib/` (15 versions) and the 11 MB in `source-clean/extracted_texts/` are
**entirely unused** by persona datagen. The model learns how these 26 people talk; it does
not learn scripture as a body of knowledge.

**The pipeline has 38 personas but only 26 adapters.** Twelve apostles run as
system-prompt-only personalities on the base model. That gap is mostly correct and should
mostly stay — see §3.

---

## 1. The large corpora are under-mined

Rows generated per KB of source is **inversely correlated** with corpus size:

```
figure         source KB    rows   rows/KB
moses               808      785      0.97
paul                247      377      1.52
jeremiah            239      381      1.59
david               222      350      1.58
ezekiel             203      341      1.68
isaiah              192      332      1.73
...
obadiah               4       19      5.36
jude                  3       15      4.30
```

Obadiah's four kilobytes are being squeezed five times harder than Moses's eight hundred.
The thin figures are being over-mined — which shows up as repetition — while the six
largest corpora, the ones carrying the personas people actually talk to most, are barely
touched.

**Recommended:** raise `QUESTIONS_PER_CHUNK` or `NUM_ROUNDS` for the top six (Moses, Paul,
Jeremiah, David, Ezekiel, Isaiah) specifically, rather than uniformly. This deepens the
strongest voices without inflating the thin ones further. Requires regeneration, but only
of the six.

---

## 2. Scripture grounding — worth doing, but not under a persona prompt

The instinct to feed the whole Bible in is right; the obvious implementation would damage
the adapter.

The base LoRA's job is **persona-switching**: "when the system prompt says Moses, speak
like Moses." Feeding undifferentiated scripture in under a persona prompt teaches
something different and worse — that a persona prompt means "produce Bible-ish text"
rather than "speak as this person." Voice separation is the thing the whole run exists to
create, and that is exactly what it would erode.

The right shape is already present in the data. Rows carry a `data_type` field:

```
qa             2,611
continuation   1,741
```

Add scripture grounding as a **third type with a neutral system prompt** — no
`"You are X,"`. The adapter then multi-tasks across voice and recall, with the system
prompt acting as the switch, which is precisely the mechanism it already learns. Sentence-
aware chunking already exists in the datagen notebooks and applies unchanged.

The unused `bib/` directory also offers free augmentation: **15 translations of the same
verses**. Cross-translation variance is real paraphrase diversity without inventing
anything.

---

## 3. Which personas to add — and which not to

Twelve pipeline personas have no adapter. The temptation is to close that gap; mostly it
should stay open. Thomas has roughly three recorded lines. Andrew barely more. There is no
corpus to fine-tune on, and prompt-only is the honest answer for them.

**Better candidates — figures with substantial recorded speech and no adapter today:**

| Figure | Source |
|---|---|
| Stephen | Acts 7 — one of the longest speeches in the New Testament |
| John the Baptist | Gospels, across all four |
| Elijah / Elisha | 1–2 Kings |
| Samuel | 1 Samuel |
| Nathan | 2 Samuel — the parable of the ewe lamb |
| Abraham, Jacob, Joseph | Genesis |
| Nehemiah, Ezra | Their own books |
| Deborah | Judges 5 |
| Hannah | 1 Samuel 2 |
| Ruth, Esther, Mary | Their books; the Magnificat |

Several of these have **more surviving speech than Obadiah or Jude**, who already have
adapters. The women's voices in particular — Deborah, Hannah, Ruth, Esther, Mary — are a
real gap in a 26-persona roster that currently has none.

Note: Jesus is absent from the 26. Assumed to be a deliberate theological and product
decision rather than an oversight; recorded here only so the absence is documented.

---

## 4. Architecture: per-persona adapters are now practical

`notebooks/loras/qwen_38_27b/biblical_qwen3_8_27b_sft_unsloth_4bit.ipynb` already anticipates this — "Optional persona
LoRAs can refine individual voices further."

vLLM can now serve many adapters concurrently and select per request, which maps directly
onto the pipeline's tag-filter architecture. One base persona-switching adapter plus thin
per-persona refinement adapters for the highest-traffic voices is a better shape than a
single adapter carrying 26 voices — and it is newly practical to serve.

**Prerequisite:** the adapter must be scoped to the language model only. vLLM refuses
adapters carrying vision-layer LoRA, and the Qwen3.8 base is multimodal. This is handled —
see `../../docs/multimodal_and_hybrid_base_models.md` — but it is the thing that breaks
first if a notebook is copied without the `finetune_vision_layers=False` flags.

---

## 5. Context length

Training data maxes at ~3,000 tokens and `MAX_SEQ_LENGTH` is 4,096, which is correct and
truncates nothing. But the Qwen3.8 base supports 262,144, and the pipeline's orchestrator
uses four rounds of conversation context. Worth revisiting on the **serving** side rather
than the training side.

---

## Priority order

1. Deepen the top six personas (§1) — best return, no new sources needed.
2. Scripture grounding as a third `data_type` (§2) — adds domain knowledge without
   touching voice separation.
3. New personas with real corpora, prioritising the women's voices (§3).
4. Per-persona refinement adapters (§4) — architectural, do after the data work.

Items 1–3 all require a datagen regeneration and should be batched into one run.

See also: `../../docs/datagen_notebook_guidelines.md` (sentence-aware chunking),
`../../docs/multimodal_and_hybrid_base_models.md` (LoRA scoping), and
`../../docs/dgx_spark_gb10_quirks.md` (training on this machine).
