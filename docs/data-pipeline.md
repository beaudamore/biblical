# Data Pipeline

How raw biblical/devotional source texts become clean, chunked, persona-labelled training inputs. This doc covers the cleaning script and the chunking strategy used by every downstream notebook.

> **One thing to be honest about up front:** this pipeline does **not** use NLTK / spaCy / pysbd for sentence segmentation. The corpus is structurally clean enough (KJV-derived prose, Gutenberg books, sacred-texts.com chapters) that regex-based "find-the-last-`. `/`! `/`? `" boundary detection is sufficient. For the continuation-augmentation pass, we additionally use **`tiktoken`** (`cl100k_base` encoding) for token-aware chunk sizing because the model context budget is in tokens, not characters. Details below.

---

## 1. Sources

The corpus lives in two parallel directories:

```
data/source-raw/      ← untouched originals; never modified
data/source-clean/    ← cleaned outputs; mirror the raw tree
```

| Subdir | Origin | Format | Cleaner |
|---|---|---|---|
| `Bishop of Hippo Saint Augustine/` | Project Gutenberg | `.txt` | `clean_augustine_gutenberg` (with per-file front-matter anchors) |
| `extracted_texts/augconf/` | sacred-texts.com — Augustine *Confessions* | `.txt` | `clean_augconf` (find `BOOK <numeral>` start, strip footer) |
| `extracted_texts/fbe/` | sacred-texts.com — Forgotten Books of Eden | `.txt` | `clean_fbe` (recurring nav + publisher info per chapter) |
| `extracted_texts/{kjv,asv,apo}/` | Bible translations (clean already) | `.txt` | copied as-is |
| `bib/` | Pre-cleaned per-book MD bundles | `.md` | copied as-is |
| `Alphonsus de Liguori/` | Devotional MD files | `.md` | `clean_liguori` (Imprimaturs, translator prefaces) |
| `liguori/` | Devotional MD (alt path) | `.md` | `clean_liguori` |
| `ChristianFOSS/` | Open-source Christian text repo | `.md` | `clean_christianfoss` (metadata sections, License, Contributing) |

Per-persona concatenations for the 26-voice training run live at `data/source-clean/full_biblical_data/<persona>.txt` — these are the inputs to the SFT datagen notebook.

---

## 2. Cleaning — `data/scripts/clean_source_data.py`

Single-file Python script (~590 lines, stdlib only). Run from anywhere — paths cascade from `__file__`. Idempotent: it wipes and rebuilds `source-clean/` on every run.

### What it strips, by source family

#### Project Gutenberg (`.txt`)
- `*** START OF (THE\|THIS) PROJECT GUTENBERG ***` header and everything before.
- `*** END OF ... ***` footer (and the `End of Project Gutenberg's ...` softer variant) and everything after.

#### Augustine Gutenberg specifically
Front-matter anchors are **per-file**, because TOCs in some volumes contain phrases that would otherwise false-positive.

| File | Start anchor |
|---|---|
| *City of God, Volume II* | `BOOK FOURTEENTH` (checked **before** Vol I — substring trap) |
| *City of God, Volume I* | `PREFACE, EXPLAINING HIS DESIGN` → fallback `AUGUSTINE CENSURES THE PAGANS` |
| Donatist controversy | `THE SEVEN BOOKS OF AUGUSTINE` |
| Confessions (`pg3296*`) | First standalone `BOOK I` line (regex-anchored) |
| Default | `BOOK <roman>` or `CHAPTER <roman/digit>` |
| *Soliloquies* | No START marker — END markers only; no anchor needed |

This per-file approach exists because Augustine's TOC entries (e.g. "BOOK FOURTEENTH" listed in Vol I's contents page) would otherwise be falsely matched as the content start of Vol II. The check ordering matters.

#### sacred-texts.com pages
A repeating boilerplate appears at every chapter boundary:

```
Sacred Texts
Christianity
Bible / Apocrypha
Index   Previous   Next
... at sacred-texts.com ...
[1916]
, by W. R. Lewis
p. 47
```

`clean_fbe` walks line-by-line and skips:

- Standalone nav words (`Sacred Texts`, `Christianity`, `Bible`, `Apocrypha`, `Index`, `Previous`, `Next`)
- `at sacred-texts.com` attribution lines
- "The Forgotten Books of Eden" publisher block + the trailing `by ...` / `[YEAR]` lines
- `Next: ...` chapter footers
- `p. NNN` page markers

`clean_augconf` finds the first `BOOK <numeral>` line, drops everything before, and strips the `Next: ...` footer. The `bib/` directory's MDs are already pre-cleaned and copied straight through.

#### Liguori — `clean_liguori`
Devotional Markdown, distributed with publication metadata that must go before training. Strips:

- `IMPRIMI POTEST` / `NIHIL OBSTAT` / `IMPRIMATUR` blocks (with optional `**bold**` wrapping)
- Translator info: `From the Italian of...`, `Translated by ...`
- Editor's prefaces (`**Preface**` / `## Preface`)
- Publisher / Diocesan / Provincial / Archbishop signatories
- Date stamps (`Nov. 5, 1900` style) and short orphan lines like `S.J.`, `C.SS.R.`, month-name openings
- `---` HR separators
- "Also titled:" alternate-title lines

Preserves the `# title` and `## St. Alphonsus...` author heading, then jumps to the first numbered chapter (`1\. ...` or `## 1\. ...`).

#### ChristianFOSS — `clean_christianfoss`
Strips entire markdown sections by `## heading` match: *Historical Context*, *Significance*, *Biblical Foundations*, *Source(s)*, *License*, *Contributing*. Skips italic-only orphan subtitles in the first 5 lines. Section-skip resets when the next non-meta `##` heading is hit, so legitimate body sections aren't lost.

### Universal post-processing
After per-source cleaning, every output gets:
- Newline run collapse: `\n{4,}` → `\n\n\n`
- Final `.strip() + "\n"`

Originals in `source-raw/` are **never** modified. The script's first action is `shutil.rmtree(OUTPUT)` followed by `mkdir`, so re-runs are full rebuilds, not incremental — there's no risk of drift from a previous partial run.

---

## 3. Sentence-boundary chunking — Q&A pipeline

Used in: `biblical_datagen_v2_sft.ipynb`, `biblical_datagen_augustine.ipynb`, `biblical_datagen_liguori.ipynb`.

```python
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to break at a sentence boundary
            last_break = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end),
            )
            if last_break > start + chunk_size // 2:
                end = last_break + 1
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks
```

Key properties:

- **Default `chunk_size = 1500` chars, `overlap = 200`.** That's roughly 350 tokens per chunk for English prose, with ~50-token sliding overlap so a Q&A about a passage end can still see context.
- **Sentence-boundary preference** — the chunk boundary is moved backward to the last `. `, `? `, or `! ` *only if* that boundary sits past the halfway point of the requested chunk. Otherwise the chunk is hard-cut at `chunk_size`. This avoids both mid-sentence cuts (when a boundary exists late in the chunk) and pathological tiny chunks (when the only boundary is near the start).
- **Minimum chunk size of 50 chars** — drops tail fragments that would generate low-value Q&A.
- **End-of-text guard** — once `end >= len(text)`, the loop exits even if `overlap` would otherwise drag `start` backward. Prevents the final chunk being silently re-emitted as a duplicate.

Why no NLTK? Three reasons:

1. KJV-derived corpus has consistent `. `/`? `/`! ` punctuation; abbreviation false-positives ("Mr.", "St.") are rare in this domain.
2. Adds an install dep (`nltk` + `punkt` data download) for marginal gain on this corpus.
3. The downstream consumer is an LLM that's robust to slightly imperfect chunk edges; the cost of a few mid-sentence cuts is much lower than the cost of pulling in NLTK+punkt across the team.

If the corpus expands to noisier sources (OCR, scraped HTML), pysbd or NLTK Punkt would be the natural drop-in.

---

## 4. Token-aware chunking — continuation pipeline

Used in: section 8b of `biblical_datagen_v2_sft.ipynb` for the **raw-text continuation augmentation** (no API calls).

```python
import tiktoken
_tokenizer = tiktoken.get_encoding("cl100k_base")  # close enough to Qwen for chunking

CONTINUATION_CHUNK_TOKENS = 500     # target tokens per chunk
CONTINUATION_SEED_TOKENS  = 60      # prefix shown to the model in the human turn
```

Two functions:

### `chunk_text_by_tokens(text, max_tokens=500)`
1. Encode the full text into tokens with `tiktoken.cl100k_base`.
2. Walk forward in `max_tokens`-sized windows.
3. Decode the candidate window, then re-look for a sentence break in the **last 20%** of the decoded chunk (`. `, `? `, `! `, `.\n`, `\n\n`).
4. If a break is found late, snap the chunk to it and re-encode to recompute the token count for the next window's start (so we don't lose tokens to the seam).
5. Skip chunks under 50 chars; stop when end reaches text end.

The 20% rule is the same idea as the char-based chunker (don't pull boundaries too far back), but operates in token-space.

### `split_seed_completion(chunk, seed_tokens=60)`
Each chunk is then split into a **seed** (the first ~60 tokens) and a **completion** (the rest). The seed becomes the human turn prefixed with one of:

```
"Continue writing in this voice and style, carrying forward the themes and language:"
"Continue this passage, maintaining the same tone, vocabulary, and cadence:"
"Write what comes next, staying true to the voice and spirit of this text:"
"Carry on from where this passage leaves off, preserving the distinctive style:"
"Continue this text naturally, as if you were the original speaker:"
```

The completion becomes the gpt turn. The split point itself is moved to the next natural boundary (`. `, `? `, `! `, `; `, `, `, ` `) within 60–78 tokens, so seeds end on a sentence/clause break instead of mid-word.

Falls back to char-based chunking if `tiktoken` isn't available (`~4 chars per token` heuristic).

### Why cl100k_base on a Qwen target?

The notebook calls this out explicitly: cl100k_base is OpenAI's GPT-4-era BPE, **not** Qwen's tokenizer. We use it because:

- Token counts are within ~5–10% of Qwen on English text — close enough for chunk sizing.
- `tiktoken` is a single light dep, fast, and doesn't require loading a multi-GB model just to count tokens.
- The continuation pipeline only needs *consistent* chunking, not *exact* per-token alignment with the trainer.

The trainer itself uses Qwen's actual tokenizer when it formats sequences, so any small discrepancies wash out before training.

---

## 5. Output: ShareGPT format

All training data — Q&A and continuation alike — is written to JSONL in **ShareGPT** form:

```json
{
  "conversations": [
    {"from": "system", "value": "You are Daniel, an exile in Babylon..."},
    {"from": "human",  "value": "When the king called you out of the fire..."},
    {"from": "gpt",    "value": "Four is the number that stays with me..."}
  ],
  "data_type": "qa"
}
```

`data_type` ∈ `{"qa", "continuation"}` for the SFT pipeline. The trainer ignores extra keys, but they're handy when verifying the post-blend mix. DPO output uses the TRL format (`chosen` + `rejected` lists of `{role, content}` dicts) — see [dpo-data-generation.md](dpo-data-generation.md).
