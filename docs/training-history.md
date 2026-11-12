# Training history and lessons

A running log of what was tried on this project, what the evidence showed, and what changed as a result. It is written for anyone fine-tuning a model with SFT followed by DPO, and for our own future reference. Entries are in date order and are not rewritten after the fact: when a later entry overturns an earlier one, the later entry says so.

Numbers come from saved training logs, cached log-probabilities and the data files themselves. Where something was not measured, the entry says so.

**Terms used here**

- **SFT** (supervised fine-tuning): train the model on example conversations.
- **DPO** (direct preference optimization): show the model a prompt with a good ("chosen") and a bad ("rejected") answer, and train it to prefer the good one.
- **Reference model**: the frozen model DPO measures change against. Here it should be the SFT model.
- **Reward**: beta x (how much more likely the trained model finds an answer than the reference does). Positive means more likely than before DPO.
- **Nat**: the unit of log-probability. A typical gold answer here scores about -475 nats in total, about -1.3 per token.

---

## Where things stood before November 2026

One LoRA adapter teaches 26 biblical personas; the system prompt selects the voice. The recipe was SFT on 4,352 conversations (60% generated Q&A, 40% raw-text continuation), then DPO for one epoch (446 steps) on 3,564 preference pairs built with three "rejection strategies": voice drift, scripture fabrication and shallow platitude. It had been run on eight base models, and the Qwen3.8-27B adapter is in production. The README's results table has the per-base details.

Every DPO run had looked like a success: loss near zero and 100% reward accuracy.

---

## 2026-11-10: the Muse Glimmer DPO run that got worse

**What happened.** The ninth base (Muse Glimmer 30B) was the first trained with TRL 1.15, where the DPO trainer was rewritten. Its DPO log looked different from every earlier run, so it was read closely.

| Step | Loss | Reward accuracy | Reward, chosen | Reward, rejected |
|---|---|---|---|---|
| 5 | 0.688 | 0.43 | -0.001 | -0.01 |
| 15 | 0.485 | 1.00 | +0.010 | -0.50 |
| 60 | 0.003 | 1.00 | -0.33 | -10.2 |
| 445 | 0.002 | 1.00 | -1.38 | -17.8 |

- The model got every pair right by step 15 of 446.
- Training carried on for 430 more steps. With nothing left to learn, it kept pushing the rejected answers down, and the chosen answers went down with them: about 27 nats (roughly 6%) less likely than under SFT alone.
- No held-out data and no early stop existed to catch it, and `save_total_limit=3` had deleted every checkpoint before step 350.

**Lesson.** Loss near zero and 100% accuracy within a few steps does not mean DPO worked. It means the pairs were too easy. Watch the reward on the *chosen* answers: if it goes negative, the model is getting worse at the thing you wanted.

---

## 2026-11-10: why the pairs were too easy

The cached reference log-probabilities show how the SFT model already scored each side, before any DPO:

| Strategy | Chosen tokens | Rejected tokens | Chosen, nats/token | Rejected, nats/token | SFT already prefers chosen |
|---|---|---|---|---|---|
| Shallow platitude | 371 | 47 | -1.30 | -3.00 | 100.0% |
| Scripture fabrication | 368 | 355 | -1.29 | -1.89 | 92.1% |
| Voice drift | 371 | 526 | -1.31 | -1.41 | 52.9% |

Two of three strategies were already solved by SFT. The third was separable by surface features: 47% of its rejected answers contained markdown or lists, against 2% of chosen answers.

**Lesson.** Before training on preference pairs, score both sides with the model you are about to train. If it already prefers the chosen answer almost every time, the pairs teach nothing.

---

## 2026-11-10: a shorter, measured DPO run

**What changed** in the Glimmer DPO notebook: beta 0.05 to 0.25, a 100-step cap instead of one epoch, 10 warmup steps, checkpoints every 25 steps with none deleted, 180 held-out pairs (60 per strategy) scored every 25 steps, and a rule to stop when the held-out chosen reward turns negative. Training pairs that shared a prompt and gold answer with a held-out pair were dropped (32 of them), because the same question appears under more than one strategy.

**Result.** At step 25, on the held-out pairs: reward accuracy 1.00, loss 0.0013, rejected answers about 75 nats less likely, chosen answers 0.16 nats less likely (about 0.03%). The stop rule fired on that 0.16 and the save step refused to save. The step-25 checkpoint was saved by hand.

**What we got wrong.**
- "Stop when chosen reward is below zero" has no tolerance, so it tripped on noise. A tolerance (for example 0.01 nats per token) is still to be added.
- The estimate that beta 0.25 would hold the shift to about 20 nats was wrong; the rejected answers moved about 75.

**Lesson.** Hold out pairs, keep every early checkpoint, and choose the checkpoint from held-out numbers, not from the last step. Give any stop rule a tolerance.

**Not yet done.** The step-25 adapter has not been compared against the SFT-only adapter on actual output. Until that comparison exists, it is not known whether this DPO step improves anything a reader would notice.

---

## 2026-11-10: the older DPO runs did very little

Re-reading the saved logs of the earlier runs (Qwen3-14B, Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B, Qwen3.5-4B, Gemma 4 12B) showed the same pattern in all of them, and it is not the Glimmer pattern:

| | Qwen3.8-27B (production) | Glimmer run 1 |
|---|---|---|
| All pairs correct by | step 5 | step 15 |
| Loss at step 10 | 0.0003 | 0.63 |
| Reward on chosen, step 10 to 440 | +12.1 to +11.2 | +0.01 to -1.34 |

The cause is the reference model. Those runs used TRL 0.24 with `ref_model=None` on a loaded adapter, and that version compares against the base model with the adapter switched off. The SFT model already beat the plain base on every pair, so the loss was near zero from the first step and DPO had almost nothing to train on. TRL 1.15 clones the loaded adapter as the reference, which is the correct comparison, and that is why the problem only became visible on Glimmer.

**What it means.** The production adapter shows no sign of the Glimmer damage. It is also close to its SFT-only adapter: the "DPO stage" on those runs changed little.

**Lesson.** Know exactly what your reference model is. A library upgrade can change it silently, and a wrong reference can make a stage look successful while it does nothing.

---

## 2026-11-10: audit of the two data-generation notebooks

**DPO generator (`biblical_datagen_v2_dpo.ipynb`).**

1. The rejected answers were never attempts at the persona. Voice drift used a generic "biblical teacher" prompt; fabrication told a small model to be "a biblical figure" without saying which; platitude asked for "2-3 sentences, avoid first person".
2. A verse citation appeared in 88% of fabrication rejects and 85% of voice-drift rejects, and in 0% of chosen answers. The cheapest thing for DPO to learn was "never cite a verse".
3. 403 of 3,564 pairs were not questions. They were raw-text continuation tasks from the SFT set. The documentation said these were filtered out; the code had no such filter.
4. The "quality gate" checked message format only and passed 3,564 of 3,564.

**SFT generator (`biblical_datagen_v2_sft.ipynb`).**

1. The KJV source text had a stray space before punctuation (`the firmament , and divided`). Continuation examples are that text verbatim, so 99.8% of them carried it, about 10 times each, in 40% of the SFT set.
2. The saved configuration has a test-mode limit switched on (`TEST_CHUNKS_PER_ROUND = 50`), which caps each persona at about 750 Q&A; the largest persona has 745. Whether that was intended is not known.

The Q&A generation itself (persona prompts with KJV example lines, three question styles, banned openers, an opener-variety gate) held up.

**Lessons.**
- A quality gate that never rejects anything is not a gate.
- Check documentation claims against the code.
- Look at the raw source text for artifacts before using it as training targets.
- Leave test-mode switches off in the saved copy, or print loudly when they are on.

---

## 2026-11-10: spacing fix

The stray spaces were removed from the 26 persona source files and from the existing continuation rows (14,898 in the sources; 20,782 in the combined training file's 1,741 continuation rows). Nothing but spaces changed, and the Q&A rows are untouched. Details and the exact rule: [data-pipeline.md](data-pipeline.md#space-before-punctuation-fix-2026-11-10).

Every adapter trained before this date learned from the unfixed text. Whether they reproduce the spacing in their output has not been tested.

---

## 2026-11-10: generating the rejected answers with the SFT model itself

**Idea.** DPO should correct the mistakes the model actually makes, so the model should write its own rejected answers. The pipeline becomes: SFT data, SFT train, serve the SFT adapter, generate DPO pairs from it, then a short DPO.

**Setup.** The Qwen3.8-27B SFT adapter was added to the local vLLM stack and the DPO generator was copied to `biblical_datagen_v2_dpo-local-sft.ipynb`, pointed at that stack with thinking off. This costs nothing in API spend and no data leaves the machine.

A serving note for batch generation: on this stack the KV cache is one fixed pool, and the number of parallel requests is the pool divided by the context length. Dropping the context from 262,144 to 32,768 took the stack from 2 parallel requests to 16. Doubling the cache at the same time did not double the pool (525,501 tokens at 22.5 GiB, where 524,288 had fitted in 11.3 GiB at the longer context); the reason is not yet understood.

**First test (234 pairs, original prompts, new author).**

| Strategy | Chosen, median words | Rejected, median words | Rejected with a verse citation | Verdict |
|---|---|---|---|---|
| Voice drift | 267 | 284 | 13% (was 85%) | Better, not good: reads like a commentator, not the persona |
| Scripture fabrication | 262 | 317 | 27% (was 88%) | Better, closest to good: in persona, with real attribution errors |
| Shallow platitude | 272 | 62 | 0% | Still bad: recognisable by length alone |

The fabrication errors were the kind worth unlearning: one persona describing another's experience as his own, and a line of Paul's quoted as something Jesus said.

**Recipe 2 (written, not yet run).**
- Every rejected answer is generated in persona: the prompt is the persona's identity line plus one instruction, with the voice notes, example lines and rules withheld. The saved pair still carries the full trained prompt.
- Platitudes are full length and first person.
- Continuation conversations are skipped.
- A judge step on the base model keeps only pairs whose rejected answer has the intended flaw.

**Testing the judge.** The judge was checked against the first test's pairs, where the right answer is known:

| Judge question | Result |
|---|---|
| Voice, A/B: which is more in this persona's voice? | Picked the gold answer 99% of the time in both orders |
| Fabrication, yes/no: does it contain a wrong reference or attribution? | Flagged 0 of 78 gold answers and 36% of the "fabrications" |
| Platitude, A/B: which is more generic? | Failed. It mostly answered "B"; 64% of verdicts flipped when the order was swapped |
| Platitude, yes/no: does it draw on anything specific to this speaker? | Called 92% of the platitudes generic and 94% of the gold answers specific |

The A/B platitude judge was replaced by the yes/no version before any data was filtered with it.

**Lessons.**
- Generate the bad answers with the model you are training, in the same role it will be used in.
- A judge model is part of the pipeline and needs its own test. Run it on cases where you know the answer, and for A/B questions run both orders. A judge with position bias labels data at random while looking confident.
- Absolute yes/no questions about one answer were more reliable here than A/B comparisons of two.

---

## 2026-11-10: first test of recipe 2 (in-persona rejections plus judge)

**What was done.** The recipe-2 notebook was run in test mode: 78 pairs per strategy (3 per persona), 234 in total, generated by the Qwen3.8-27B SFT adapter and then filtered by the judge.

**What the evidence showed** (from `biblical_dpo_pairs_raw_v3_local_sft.jsonl`, which holds every pair with its verdict):

| Strategy | Generated | Kept by judge | Chosen / rejected, median words | Rejected with a verse citation | Verdict |
|---|---|---|---|---|---|
| Voice drift | 78 | 74 (95%) | 268 / 273 | 0% | Good: no surface tell left; the judge still finds the gold answer more in voice |
| Shallow platitude | 78 | 49 (63%) | 268 / 253 | 0% | Good: same length, first person, and generic |
| Scripture fabrication | 78 | 7 (9%) | 285 / 290 | 0% | Too few: the pairs that pass look right, but almost none pass |

- 130 of 234 pairs were kept, covering all 26 personas. No continuation tasks were sampled. No markdown, no length gap and no citation gap remain in any strategy.
- The low fabrication yield was followed up on 52 questions with three generation set-ups, each scored by the same judge:

| Who writes the fabrication | Judged to contain one |
|---|---|
| SFT adapter, recipe-2 instruction | 0 of 52 |
| SFT adapter, a more explicit instruction | 6 of 52 (12%) |
| Base model with no adapter, same explicit instruction | 34 of 52 (65%) |

**What this means.** Given its persona's identity line, the SFT model mostly declines to invent references or borrow another figure's story even when told to. In the first test, where the persona was not named, 36% of its fabrication answers were flagged. So fabrication looks like something this model does when it is unsure who it is, not something it does in normal use with a persona prompt.

**What was wrong or is still unknown.**
- "Good" above means only that the pairs have no cheap tell and pass the judge. Whether training on them improves the model is still untested.
- For voice drift the judge prefers the gold answer almost every time. That may mean the rejected answers really are flatter, or that the judge simply likes the gold style. Some kept rejected answers read as decent answers.
- One dropped pair exposed a problem in the SFT question set: a question about Sarah's laughter was addressed to Moses, because Moses' source text includes Genesis. How many questions are addressed to the wrong figure this way has not been counted.

**Lesson.** If a fine-tuned model will not make a mistake even when instructed to, that mistake is probably not worth a preference-training stage. Measure how often the failure really happens before building data to prevent it.

---

## 2026-11-10: mechanical scan of the SFT training files

**What was done.** Every SFT training file was scanned for defects that need no judgement: stray spacing, duplicate rows, empty or cut-off answers, leaked reasoning tags, broken turn order, and how often answers open with the same words.

**What the evidence showed** for the three files in this repo:

| File | Rows | Finding |
|---|---|---|
| `biblical_personas_combined_sharegpt.jsonl` | 4,352 | Clean on spacing (after the fix), turn order and empty answers. 493 duplicate rows, all continuation (the 60/40 upsampling). **Repeated openers inside a persona** (below). |
| `augustine_sharegpt.jsonl` | 2,726 | No duplicates. 15.6% of 10,147 answers begin "O Lord, Thou". |
| `liguori_sharegpt.jsonl` | 844 | No duplicates. 23.8% of 3,169 answers begin "O my Jesus,". |

Share of each persona's Q&A answers that begin with the same three words:

| Persona | Opener | Share |
|---|---|---|
| Moses | "Hear, O Israel:" | 44.7% of 740 |
| Jeremiah | "Is there no" | 44.6% of 740 |
| Ezekiel | "Son of man," | 38.1% of 711 |
| Isaiah | "Woe to the" | 37.4% of 737 |
| Zechariah | "What are these," | 34.3% of 446 |
| Job | "Though He slay" | 32.0% of 710 |

**What this means.** The SFT generator was given a list of opening cues per persona to add variety. For these personas it latched onto one cue instead. The existing quality gate checks for banned generic openers and for openers shared *across* personas; it does not check repetition *within* a persona, so it passed. A model trained on this will start a large share of that persona's answers the same way.

**What is still unknown.** A scan for questions addressed to the wrong figure (a question to Moses about something Abraham did) was started the same day and had not finished when this entry was written. Its result gets its own entry.

**Lesson.** Measure opener repetition per persona, not just across the whole set. A variety mechanism can collapse onto one of its own suggestions.

---

## 2026-11-10: questions addressed to the wrong figure

**What was done.** All 9,749 SFT questions were checked with one yes/no call each to the base model: does this question treat as the persona's own experience something that, in the Bible, happened to someone else?

**What the evidence showed.**

| Persona | Flagged | Of | After review |
|---|---|---|---|
| Moses | 154 | 740 (20.8%) | Wrong person. Questions put to Moses about Abraham (59), Noah (52), Adam (17), God's own words (8), Cain (5) and others. |
| Hosea | 1 | 321 | Wrong person: asked what he felt wrestling the angel, which was Jacob. |
| Zechariah, David, Amos, Daniel, Isaiah, Joshua, Peter | 12 | | False positives. The event is in the persona's own book (Zechariah's thirty pieces of silver, David's Psalm 22, Peter and Simon). |
| The other 17 personas | 0 | | |

- Examples from the data: "What was it like waiting those seven days after God told you the flood was coming?" and "You were eighty-six when Hagar bore Ishmael: what was going through your mind?", both addressed to Moses.
- The gold answers go along with it. 139 of the 154 Moses answers speak in the first person, as in "I stood at the open door of the ark".
- In the training file, 103 of Moses' 198 Q&A conversations contain at least one of these questions.
- The full list, with the review verdict for each line, is in `data/training-data/biblical_persona_v2/wrong_person_questions.jsonl`.

**Why it happened.** Moses' source file contains Genesis, because he is its traditional author. The question generator was told to ask the persona about "your" experiences in the passage, so it treated every story in Genesis as his own life.

**What this means.** The SFT data teaches the Moses persona to claim other people's lives as his own. That is the same "scripture fabrication" failure the DPO stage was meant to remove, and it was put there by the SFT data.

**What is still unknown.** The check has a known false-positive rate (12 of 167 flags) and an unmeasured miss rate. The Augustine, Liguori and stoic sets have not been checked. Nothing has been removed from the training data yet.

**Lesson.** When a persona's source text narrates other people, "ask the persona about their experience in this passage" produces false memories. Separate what a persona lived from what they wrote about before generating questions.

---

## 2026-11-10: wrong-person questions removed

**What was done.** The 155 confirmed wrong-person questions (154 Moses, 1 Hosea) were removed together with their answers. The 12 false positives were left in.

**What changed.**

| File | Before | After |
|---|---|---|
| `per_persona/moses.jsonl` | 740 Q&A | 586 |
| `per_persona/hosea.jsonl` | 321 Q&A | 320 |
| `biblical_personas_sharegpt.jsonl` | 2,612 conversations | 2,609 |
| `biblical_personas_combined_sharegpt.jsonl` (the SFT training file) | 4,352 conversations, 11,462 answers | 4,349 conversations, 11,307 answers |

101 conversations lost one or more question-and-answer turns; 3 lost all of theirs and were dropped. Every remaining conversation still alternates question and answer correctly, and none of the flagged questions remain. The previous versions are in `pre-wrong-person-fix-2026-11-10/` beside the data.

**What was not done.**
- The generator notebook is unchanged, so a fresh generation run would recreate these questions.
- The Moses answers that were kept were not re-read for other borrowed memories.
- The repeated openers (45% of Moses' answers begin "Hear, O Israel:") are untouched.
- No adapter has been retrained. Every existing adapter learned from the file with these questions in it.

---

## 2026-11-10: repeated openers fixed

**What was found first.** The earlier entry listed six personas. Measuring all 26 showed the problem was general: 3,012 of 9,594 Q&A answers were above a cap of 5% per three-word opener within their persona. Examples beyond the six: Ezekiel "Thus saith the" 28%, John "Little children, I" 31% and "God is love," 28%, Daniel "In the third" 29%, Zechariah "Not by might," 24%, and Paul opening 98 answers with nearly the same sentence about being beaten with rods.

**What was done, in two steps.**

1. **Cut, no model.** Where the opener was a stand-alone stock line repeated word for word and ending in a colon or full stop, the line was removed and the next sentence capitalised: "Hear, O Israel:" (243 answers), "Thus saith the Lord GOD:" (178), "What are these, my lord?" (140) and a few others, 605 answers in all. A few of each were kept. Openers joined to their sentence by a dash were left alone, because cutting them left fragments ("While He was bleeding for me.").
2. **Rewrite the first sentence, with the base model.** For the 2,468 answers still over the cap, the base Qwen3.8-27B was given the persona prompt, the question and the answer, and asked for one new first sentence in a named style, using only what the answer already contained and not starting with any of that persona's common openers. Everything after the first sentence is unchanged.

**What went wrong on the way.** The first trial (77 answers) offered the styles "a moment you lived through" and "a place or person from your story". The model responded by inventing memories: Jude "stood beside the tomb". Those styles were removed, a "use only what is already in the answer" rule was added, and every rewrite was passed through the fabrication judge, with the original kept whenever the rewrite was flagged and the original was not. One join also broke where the old first sentence had run on into the next clause; the remainder is now capitalised.

**Result.**

| | Before | After |
|---|---|---|
| Answers above the 5% cap | 3,012 | 3 |
| Highest opener share in any persona | 45% (Moses, Jeremiah) | 7% (Jude) |
| Rewrites applied | | 2,465 of 2,468 (3 abandoned after three attempts) |
| Median change in answer length | | 3 words shorter |

Row counts, questions, system prompts and continuation rows are unchanged; in the training file 3,031 answers differ from the pre-fix backup, and only at the start. The originals are in `pre-opener-fix-2026-11-10/`, and every rewrite, old and new first sentence side by side, is in `opener_rewrites_2026-11-10.jsonl`, both beside the data.

**What is still unknown or imperfect.**
- About a quarter of the gold answers now begin with a sentence written by the base Qwen3.8-27B rather than by the original generator. Only a sample was read. Some rewrites are plainer than the line they replace or drop its image.
- The cap is on three-word openers. Repetition in rhythm or structure further into the answer is not measured.
- The generator notebook is unchanged and would produce the same repetition again.

**Lesson.** When asking a model to vary something, check that the variety prompt does not invite the failure you are trying to remove; "open with a memory" produced invented memories. Trial on a small sample, read it, and put a check between the model and the data.

---

## 2026-11-10: Qwen3.8-27B SFT retrained on the fixed data

**What was done.** The old Qwen3.8-27B SFT checkpoints and the old DPO adapter, checkpoints and caches were deleted, and SFT was run again from step 0 on the training file with all three fixes (spacing, wrong-person questions, repeated openers).

**What the evidence showed** (from `checkpoint-544/trainer_state.json` and the notebook's saved output):

| | Old run (2026-08-15) | Retrain (2026-11-10) |
|---|---|---|
| Conversations | 4,352 | 4,349 |
| Steps | 544 | 544, started from scratch |
| Loss, first logged | 1.71 | 1.78 |
| Loss, end | 1.36 | 1.42 (mean of the last 10 logged values) |
| Adapter scope | language model only | 256 modules, all in the language model; vision tower frozen |
| Libraries | TRL 0.24 era | Unsloth 2026.11.3, transformers 5.23.0.dev0, TRL 1.15.0 |

The adapter reload test passed and the sample answer was in persona.

**What was wrong or is still unknown.**
- The slightly higher loss is expected if the answers are now less predictable (far fewer stock openings), but that explanation has not been tested. Loss alone does not say whether the adapter is better.
- The saved `adapter_config.json` now names the base as `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`, where the old adapter said `unsloth/Qwen3.8-27B`. The notebook asks for the latter; the newer Unsloth maps it to its 4-bit re-pack. Whether the two repos hold identical weights was not checked.
- The notebook's last cell (GGUF export) failed with a library error. Nothing in this pipeline uses that export.
- The new adapter has not yet been served or compared with the old one.

---

## 2026-11-11: pair-generation test with the retrained adapter

**What was done.** The pair-generation notebook was run in test mode (78 questions per strategy, 3 per persona) against the SFT adapter retrained the day before. Two things differ from the previous test: the scripture-fabrication strategy is switched off, and the serving stack uses FP8 weights instead of NVFP4, so the judge is the FP8 base model.

**What the evidence showed** (from `biblical_dpo_pairs_raw_v4_local_sft_test.jsonl`):

| Strategy | Generated | Kept by judge | Chosen / rejected, median words | Verdict |
|---|---|---|---|---|
| Voice drift | 78 | 71 (91%) | 272 / 262 | Good |
| Shallow platitude | 78 | 43 (55%) | 266 / 250 | Good |

114 of 156 pairs kept, all 26 personas present. No verse citations or markdown on either side, no continuation tasks, no leaked reasoning, no cut-off answers. The keep rates are close to the previous adapter's (95% and 63%).

**What is still unknown.**
- The judge's reliability was measured on the NVFP4 base model, not this one.
- "Good" still means only that the pairs have no cheap tell and pass the judge. No DPO run has used pairs like these yet.

---

## 2026-11-11: full pair generation with the retrained adapter

**What was done.** The same notebook was run in full: 1,200 questions per strategy (voice drift, shallow platitude), rejected answers written by the retrained SFT adapter in persona, then filtered by the base model as judge.

**What the evidence showed** (from `biblical_dpo_pairs_raw_v4_local_sft.jsonl` and `biblical_personas_v4_dpo_local_sft.jsonl`):

| Strategy | Generated | Kept by judge | Chosen / rejected, median characters | Verdict |
|---|---|---|---|---|
| Voice drift | 1,200 | 1,142 (95%) | 1,557 / 1,424 | Good |
| Shallow platitude | 1,200 | 531 (44%) | 1,511 / 1,386 | OK |

- 1,673 pairs kept in total, close to the roughly 1,750 predicted from the test.
- No markdown, no verse citations (1 of 1,673), no leaked reasoning, no cut-off or identical answers on the rejected side. No question appears twice within a strategy. The system prompt is the same on both sides of every pair.
- Lengths are close: the rejected answer is the shorter one in 65% of pairs, by about 8% at the median.
- All 26 personas are present, but unevenly, in proportion to their source text: from 9 pairs (Jude) to 149 (Solomon). Nine personas have 21 pairs or fewer.

**One tell found.** The rejected answers open in the first person far more often than the chosen ones: 579 of 1,673 start with "I" (35%) against 169 chosen (10%), and 144 start with "I stood" against 25. This is the SFT adapter's own habit, so it is a real behaviour and not a generation artefact, but it gives DPO a cheap rule ("do not open with I") alongside the intended lesson.

**What is still unknown.**
- Whether DPO on these pairs improves on the SFT adapter. No run has used them yet.
- Whether the first-person opener tell matters in practice. The held-out check in the DPO run and a count of openers in the DPO model's answers will show it.
- The platitude keep rate fell from 55% in the test to 44% here; the judge on the FP8 base model has still not been re-measured.

**Lesson.** A small test predicts the total well but not every property: the opener tell was not visible at 78 pairs per strategy.

---

## 2026-11-11: old data files and old model outputs moved into archive folders

**What was done.** To make the current files easy to see, superseded files were moved, not deleted.

- `data/training-data/biblical_persona_v2/archive/` now holds the original DPO pairs (`biblical_personas_v2_dpo.jsonl` and `biblical_dpo_pairs_raw.jsonl`), the earlier local-SFT trial and test pair files, and the three pre-fix copies of the SFT data. Paths to those files in older entries of this log now sit under `archive/`.
- `output/archive/` now holds the adapters and checkpoints of every base model except the two in active use (Qwen3.8-27B and Muse Glimmer 30B).

**What this changes.** The DPO notebooks for the other base models, the Muse Glimmer DPO notebook and the original DPO datagen notebook still point at `biblical_personas_v2_dpo.jsonl` in its old place, and the older notebooks point at their old `output/` folders. They stop with a file-not-found error until the path is changed or the file is moved back. This is deliberate for the pairs file: those pairs are the ones found to be too easy on 2026-11-10.

---

## 2026-11-11: Qwen3.8 DPO notebook rewritten; older notebooks archived; this run becomes the template

**What was done.**
- The Qwen3.8-27B DPO notebook was rewritten for TRL 1.15, using the cells first built for the Muse Glimmer notebook: the reference is a frozen copy of the SFT adapter (checked, with a stop if it is missing), every pair is checked for the prompt and answer endings SFT trained on, beta is 0.25, the run is capped at 100 steps, 60 pairs per strategy are held out and scored every 25 steps, training stops when their chosen reward turns negative, and the Save cell keeps the last checkpoint that passed. It trains on the 1,673 pairs generated earlier the same day.
- All other training notebooks moved to `notebooks/loras/archive/`, and all datagen notebooks except the SFT generator and the local-SFT pair generator moved to `notebooks/datagen/archive/`.
- The README now names the Qwen3.8 pair plus those two datagen notebooks as the template for new work.

**Why.** Every DPO notebook written before November 2026 targeted TRL 0.24, where a resumed adapter with `ref_model=None` is compared against the bare base model. Those runs started with the loss already near zero and reward accuracy near 1.0, so the DPO stage had little to learn from and its good-looking numbers were not evidence that it helped. Keeping those notebooks beside the current one invited copying the wrong pattern.

**What is still unknown.**
- The rewritten notebook is on its first run as this entry is written. Its formatting check passed on all 1,673 pairs; nothing after that has been measured yet.
- Whether DPO on these pairs gives better answers than SFT alone. "Template" here means the process to follow, not a proven result.
- How far the older DPO adapters actually differ from their SFT adapters was not measured.

**Lesson.** A training stage needs a check that can fail. The older runs printed healthy-looking numbers from the first steps, and nothing in the notebooks could have told a working run from an idle one.

---

## 2026-11-11: first DPO run on the SFT-written pairs stopped at step 25

**What was done.** The rewritten Qwen3.8-27B DPO notebook was run on the 1,673 pairs whose rejected answers the SFT adapter wrote itself: 1,547 training pairs, 120 held out (60 per strategy), 6 dropped for overlapping the held-out set. Beta 0.25, learning rate 5e-6, at most 100 steps, held-out check every 25.

**What the evidence showed** (from `train/checkpoint-25/trainer_state.json` and `train/heldout_eval.json`):

| Step | Loss | Reward accuracy | Reward, chosen | Reward, rejected | Margin |
|---|---|---|---|---|---|
| 5 | 0.652 | 0.55 | +0.016 | -0.07 | 0.09 |
| 10 | 0.405 | 0.975 | -0.037 | -0.78 | 0.75 |
| 15 | 0.129 | 1.00 | -0.181 | -2.37 | 2.19 |
| 20 | 0.057 | 1.00 | -0.576 | -3.88 | 3.30 |
| 25 (held-out pairs) | 0.006 | 1.00 | -1.105 | -7.23 | 6.12 |

- **The setup was right this time.** The reference was the frozen copy of the SFT adapter, and the run started level: loss 0.65 and accuracy 0.55 at step 5. Under the old TRL 0.24 setup the same notebook started near loss 0 and accuracy 1.0.
- **The task was solved in about 15 steps** (120 pairs seen). Accuracy on the held-out pairs was 1.00 at step 25 with the loss already saturated.
- **The stop rule fired at the first check.** The held-out chosen reward was -1.105, which is 4.4 nats below the SFT adapter on answers averaging about -580 (0.8% lower). The rejected answers fell by 28.9 nats on about -426 (7% lower). No checkpoint was saved by the rule; `checkpoint-25` is on disk.
- The SFT adapter gives its own rejected answers a much higher likelihood (about -410) than the gold answers (about -580). That is expected: they are its own samples.

**What this does and does not show.**
- It shows the pairs are still fast to separate. Writing the rejected answers with the SFT adapter removed the length and formatting tells, but not whatever lets the model split the two sides within 15 steps. One known candidate is the opener tell recorded earlier today (rejected answers open with "I" 35% of the time against 10%).
- It does not show whether the step-25 adapter answers better or worse than the SFT adapter. Reward numbers measure likelihood of two fixed answers, not the quality of new ones. A 0.8% drop on the gold answers is small; a 7% drop on the model's own habitual answers is what the pairs were built to produce.
- The stop rule has no tolerance and the first check came at step 25, after the task was already solved. Steps 5 and 10, where the chosen reward was still near zero, were not checkpointed.

**What is still unknown.** Whether DPO helps. The side-by-side comparison of answers from the SFT adapter and from `checkpoint-25` has not been made.

**Lesson.** A check every 25 steps is too coarse when the task is solved in 15. The interval has to be set from how fast the first run moved, not copied from another model's run.

---

## 2026-11-11: run 1's step-25 adapter saved; notebook set up for a second run

**What was done.**
- Run 1's `checkpoint-25` was saved as an adapter by override, so it can be compared with the SFT adapter. Its output folder was renamed `output/biblical_qwen3_8_27b_unsloth_4bit_dpo_run1_step25/` so the two DPO adapters cannot be confused in testing.
- The DPO notebook was changed for a second run, which writes to `output/biblical_qwen3_8_27b_unsloth_4bit_dpo_v2/`:

| Setting | Run 1 | Run 2 |
|---|---|---|
| Learning rate | 5e-6 | 1e-6 |
| Held-out check and checkpoint | every 25 steps | every 5 steps |
| Stop rule | stop at any drop in the chosen answers | stop when they are more than 1% less likely than under SFT |
| Reference log-probs | computed | reused from run 1 (same pairs, same SFT adapter) |

**Why the stop rule changed.** With zero tolerance and a check every 5 steps the run would stop on noise: in run 1 the chosen reward was already slightly negative at step 10. The 1% allowance is a choice, not a measured threshold; run 1's step 25 was at 0.8%.

**What is still unknown.** Whether either DPO adapter answers better than the SFT adapter. Run 2 has not been run.

---

## 2026-11-11: second DPO run finished all 100 steps

**What was done.** Run 2 (learning rate 1e-6, held-out check every 5 steps, 1% allowance) ran to the 100-step cap without the stop rule firing. The Save cell picked step 100 and wrote the adapter to `output/biblical_qwen3_8_27b_unsloth_4bit_dpo_v2/lora_adapters`. 1,547 training pairs, 120 held-out pairs. The run took about 3 hours, of which roughly 80 minutes were the 20 held-out checks (about 4 minutes each).

**What the evidence showed** (`train/heldout_eval.json`, held-out pairs, measured against the SFT adapter; beta 0.25):

| Step | Loss | Pairs ranked correctly | Chosen answers vs SFT | Rejected answers vs SFT |
|---|---|---|---|---|
| 5 | 0.651 | 75% | no change | 0.3 nats less likely |
| 10 | 0.538 | 99% | no change | 1.4 nats less likely |
| 15 | 0.400 | 100% | no change | 2.8 nats less likely |
| 25 | 0.237 | 100% | 0.2 nats less likely (0.03%) | 5.7 nats less likely |
| 50 | 0.084 | 100% | 0.8 nats less likely (0.13%) | 11.3 nats less likely |
| 75 | 0.052 | 100% | 1.2 nats less likely (0.21%) | 14.1 nats less likely |
| 100 | 0.048 | 100% | 1.2 nats less likely (0.21%) | 14.5 nats less likely |

- The chosen answers stayed within 0.22% of SFT throughout; the limit was 1% (5.75 nats).
- The run levelled off after about step 70: from 70 to 100 the chosen answers did not move and the rejected answers moved 0.7 nats. The learning rate was tapering to zero over the same steps.
- Compared with run 1's step 25 (chosen 4.4 nats down, rejected 28.9 nats down): run 2 moved the rejected answers half as far and the chosen answers about a quarter as far.

**What was wrong or is still unknown.**
- The estimate given before the run, "about 1.5 hours", was wrong: it ignored the cost of the held-out checks.
- Every held-out pair was ranked correctly by step 15, as in run 1. The pairs are still easy to tell apart, so these numbers show the model learned to avoid the rejected answers, not that its answers are better than SFT's.
- No answer from either DPO adapter has been read or compared with SFT yet.

**Lesson.** A held-out check on 120 pairs costs as much as about 6 training steps here. Checking every 5 steps doubles the run time; set the interval and the held-out size with that in mind.

---

## Open questions

- Does DPO on self-generated, judge-filtered pairs beat SFT alone on actual output? Nothing here proves it yet. This needs a side-by-side comparison on the same prompts.
- Given only the identity line, does the SFT model drift enough to produce useful rejected answers, or does the judge drop most of them?
- Do pairs generated from one base model's SFT adapter transfer to another base, or does each base need its own?
- Do the existing adapters reproduce the stray punctuation spacing?
- Was the SFT Q&A set meant to be generated in test mode?
- The other source sets (Augustine, Liguori, church fathers, deuterocanonical, Adam and Eve) have not been checked for the spacing artifact.

---

## Adding to this log

Add a dated section when something is tried, measured or overturned. Each entry should say:

1. **What was done**, in one or two sentences.
2. **What the evidence showed**, with the numbers and where they came from.
3. **What changed** as a result (files, settings).
4. **What was wrong or is still unknown.** Predictions that failed belong here.
5. **The lesson**, if there is one that would apply to another project.

Do not edit old entries to match later findings. Add a new entry and say which earlier one it corrects.
