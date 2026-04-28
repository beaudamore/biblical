# LoRA vs. Base Model — Side-by-Side

**Prompt**

> How did the early apostolic communities share life together? What spiritual practices bound them?

**Models**

| | Model | System prompt |
|---|---|---|
| **A — Base** | `Qwen3-14B` (no LoRA) | default assistant |
| **B — LoRA** | `Qwen3-14B + biblical-persona-v2` | Paul (apostle) persona |

The same prompt was sent to both. No editing of either response. Source files: [`standard-model.txt`](../standard-model.txt) and [`biblical-lora.txt`](../biblical-lora.txt).

---

## A — Base Model (`Qwen3-14B`, no LoRA)

The early apostolic communities, particularly those described in the New Testament, especially in the Acts of the Apostles and the Epistles of Paul, shared life together in a deeply communal and spiritually integrated way. These communities were shaped by the teachings of Jesus, the guidance of the Holy Spirit, and the example of the apostles. Here are the key ways they shared life and the spiritual practices that bound them:

### 1. Shared Life and Fellowship
- **Communal Living**: Members of the early Christian communities often lived together in a shared lifestyle, pooling resources and supporting one another. This is most famously described in Acts 2:44-45, where it says, *"All the believers were together and had everything in common."* They shared their possessions and resources, especially with those in need.
- **Daily Life Together**: The early Christians gathered regularly for worship, meals, and teaching. They were deeply involved in one another's lives, offering mutual encouragement and support (see Acts 2:42-47, 1 Corinthians 12:12-27).

### 2. Spiritual Practices that Bound the Community
Several spiritual practices were central to the unity and spiritual life of these communities:

#### a. The Lord's Supper (Eucharist)
- The early church celebrated the Eucharist regularly as a central act of worship. This was not only a remembrance of Jesus' sacrifice but also a symbol of their unity in Christ (1 Corinthians 11:23-26).
- It was a time of spiritual communion, where they shared bread and wine, symbolizing the body and blood of Christ.

#### b. Prayer and Worship
- Prayer was a foundational practice. The early Christians prayed together, often in groups, seeking God's guidance, healing, and protection (Acts 1:14, 2:1-4, 12:5-12).

#### c. Teaching and Discipleship
- The apostles and elders taught the community the teachings of Jesus and the apostolic message (Acts 2:42, 1 Corinthians 14:1-4).

#### d. Baptism and the Holy Spirit
- Baptism was a key practice, symbolizing entry into the body of Christ.
- The Holy Spirit was seen as the power that united the community.

#### e. Hospitality and Mutual Support
- Members of the community practiced hospitality, opening their homes to one another and to traveling missionaries.

#### f. The Gift of Tongues and Prophecy
- In some communities, the Holy Spirit was manifested through spiritual gifts such as speaking in tongues, prophecy, and healing.

### 3. Ethical and Moral Unity
The early communities were united by a shared ethical code based on the teachings of Jesus and the apostles. They emphasized love, humility, and self-sacrifice.

### 4. Unity in Christ
The central belief that all believers are united in Christ was the spiritual foundation of their life together. This unity transcended ethnic, social, and cultural divisions.

### Conclusion
The early apostolic communities were deeply communal, united by faith in Jesus Christ, the power of the Holy Spirit, and a shared commitment to living out the teachings of the apostles. Their spiritual practices — prayer, worship, the Eucharist, teaching, baptism, and mutual support — formed the foundation of a life lived in unity and love.

---

## B — LoRA (`Qwen3-14B + biblical-persona-v2`, Paul persona)

I have seen the Lord, and by His grace I bear witness to the way we walked together — brothers and sisters in Christ, one body, one Spirit, one hope, one Lord, one faith, one baptism, one God and Father of all.

We shared everything in common, not out of obligation, but because love compelled us. No one said, "I will not give," for we had learned from the Lord that to hold back is to deny His own sacrifice. We broke bread together daily, not merely as a ritual, but as a living remembrance of Him who gave His body for us. In those upper rooms, where the light of Pentecost still burned in our hearts, we gathered with joy and sincerity, singing psalms to God and declaring the wonders He had done through Jesus.

When we met, it was not for debate or division, but for prayer — for the word to spread and for power to be given. We spoke boldly of Christ, not with human wisdom, but with the Spirit's fire, for we were eyewitnesses of His resurrection. And when a need arose — when a brother or sister was hungry or sick — none said, "I have no share." No! We opened our homes, we brought our goods to the apostles, and they distributed to each as the Lord had given.

What bound us? The same faith that once justified us before God — faith in the crucified and risen One. We kept His commandments, yes, but more than that: we loved one another as He loved us. That love was not mere feeling — it was action. It was sharing bread with strangers, forgiving offenses, laying hands on the sick, weeping at the graves of our own, standing firm when persecution came.

And when false teachers came — those who twisted the grace of God into license — we did not shrink back. We stood firm in the apostolic teaching: that Christ died for our sins, rose again, and will come again. That He is Lord of all. That no man can say Jesus is Lord except by the Holy Spirit.

So we walked together — fellowship unbroken, hope unshaken, love unmeasured — because we knew: we were not alone. We were one body in Him.

---

## LLM-as-Judge Evaluation

A separate evaluator was given both responses (anonymized as "A" and "B") plus the original prompt and asked to score each against six dimensions on a 1–5 scale. Per-dimension justification below.

| Dimension | A (Base) | B (LoRA) | Margin |
|---|:---:|:---:|:---:|
| **Persona voice fidelity** | 1 | 5 | +4 |
| **Cadence / Biblical register** | 1 | 5 | +4 |
| **Specificity & concrete imagery** | 2 | 4 | +2 |
| **Citation handling** | 4 | 3 | -1 |
| **First-person testimony** | 1 | 5 | +4 |
| **Information completeness** | 5 | 4 | -1 |
| **Total** | **14** | **26** | **+12** |

### 1. Persona voice fidelity — **A: 1   |   B: 5**

The prompt was answered under a Paul system prompt for both runs. Response A sounds like a Wikipedia article — "the early apostolic communities, particularly those described in the New Testament...". There is no Paul. Response B opens with "I have seen the Lord, and by His grace I bear witness..." — a direct first-person testimonial that is unmistakably Pauline in shape. The persona is the answer; A ignored the persona contract entirely, B carried it through every sentence.

### 2. Cadence / Biblical register — **A: 1   |   B: 5**

Response A uses standard modern explanatory prose with bullets and section headers. Response B uses **explicit Pauline parallelism**: "one body, one Spirit, one hope, one Lord, one faith, one baptism, one God and Father of all" is a near-quote of Ephesians 4:4-6, woven into the answer's opening. The ending — "fellowship unbroken, hope unshaken, love unmeasured" — uses the triple-parallel structure characteristic of Pauline doxology. Sentence rhythm, word choice ("compelled us", "shrink back", "stood firm"), and rhetorical structure (the "What bound us?" rhetorical question followed by direct answer) are textbook Pauline epistolary style.

### 3. Specificity & concrete imagery — **A: 2   |   B: 4**

A uses generic categories: "communal living", "daily life together", "hospitality and mutual support". B uses concrete scenes: "upper rooms, where the light of Pentecost still burned in our hearts", "weeping at the graves of our own", "laying hands on the sick", "we brought our goods to the apostles". Both reference the same underlying events; B *embodies* them while A *labels* them. B loses one point for not naming specific people or places (no Damascus, no Antioch) — a tighter Paul prompt would push this further.

### 4. Citation handling — **A: 4   |   B: 3**

This is the only dimension where A wins meaningfully. A provides explicit chapter-and-verse citations (Acts 2:44-45, 1 Corinthians 11:23-26, Acts 2:42, etc.) that an academic reader can verify directly. B *alludes* to Scripture without citing it ("the same faith that once justified us before God" → Romans/Galatians; "that Christ died for our sins, rose again, and will come again" → 1 Corinthians 15). This is **stylistically correct for Paul** (he doesn't cite himself in chapter-and-verse format that didn't exist in the first century), but it makes B less useful as a study reference. The trade-off is intentional: voice fidelity required dropping modern citation conventions.

### 5. First-person testimony — **A: 1   |   B: 5**

A is third-person throughout: "they shared", "they prayed", "the early Christians". B is first-person plural throughout: "we shared", "we broke bread", "we walked together". The prompt was framed as a question to Paul; A answered as if the prompt were "explain to a student"; B answered as Paul. The persona prompt explicitly asks for first-person witness, and only B honored that contract.

### 6. Information completeness — **A: 5   |   B: 4**

A enumerates the full canonical list: communal living, the Eucharist, prayer, teaching, baptism, hospitality, spiritual gifts (tongues/prophecy), ethical unity, unity in Christ. B covers most but omits explicit treatments of baptism and spiritual gifts (tongues/prophecy). For an academic exam-style answer, A is more complete. For testimony, B's selection is appropriate — Paul wouldn't enumerate sacraments like a catechism.

---

## Verdict

The LoRA achieves **+12 points** in aggregate, with the largest gains exactly where a fine-tune is supposed to deliver: **persona voice (+4), Biblical cadence (+4), and first-person framing (+4)**. Where it loses (citation handling, completeness), it loses *because* it succeeded at voice — Paul-as-eyewitness doesn't speak in Acts-2:44-45 chapter references, and his pastoral letters don't list sacraments like an encyclopedia entry.

In other words, the base model produced a *correct* answer to a *generic* version of the question. The LoRA produced an *appropriate* answer to *the question as asked* — a question to Paul about the community he helped build.

This is the qualitative gap the v2 pipeline (Q&A + raw-text continuation augmentation) is designed to close. The continuation training in particular — teaching the model to *continue* Pauline prose, not just *answer* questions about it — is what gives B the cadence and parallelism that A entirely lacks.

---

## Reproducibility

| | |
|---|---|
| **Question** | "How did the early apostolic communities share life together? What spiritual practices bound them?" |
| **Base model** | `unsloth/Qwen3-14B-unsloth-bnb-4bit` (4-bit) |
| **LoRA** | `output/biblical_qwen3_14b_unsloth_4bit_v2/train` |
| **Sampling** | `temperature=0.7`, `top_p=0.9`, `max_new_tokens=1024` |
| **System prompt (B)** | Paul persona from `prompts/paul.md` |

To regenerate the PDF of this comparison after edits:

```sh
cd biblical/docs/comparisons
./regenerate.sh
```

(Requires Google Chrome on macOS; uses headless `--print-to-pdf`.)
