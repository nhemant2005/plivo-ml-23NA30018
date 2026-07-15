# Questionable Items & Suggested Improvements

Baseline (Run 0): constant lr=3e-4, Adam, batch=8, block=128, 4L/4H/160d,
untied embeddings, flat `normal_(std=0.05)` init, dropout=0, no grad clip,
byte tokenizer (vocab 256). **Dev bpb: 2.3718**, params: 1,339,840 (67% of
2M cap), steps: 2000.

**Current best (Run 9, shipped as `ckpt.pt`):** char tokenizer + n_embd=184 +
grad clip + batch=32 → bpb **1.9863**, 1,994,560 params (99.7% of cap). Run 5's
batch=32 win stacked cleanly on Run 8's tokenizer+width recipe: -0.2529 stacked
vs -0.2690 isolated at byte level, i.e. ~94% of the isolated effect survived.
Weight tying retested at vocab=913, still regresses (Run 7) — closed.

**Open at time of submission (see "Known-open levers" in SUMMARY.html):** the LR
was never varied in any run — all ten used lr=3e-4 inherited from the baseline.
Phase 1's schedule runs held peak = the baseline constant, so they tested LR
*magnitude*, not schedule *shape*; sorted by mean LR all four points are
monotonic with the best at the highest LR ever tested. Run 10 (lr=1e-3 at
batch=32) is the one probe that fit in the remaining time.

Status legend: ✅ closed · 🔲 planned

---

## Phase 1 — trainer-only (CLOSED, see RUNLOG Run 0-4)

Only grad clip helped. All others regressed and were reverted:

| Item | Result | Verdict |
|---|---|---|
| LR schedule (3 variants: warmup+cosine→0/floor, no-warmup+cosine→floor) | 2.60–2.70 | regression — decay throttles a still-improving 2000-step run |
| Weight tying alone | 2.4122 | mild regression at n_embd=160 — retested at vocab=913, still regresses (Run 7), ✅ closed for good |
| GPT-2 scaled init (std=0.02 + residual scaling) | 2.4519 | regression — 4 layers is too shallow for this to pay off |
| **Gradient clipping (max_norm=1.0)** | **2.3526** | **win, kept** |
| AdamW + weight_decay=0.1 (stacked on clip) | 2.3597 | regression — 2000 steps isn't enough to overfit, decay just costs capacity |

Not worth revisiting: scaled init, AdamW/decay.

### ⚠️ Correction (post-audit): the LR schedule row above is confounded

It should **not** be read as "schedules don't work." All three variants held
peak LR equal to the baseline's constant 3e-4, so each ran at roughly half the
baseline's *average* LR — they varied LR magnitude, not schedule shape. Sorted
by mean LR, the four Phase 1 points are cleanly monotonic:

| Run | Mean LR over run | Dev bpb |
|---|---|---|
| 1 (warmup+cosine→0) | 1.50e-4 | 2.6976 |
| 1b (cosine→10% floor) | 1.63e-4 | 2.6338 |
| 1c (cosine→floor, no warmup) | 1.65e-4 | 2.6076 |
| 0 (constant baseline) | **3.00e-4** | **2.3718** |

More LR, better bpb, every time — and the best result sits at the *highest LR
ever tested*, i.e. the edge of the search range, never an interior optimum. Run
1c's conclusion identified the mechanism correctly ("still improving fast at
full 3e-4 constant") and then drew the opposite inference from it. A schedule
earns its keep by permitting a **peak above** the best stable constant; that
hypothesis was never tested. **Status: open, not closed.**

### ⚠️ Correction: weight tying is closed at small vocab, not in general

The regression is roughly constant (+0.0404 at V=256, +0.024 at V=913) while
the params freed scale linearly with vocab: 41K at V=256, 146K at V=913, ~750K
at V=4096. The trade only becomes interesting at BPE-scale vocab, which this
run never reached. Correctly closed for the vocab sizes actually tried.

---

## Phase 2 — batch/block retune ✅ CLOSED (batch increase banked, LR retune deferred to Phase 4)

### 1. Expose `--block` in train.py ✅ DONE
Added `--block` CLI arg to `train.py`. `evaluate.py` needed no changes — it
rebuilds `Config` from the checkpoint's saved dict automatically.

### 2. Batch size increase ✅ CLOSED — win, bpb 2.3526 -> 2.0836 (Run 5)
Batch scaling is ~linear cost, block scaling is quadratic (attention) — batch
is the cheap lever, block is the expensive one. Measured timing:

| batch | block | ms/step | time/2000 steps | tokens seen |
|---|---|---|---|---|
| 8 | 128 | 58 | 1.9 min | 2.05M |
| 32 | 128 | 223 | 7.4 min | 8.19M (>1 epoch) |
| 16 | 256 | 227 | 7.6 min | 8.19M |
| 64 | 128 | 421 | 14.0 min | 16.4M |

**Deferred to Phase 4 lock-in** — batch=32 triples wall-clock/run, which is
too slow for Phase 3's tokenizer iteration. Reverted to batch=8 for now; the
2.0836 result is banked and gets re-applied at final lock-in.

### 3. LR retune at new batch size 🔲 DEFERRED TO PHASE 4
lr=3e-4 was only ever validated at batch=8; Run 5 used it unchanged at
batch=32. One isolated comparison run (e.g. ~2x lr) at batch=32, done last
alongside re-stacking the batch increase — not worth the wall-clock cost
until the tokenizer (and possibly width/tying) choices are locked.

---

## Phase 3 — tokenizer swap: char-level + byte-fallback ✅ CLOSED (revised from "BPE")

Corpus has only **657 unique codepoints** (7.3M bytes / 5.7M chars, 28%
byte-overhead from encoding alone; Devanagari is 14% of chars but 33% of
bytes — the 3x cost is real). This makes a **plain char-level tokenizer**
viable and much lower-risk than merge-trained BPE, while still removing most
of the byte-encoding tax.

### 4. Build char + byte-fallback tokenizer in `tokenizer.py` ✅ DONE — win, bpb 2.3526 -> 2.288 (Run 6, batch=8)
`CharByteTokenizer`: 657-entry char vocab (`char_vocab.json`, built from the
corpus, resolved via `__file__`) + 256 byte-fallback ids, vocab_size=913.
Losslessness verified: full train corpus, dev_eval, a 2000-char Devanagari
slice, and an out-of-vocab stress string (emoji/Cyrillic/Chinese/accented
Latin) — all exact round-trips. Params 1,550,080 (untied), 78% of cap.

### 4a. Vocab cost is cheap here ✅ CONFIRMED — good news for Phase 2b sequencing
At vocab≈913, embedding cost is far less than the 512–2048 BPE vocab
candidates previously sized (640K–1.28M). Confirmed by Run 6: 1,550,080
params (untied, 78% of cap), leaving ~450K headroom for width — avoids most
of the "Phase 2 and 3 must be co-designed" tension a larger BPE vocab would
have created.

### 4b. Retest weight tying once vocab is real ✅ CLOSED — still regresses, bpb 2.288 -> 2.312 (Run 7)
Tested at vocab=913 (Run 6's char tokenizer). Freed 146,080 params but bpb
regressed the same way it did at vocab=256 (Run 2) — the input/output
embedding roles still want to diverge at this width/step budget regardless
of vocab size. Reverted to untied. Closed for both vocab sizes tried, don't
retest again.

### 4c. Rare-token risk ✅ CONFIRMED NON-ISSUE
913-symbol vocab is small enough that this was never likely to bite (unlike
a 4000+ BPE vocab, where long-tail merges would get single-digit updates).
Runs 6 and 7 both trained with clean, smooth loss curves — no sign of
under-trained rare tokens. Closed, no action needed.

---

## Phase 2b — spend the param budget

### 5. Width increase (n_embd) ✅ CLOSED — win, bpb 2.288 -> 2.2392 (Run 8)
Solved the exact param formula for the largest n_embd (divisible by
n_head=4) fitting under 2M at vocab=913, untied: n_embd=184 ->
1,994,560 params (99.7% of cap, verified exact against the real model).
Converged cleanly in 2000 steps, no under-fitting from the added capacity.

### 6. Depth increase (n_layer) — not pursued
No headroom left: n_embd=184 already uses 99.7% of the param cap. Adding a
layer would require shrinking width to compensate, and per the audit, depth
is the riskier/lower-leverage lever anyway in a step-scarce regime. Skipped.

---

## Deprioritized — low leverage, don't spend a run here

- **`get_batch` random sampling (no epoch structure):** at batch=32, 2000
  steps already sees >1 full epoch (8.19M tokens vs 7.3M corpus) — a
  shuffled-epoch sampler has little left to gain.
- **In-loop dev-loss logging:** pure observability, no effect on score.

---

## Phase 4 — lock in + deliverables

- One clean final 2000-step run → `ckpt.pt` (must retain recorded step count).
- `python evaluate.py --checkpoint ckpt.pt --text_file <file>` confirmed
  working unmodified from the submission folder.
- `RUNLOG.md` — one entry per run (already current through Run 4).
- `NOTES.md` — max 10 sentences, best config + why.
- `SUMMARY.html` — agent-generated summary of RUNLOG/NOTES/architecture +
  full param list + human-vs-AI contribution breakdown.
