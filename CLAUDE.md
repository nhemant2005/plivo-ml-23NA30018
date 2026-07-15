# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A timed exercise (Plivo "2,000 Step LLM Speedrun", see `LLM_assignment.pdf`): improve a
deliberately-mediocre from-scratch GPT trainer under hard, graded constraints. The working
code lives in `llm_handout/`.

`docs/` contains prep notes and problem statements for a *different*, unrelated Plivo OA
(a speech/voice pipeline exercise using resemblyzer/kokoro/librosa) — it is reference material
only, not part of this LLM assignment, and none of those libraries are used here.

## Hard caps (violating any of these disqualifies the run)

- Max **2,000 optimizer steps** for the run producing the final checkpoint.
- Max **2,000,000 total parameters**, counted from the checkpoint.
- Training/tokenizer data: **only** `llm_handout/data/train_corpus.txt`. No pretrained weights.
- Pure PyTorch + numpy + stdlib only — no `transformers`, no custom/compiled kernels, no
  mamba-ssm/flash-attn. CPU only (grading runs on CPU, no GPU/cloud).
- `tokenizer.load()` must accept arbitrary UTF-8 text with a byte fallback, and must be
  **lossless**: `decode(encode(text)) == text` exactly (checked by `evaluate.py` and by
  the graders — a lossy tokenizer makes the score meaningless).
- This exact command must work unmodified from the submission folder:
  `python evaluate.py --checkpoint ckpt.pt --text_file <any_text_file>`

Everything else is fair game to change: architecture (attention/SSM/hybrid), tokenizer
(byte/char/BPE), optimizer, LR schedule, warmup, init, weight tying, gradient clipping,
batch size, sequence length, dropout, normalization, positional encoding.

## Environment: edit on Windows, run in WSL2 Ubuntu

The repo lives at `D:\Plivo Test` on Windows but the pinned Python venv (per
`docs/Plivo_Assignment_Setup_Instructions_IIT_KGP.pdf`) lives inside WSL2 Ubuntu at
`~/speedrun/env`, not on the Windows filesystem. WSL sees this repo at `/mnt/d/Plivo Test`
— same files, no syncing needed, editing from either side edits the same bytes.

Run commands from a Windows shell via `wsl.exe`, or drop into an interactive WSL shell:

```bash
wsl.exe bash -lc 'source ~/speedrun/env/bin/activate && cd "/mnt/d/Plivo Test/llm_handout/starter" && python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt'
```

`-lc` gives a login shell so `~/.bashrc` and venv activation work normally. The Bash tool in
this session can invoke `wsl.exe` directly — no separate WSL terminal window is required.

Do not create `.sh` wrapper scripts on the Windows side and run them as `./script.sh` in WSL —
Windows line endings (`\r\n`) break the shebang. Plain `python ...` invocations are unaffected.

## Commands

Run from `llm_handout/starter/` (inside WSL, per above):

```
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

- Baseline run takes ~1.5–3 min on a laptop CPU.
- `evaluate.py` prints one JSON line: `{"bpb": ..., "n_params": ..., "steps": ...}`.
- Score is **bits per byte (bpb)** — lower is better, measured per byte (not per token) so it
  stays comparable across different tokenizer choices.

## Architecture (`llm_handout/starter/`)

- **`model.py`** — small GPT (`Config` + `GPT`). Baseline: 4 layers, 4 heads, n_embd=160,
  block_size=128, dropout=0, untied embeddings, plain `normal_(std=0.05)` init for every
  Linear/Embedding. `Config`'s public attributes are round-tripped through the checkpoint
  (see below), so new fields added to `Config` are automatically saved/restored.
- **`tokenizer.py`** — must expose `load(path=None) -> tokenizer` with `.encode(str) -> list[int]`,
  `.decode(list[int]) -> str`, `.vocab_size`. Baseline is raw UTF-8 bytes (vocab 256) — note the
  corpus is mixed English + Hindi, so Devanagari text costs 3 bytes/char under this tokenizer.
  `train.py`/`evaluate.py` call `load()` with no arguments; any files a custom tokenizer needs
  must be saved under the submission folder and resolved relative to `__file__` (grading runs
  with `cwd` = the submission folder, no internet).
- **`train.py`** — baseline trainer, intentionally mediocre: plain Adam, constant LR, no
  warmup/schedule/weight decay/grad clipping. Saves `{"model", "config", "steps",
  "train_loss_curve"}` to the checkpoint — `config` is every non-callable public attribute of
  `Config`, so `evaluate.py` can rebuild an identical model from the checkpoint alone.
- **`evaluate.py`** — the official scorer; its CLI interface (`--checkpoint`, `--text_file`) must
  not change. Loads the model from the checkpoint's saved config, re-verifies the tokenizer
  round-trip, then computes bpb using a **sliding window with 50% context overlap** (stride =
  `block_size // 2`) so every token except the first is scored with real left context.

## Deliverables (graded)

Final submission folder needs: `ckpt.pt` (must retain the recorded step count), modified code
including a working `evaluate.py` + `tokenizer.py`, `RUNLOG.md` (one entry per training run:
hypothesis, what changed, dev bpb before/after, conclusion — this is graded), `NOTES.md` (max
10 sentences on the best configuration and why), and `SUMMARY.html` (agent-generated summary of
`RUNLOG.md`/`NOTES.md`/architecture with full parameter list and human-vs-AI contribution
breakdown).

## Working method (established, keep following this)

The user follows a strict **4-phase plan** (their own wording, not to be reordered without
asking):
- **Phase 1** — trainer-only fixes (schedule/warmup, gradient clipping, init scaling, weight
  tying). No architecture/tokenizer changes. **Concluded**, see below.
- **Phase 2** — spend the param budget (width/depth) on top of Phase 1's winning recipe.
- **Phase 3** — BPE tokenizer swap (highest ceiling, highest risk — must independently verify
  losslessness on the mixed English/Hindi corpus, not just ASCII).
- **Phase 4** — lock in final recipe, one clean 2000-step run, deliverables.

**Standing instruction (given verbatim, still in force): "DO NOT JUMP TO IMPLEMENTATION."**
Always present analysis (pros/cons/leverage) and wait for explicit go-ahead ("proceed", "start
implementing", a named item) before writing code — including for new phases. Change one
variable at a time in experiments unless the user explicitly says to combine.

Two working docs live in `llm_handout/` and should be kept up to date as experiments run —
read them before proposing new experiments, they contain the full reasoning trail:
- **`llm_handout/RUNLOG.md`** — the graded deliverable. One entry per training run.
- **`llm_handout/QUESTIONABLE_ITEMS.md`** — non-graded planning doc auditing all 4 phases for
  missed items, with pros/cons/leverage per item and status markers (✅ closed / 🔲 planned).

### Phase 1 results (concluded)

Baseline (Run 0): bpb **2.3718**, params 1,339,840 (67% of 2M cap).

Tested 5 trainer-only changes, isolating each on top of the confirmed baseline — **only
gradient clipping helped**:
- LR schedule (3 variants: warmup+cosine→0, warmup+cosine→floor, no-warmup+cosine→floor) —
  all regressed (bpb 2.60–2.70). Root cause: at only 2000 steps the model is still improving
  fast at constant LR; any decay throttles it before earning back the benefit. Fully reverted,
  don't retry.
- Weight tying alone — bpb 2.4122, mild regression. **Revisit under Phase 3**: at vocab=256 the
  freed params (40,960) weren't worth it, but at BPE-scale vocab (1000+) tying frees far more
  (see Phase 2/3 interaction below) — Run 2's conclusion does not transfer, retest.
- GPT-2-style scaled init (std=0.02 + residual-branch scaling) — bpb 2.4519 alone, regressed.
  Only 4 layers means little residual-variance problem to fix, while the smaller base std
  slows early convergence — same failure shape as the LR schedule. Not worth pursuing at this
  depth.
- **Gradient clipping (`max_norm=1.0`) — bpb 2.3526, the one confirmed win** (-0.0192 vs
  baseline). Currently the only active deviation from original baseline in `train.py`. Helped
  throughout the whole curve, not just late — cleaner gradient steps from early on.
- AdamW + weight_decay=0.1 (stacked on grad clip) — bpb 2.3597, small regression vs grad-clip-
  only. 2000 steps isn't long enough to overfit, so decay just costs capacity. Reverted to
  plain Adam.

**Phase 1 running-best recipe currently in code**: baseline architecture + plain Adam + grad
clip only. bpb 2.3526, params unchanged at 1,339,840 (no tying, no init change kept).

**Phase 1 conclusion**: only 1 of 5 trainer-level changes improved bpb — most of the baseline's
2.37 is closer to real capacity limits than "mediocre trainer tax."

### Phase 2 × Phase 3 interaction (analyzed, not yet implemented)

Key finding: these two phases **cannot be tuned sequentially** — vocab cost scales linearly
with width (`vocab × n_embd`, doubled if untied) while transformer-block cost scales
quadratically with width (`~12 × n_embd²` per layer, exact formula derived and verified against
Run 0's known param count). A bigger BPE vocab makes every unit of width more expensive, so
picking width before vocab risks having to redo Phase 2 once Phase 3 lands.

Exact param formula (verified exact against Run 0 = 1,339,840):
```
block(n_embd)     = 12·n_embd² + 13·n_embd            (per layer, incl. biases/LN)
total(V,e,layers) = layers·block(e) + e·(V if tied else 2·V) + block_size·e + 2·e
```

Computed candidate configs (n_layer=4, block_size=128, tied=True, solved for max n_embd under
the 2M cap, n_embd kept divisible by n_head=4):

| Candidate | vocab | n_embd | Total params | Headroom |
|---|---|---|---|---|
| A conservative | 512 | 196 | 1,979,992 | 20,008 |
| **B recommended** | **1,024** | **188** | **1,923,240** | **76,760** |
| C aggressive vocab | 2,048 | 180 | 1,956,600 | 43,400 |
| D untied @ V=1024 (comparison) | 1,024 | 180 | 1,956,600 | 43,400 |

Tying roughly doubles the affordable vocab for the same width (untied V=1,024 costs the same
width-budget as tied V=2,048) — concrete evidence to retest weight tying once vocab is large.

Other risks flagged for Phase 3, not yet hit in practice:
- **Rare-token under-training**: a large vocab (4000+) with only 16,000 training windows total
  (2000 steps × batch 8) means long-tail merge tokens get single-digit gradient updates — same
  failure shape as every other "not enough steps to pay it back" regression this run. Lean
  toward a smaller vocab (~1,000–2,000), not the largest that fits the param cap.
- **Corpus token-count shrinkage**: BPE compresses `len(ids)`, shrinking the pool of valid
  sample start positions in `get_batch` — should hold fine at 7.3MB but worth sanity-checking,
  not assuming.
- **Losslessness must be verified specifically on Devanagari spans**, not just ASCII — a merge
  that mishandles multi-byte UTF-8 boundaries silently corrupts round-trip and disqualifies the
  run per the hard caps above.
- **Exact vocab size is a training hyperparameter, not a precomputed number** — candidate B's
  n_embd=188 assumed vocab=1,024 will "take"; actual usable vocab depends on how fast BPE merge
  frequency decays on this specific mixed English/Hindi corpus. **Recommended next step (agreed
  direction, not yet started): pilot-train a BPE tokenizer offline first** (no cost to the
  2000-step budget or the ~6-10 run budget — it's not a training run), verify lossless
  round-trip on both scripts, inspect real compression ratio and merge-frequency tail, *then*
  solve the width equation with the real vocab number — before touching `model.py`/`train.py`.

### Pending / not yet started

- Phase 3 BPE pilot (train tokenizer, verify round-trip, measure compression) — agreed as next
  step, awaiting go-ahead to actually implement.
- Phase 2 width/depth reinvestment — blocked on Phase 3 pilot results per the interaction above.
- Weight tying retest at BPE-scale vocab.
- Phase 4: final lock-in run, `NOTES.md`, `SUMMARY.html`.
