# 2,000-Step LLM Speedrun

**Hemant Nallamasa · 23NA30018**

A GPT-style LLM trained from scratch on mixed English + Hindi text, under hard caps of
2,000 optimizer steps and 2,000,000 parameters, CPU only. Plivo ML assignment.

**Result: dev bpb 1.8798** (baseline 2.3718, −20.7%) · 1,994,560 params (99.7% of cap) · 2,000 steps.

## Score it

```bash
cd llm_handout/starter
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

Prints one JSON line: `{"bpb": ..., "n_params": ..., "steps": ...}`. Lower is better.
Works with any UTF-8 text file, runs on CPU, needs no internet.

## Reproduce it

```bash
cd llm_handout/starter
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
```

Defaults are the shipped recipe (batch=32, lr=1e-3, seed=1337), so this reproduces
`ckpt.pt`. ~11 min on a laptop CPU.

## Final configuration

| | |
|---|---|
| Tokenizer | char-level, 657 corpus chars + 256 byte-fallback ids (vocab 913) |
| Architecture | 4 layers, 4 heads, n_embd=184, block_size=128, untied embeddings |
| Optimizer | plain Adam, lr=1e-3 constant, grad clip max_norm=1.0 |
| Batch / steps | 32 / 2,000 |

## Layout

```
llm_handout/
  starter/          the submission — code, ckpt.pt, and deliverables
    ckpt.pt         final checkpoint (records step count)
    model.py        the GPT
    tokenizer.py    char + byte-fallback tokenizer (needs char_vocab.json)
    char_vocab.json 657 chars, built from train_corpus.txt only
    train.py        trainer
    evaluate.py     official scorer — unmodified
    ckpt_run*.pt    experiment checkpoints, kept for reference
  data/             train_corpus.txt (the only training data), dev_eval.txt
  RUNLOG.md         every run: hypothesis, change, bpb before/after, conclusion
  NOTES.md          best config and why, in 9 sentences
  SUMMARY.html      full summary — open in a browser
  QUESTIONABLE_ITEMS.md   planning doc auditing every lever considered
```

Deliverables are mirrored into `llm_handout/starter/` so the submission folder is
self-contained.

## The short version

Eleven runs, one variable at a time. Five changes helped: batch 8→32 (−0.2529),
lr 3e-4→1e-3 (−0.1065), the char tokenizer (−0.0646), width 160→184 (−0.0488), and
gradient clipping (−0.0192). Six regressed and were reverted — LR schedules, warmup,
GPT-2 scaled init, weight decay, and weight tying (twice).

They all fail for one reason: at 2,000 steps this model is **under-trained, not
capacity-limited**, so anything trading early progress for late-training polish never
gets paid back. The same fact is why the biggest wins were batch size and learning
rate — both simply buy more learning per step.

The learning rate turned out to be the interesting one. It was inherited from the
baseline and never varied for nine runs, and the early "LR schedules don't work"
conclusion came from three runs that had accidentally held peak LR equal to the
baseline constant — testing LR *magnitude*, not schedule *shape*. Sorted by mean LR,
those runs form a clean monotonic curve whose best point was the highest LR ever
tried. One run at lr=1e-3 then delivered −0.1065, beating the tokenizer and width
changes combined. It still hasn't turned the loss curve over, so the optimum is
bracketed only from below — see `RUNLOG.md` Run 10 and the open levers in `NOTES.md`.

See [`SUMMARY.html`](llm_handout/SUMMARY.html) for the full run history, parameter
list, and human/AI contribution breakdown.
