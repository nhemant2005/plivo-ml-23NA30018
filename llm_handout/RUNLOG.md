# RUNLOG

## Run 0 — baseline
- Hypothesis: n/a, establish baseline.
- Config: constant lr=3e-4, Adam, batch=8, block=128, 4L/4H/160d, no schedule/decay/clip, byte tokenizer.
- Result: train loss 5.65 -> 1.73, dev bpb **2.3718**, params 1,339,840.
- Conclusion: baseline works, matches handout description of "mediocre on purpose."

## Run 1 — LR warmup + cosine decay to 0
- Hypothesis: warmup + cosine decay reduces late-training noise and should beat constant LR.
- Change: added 100-step linear warmup then cosine decay to 0 over remaining 1900 steps (train.py `lr_at`). Everything else identical to baseline (same seed, steps, batch, model).
- Result: train loss plateaued at ~1.96-1.97 by step 2000 (vs baseline 1.73). Dev bpb **2.6976** — worse than baseline's 2.3718.
- Conclusion: **regression**. Suspected decay-to-zero throttles the back half of a 2000-step run. Next: try a floor instead of decaying to 0.

## Run 1b — same schedule, decay to 10% floor instead of 0
- Hypothesis: a floor lets the model keep learning late instead of freezing, fixing run 1.
- Change: `lr_at` decays to `0.1 * peak_lr` instead of 0. Warmup (100 steps) unchanged.
- Result: train loss ~1.93 at step 2000. Dev bpb **2.6338** — still worse than baseline.
- Conclusion: floor alone didn't fix it. Comparing step-100 loss: baseline 2.85 vs this run 3.63 — the 100-step warmup ramp is costing early steps that a 2000-step budget can't recover from. Next: isolate warmup by removing it, keep decay-to-floor.

## Run 1c — decay to 10% floor, no warmup
- Hypothesis: warmup was the culprit; decay-only (starting at full LR) should recover baseline-level performance.
- Change: `warmup=0`, decay to 10% floor over all 2000 steps starting at full 3e-4.
- Result: train loss ~1.90 at step 2000 (step-100 loss matches baseline: 2.85). Dev bpb **2.6076** — still worse than baseline 2.3718.
- Conclusion: **it's the decay itself, not warmup.** With warmup removed, early-training matches baseline exactly, but the model still ends up worse because decaying LR down over the run slows learning while the model is still improving fast at full 3e-4 constant. At a 2000-step budget this GPT hasn't reached the regime where decay's noise-reduction benefit exceeds the cost of a shrinking step size — constant LR wins outright. Reverted train.py to the constant-LR baseline (schedule code removed, not just disabled) rather than keeping dead machinery around. LR scheduling is not worth pursuing further at this step budget.

## Run 2 — weight tying
- Hypothesis: `tie_weights=True` frees ~41K params for Phase 2 and often improves perplexity directly by sharing input/output embedding representation. Everything else identical to baseline (same seed, steps, batch, LR, no schedule).
- Change: `model.py` Config `tie_weights = False` -> `True` (one line; head.weight = tok_emb.weight already wired).
- Result: params 1,339,840 -> 1,298,880 (-40,960 as expected). Train loss 1.7651 at step 2000 (vs baseline 1.7315, slightly higher). Dev bpb **2.4122** — slightly worse than baseline's 2.3718 (+0.0404).
- Conclusion: **mild regression standalone.** At n_embd=160 with only 2000 steps, forcing input and output embeddings to share one matrix cost a small amount of fit quality rather than helping — plausible that the two roles (token lookup vs. next-token prediction) want slightly different directions at this width, and/or 2000 steps isn't enough to let one shared matrix converge as well as two independent ones. Still keeping the freed 40,960 params earmarked for Phase 2 reinvestment (more width/depth), since the params-are-the-constraint argument for tying holds even though the standalone perplexity argument didn't pan out here. Verdict: don't ship tying alone; re-evaluate once combined with Phase 2 capacity increase — the freed budget may make it net-positive when spent on width.

## Run 3 — scaled init + gradient clipping (combined)
- Hypothesis: GPT-2-style init (base std=0.02, residual-branch output projections scaled by 1/sqrt(2*n_layer)) plus grad clipping (max_norm=1.0) should be neutral-to-positive — init fixes a real structural gap, clipping is near-risk-free per the audit doc.
- Change: `model.py` `_init` now uses std=0.02 base, `RESID_SCALE` marker on `attn.proj` and the MLP's second Linear scales their std by `(2*n_layer)**-0.5`. `train.py` adds `clip_grad_norm_(max_norm=1.0)` before `opt.step()`. tie_weights left False (isolated from Run 2). Same seed/steps/batch/LR as baseline.
- Result: step-100 loss 3.01 (vs baseline 2.85 — slower start). Step-2000 loss 1.8140 (vs baseline 1.7315). Dev bpb **2.4465** — worse than baseline 2.3718 (+0.0747), and worse than tying alone (2.4122).
- Conclusion: **regression, combined change was net negative.** The slower early loss (step 100) points at the smaller base std (0.02 vs original 0.05) costing early progress in a 2000-step budget — same shape of problem as Run 1's warmup: anything that slows early convergence doesn't have time to pay itself back. Need to isolate: is it the init, the clip, or both? Testing clip alone next, holding init at baseline.

## Run 3b — gradient clipping alone (init reverted to baseline std=0.05)
- Hypothesis: isolate whether Run 3's regression came from init or clip; audit doc predicted clip should be near-risk-free.
- Change: `clip_grad_norm_(max_norm=1.0)` only; `model.py` init reverted to flat std=0.05. Same seed/steps/batch/LR as baseline.
- Result: step-100 loss 2.7734 (better than baseline's 2.8522), step-2000 loss 1.7133 (better than baseline's 1.7315). Dev bpb **2.3526** — **first improvement over baseline** (2.3718 -> 2.3526, -0.0192).
- Conclusion: **win, confirmed.** Clipping wasn't just risk-free as predicted — it measurably helped, better throughout the whole curve not just at the end, meaning it's not just a late-training stabilizer but is producing cleaner gradient steps from early on. This also confirms Run 3's regression traces to the init change, not the clip. Grad clip (max_norm=1.0) is the first accepted change into the running best recipe. Testing scaled init alone next to see if it's actually bad, or just bad stacked with something else.

## Run 3c — scaled init alone (grad clip removed, isolating against pure baseline)
- Hypothesis: isolate whether init or clip caused Run 3's regression.
- Change: GPT-2-style init (std=0.02 base, residual-branch scale-down) only, no grad clip. Same seed/steps/batch/LR as baseline.
- Result: step-100 loss 3.0864 (worse than baseline's 2.8522, worse even than the combined Run 3's 3.01), step-2000 loss 1.8041. Dev bpb **2.4519** — worse than baseline (2.3718) and worse than the combined run (2.4465).
- Conclusion: **scaled init is a standalone regression, confirmed as the culprit.** At only 4 layers, the residual-stream variance-accumulation problem this technique guards against barely exists — there's little to fix — while the smaller base std (0.02 vs 0.05) genuinely slows early convergence, and 2000 steps isn't enough to make up the lost ground (same failure shape as Run 1's LR warmup and Run 3's combined test). Reverted `model.py` init to flat std=0.05, restored grad clip (the confirmed Run 3b win) as the running best recipe. GPT-2 init tricks are tuned for much deeper nets; not applicable at this depth/step budget. **Phase 1 running best: baseline + grad clip only, bpb 2.3526.**

## Run 4 — AdamW + weight_decay=0.1 (stacked on grad clip)
- Hypothesis: plain Adam has zero regularization; AdamW's decoupled weight decay should be neutral-to-positive, same low-risk profile as grad clip. Stacked on the confirmed Run 3b win (grad clip stays in) since both are optimizer-level changes.
- Change: `torch.optim.Adam` -> `torch.optim.AdamW(lr=args.lr, weight_decay=0.1)`. Grad clip (max_norm=1.0) unchanged. Same seed/steps/batch/LR as baseline.
- Result: step-2000 loss 1.7219 (vs grad-clip-only's 1.7133 — nearly identical curve throughout). Dev bpb **2.3597** — slightly worse than grad-clip-only's 2.3526 (+0.0071), though still better than the original baseline (2.3718).
- Conclusion: **small regression, reverted.** Matches the audit doc's prediction: at only 2000 steps the model hasn't trained long enough to overfit in a way decay would catch, so weight decay just pulls weights toward zero a little without a corresponding generalization benefit — a small net cost, not free. Reverted to plain Adam (weight decay is a no-op/negative at this step budget). **Phase 1 final: baseline + grad clip only remains the best recipe, bpb 2.3526.** Phase 1 conclusion: only 1 of 5 tested trainer-level changes (schedule, tying, init, decay, clip) improved bpb — most of the baseline's 2.37 is closer to "real capacity limits" than "mediocre trainer tax"; grad clip was the one genuine tax being paid.

## Run 5 — batch size 8 -> 32 (block=128 unchanged)
- Hypothesis: batch scaling is ~linear in wall-clock cost while block scaling is quadratic (attention); larger batch at fixed step count sees more distinct data (2000 steps x batch 32 = 8.19M tokens, >1 full epoch over the 7.3M-byte corpus) and reduces gradient noise per step. Stacked on the confirmed Phase 1 recipe (grad clip only, plain Adam, lr=3e-4 unchanged — isolating batch alone).
- Change: added `--block` CLI arg to `train.py` (previously only settable by editing `model.py`'s `Config`), ran with `--batch 32 --block 128`. Everything else identical to the Phase 1 best recipe.
- Result: step-2000 train loss 1.4254 (vs grad-clip-only's 1.7133 at batch=8). Dev bpb **2.0836** — a large improvement over 2.3526 (-0.2690), the biggest single-run win so far, exceeding pre-run estimates (-0.05 to -0.15).
- Conclusion: **win, confirmed, by a wide margin.** Batch increase alone (no LR retune yet) already substantially beats every Phase 1 trainer change combined. lr=3e-4 was only ever validated at batch=8. Batch=32 triples wall-clock per run (7.4min vs 1.9min) — deferring the LR retune to Phase 4's final lock-in and reverting to batch=8 for Phase 3 tokenizer development, so tokenizer iteration stays fast. Run 5's win is banked and will be re-stacked at lock-in.

## Run 6 — char-level + byte-fallback tokenizer (batch=8, isolating tokenizer alone)
- Hypothesis: corpus has only 657 unique codepoints; a char-level tokenizer with byte fallback removes most of the byte-encoding tax (28% sequence-length inflation measured, Devanagari costing 3x bytes/char) at much lower implementation risk than merge-trained BPE.
- Change: `tokenizer.py` rewritten as `CharByteTokenizer` — 657-entry char vocab (`char_vocab.json`, built from `train_corpus.txt`, resolved via `__file__`) + 256 byte-fallback ids, vocab_size=913. Ran at **batch=8** (reverted from Run 5's batch=32, deliberately isolating the tokenizer alone at the cheaper/faster batch size — see Run 5 conclusion). Grad clip, plain Adam, lr=3e-4, block=128 unchanged.
- Verified losslessness before training: full train_corpus.txt round-trip, dev_eval.txt round-trip, a 2000-char Devanagari-only slice, and an out-of-vocab stress string (emoji, Cyrillic, Chinese, accented Latin — none in the 657-char vocab) — all exact `decode(encode(text)) == text`.
- Result: params 1,550,080 (untied embeddings; +210,240 vs byte tokenizer's 1,339,840, still 78% of 2M cap). Dev bpb **2.288** vs grad-clip-only baseline's 2.3526 at the same batch=8 (-0.0646).
- Conclusion: **win, confirmed, isolated at batch=8.** Smaller than the audit's -0.15 to -0.30 estimate but real and in the predicted direction. Not yet combined with Run 5's batch=32 win — that stacking happens at Phase 4 lock-in.

## Run 7 — weight tying retest at vocab=913 (char tokenizer, batch=8)
- Hypothesis: Run 2's mild regression from tying was measured at vocab=256, where the freed ~41K params weren't worth it. At vocab=913 tying frees ~146K — worth re-testing since the params-per-vocab-row calculus is different now.
- Change: `model.py` `Config.tie_weights = True`, stacked on Run 6's char tokenizer. Batch=8, block=128, grad clip, plain Adam, lr=3e-4 unchanged — isolating tying alone.
- Result: params 1,550,080 -> 1,404,000 (-146,080 as expected). Step-2000 train loss 2.1361 (vs Run 6 untied's 2.1022, slightly higher). Dev bpb **2.312** vs Run 6's 2.288 — mild regression (+0.024), same shape as Run 2.
- Conclusion: **regression, confirmed, reverted.** Tying still costs more in fit quality than it returns in freed params, even at the larger vocab — the input/output embedding roles still want to diverge at this width/step budget regardless of vocab size. `tie_weights` reverted to `False`. Don't retest again; this axis is closed for both vocab sizes tried.

## Run 8 — width increase, n_embd 160 -> 184 (Phase 2b, batch=8, isolating width alone)
- Hypothesis: at vocab=913 untied, Run 6 used only 1,550,080 of the 2M cap (78%). Solving the exact param formula (`total(V,e,layers)=layers*(12e²+13e) + e*2V + block_size*e + 2e` for V=913, layers=4, block_size=128) for the largest n_embd divisible by n_head=4 that fits gives n_embd=184 -> 1,994,560 params (verified exact against the real model, not just the formula). More width should add capacity if 2000 steps is enough to fit it.
- Change: `model.py` `Config.n_embd = 160 -> 184`. Everything else identical to Run 6 (char tokenizer, untied, grad clip, plain Adam, lr=3e-4, batch=8, block=128) — isolating width alone.
- Result: params 1,994,560 (99.7% of cap). Step-2000 train loss 2.0735 (vs Run 6's 2.1022, slightly better). Dev bpb **2.2392** vs Run 6's 2.288 — improvement (-0.0488).
- Conclusion: **win, confirmed.** The model converged cleanly with the added capacity in 2000 steps — no sign of under-fitting from the wider layers. Current best isolated (batch=8) recipe: char tokenizer + n_embd=184 + grad clip, bpb 2.2392. Still to combine at Phase 4 lock-in: Run 5's batch=32 win and the deferred LR retune.

## Run 9 — batch size 8 -> 32, stacked on Run 8's recipe (block=128, lr=3e-4 unchanged)
- Hypothesis: Run 5 showed batch=32 is a large win on its own (byte tokenizer); re-applying it on top of Run 8's char tokenizer + n_embd=184 should stack rather than cancel, since the two levers (data coverage per step vs. model capacity) are largely independent.
- Change: `--batch 32 --block 128` on top of Run 8's config (char tokenizer, n_embd=184, untied, grad clip, plain Adam, lr=3e-4 unchanged — isolating batch alone, same as Run 5's isolation).
- Result: step-2000 train loss 1.7102 (vs Run 8's 2.0735). Dev bpb **1.9863** vs Run 8's 2.2392 — large improvement (-0.2529).
- Conclusion: **win, confirmed, stacks cleanly.** Combined recipe (char tokenizer + n_embd=184 + grad clip + batch=32) already reaches 1.9863, near the audit's "stretch" estimate (1.75-1.9) before even retuning LR. lr=3e-4 still unvalidated at batch=32 with this architecture — testing a higher LR next as the last open lever before Phase 4 lock-in.

## Run 10 — learning rate 3e-4 -> 1e-3 (batch=32, stacked on Run 9's recipe)
- Hypothesis: an audit of Runs 0-9 found lr=3e-4 was inherited from the baseline and **never varied in any run**. Phase 1's three "LR schedule" runs all held peak LR equal to the baseline's constant, so each ran at roughly *half* the baseline's average LR — they tested LR magnitude, not schedule shape. Sorted by mean LR, all four Phase 1 points are cleanly monotonic (1.50e-4 → 2.6976, 1.63e-4 → 2.6338, 1.65e-4 → 2.6076, 3.00e-4 → 2.3718): more LR, better bpb, every time, with the best result sitting at the *highest LR ever tested* — the edge of the search range, not an interior optimum. Every checkpoint including Run 9 is also still descending steeply at step 2000 (slope -0.05/250 steps, not flattening), i.e. under-trained rather than capacity-limited. Batch=32 additionally lowers gradient-noise scale, which raises the maximum stable LR — so 3e-4 is a worse mismatch at batch=32 than it was at batch=8. Predicted: a higher constant LR should be a large win.
- Change: `--lr 1e-3` (~3.3x). Everything else identical to Run 9 (char tokenizer, n_embd=184, untied, grad clip max_norm=1.0, plain Adam, batch=32, block=128, seed=1337) — isolating LR alone. 1e-3 chosen over 2e-3 as the expected-value play given only one run fit in the remaining time; `clip_grad_norm_(1.0)` already active as divergence insurance.
- Result: step-100 loss 3.0364 (worse than Run 9's early curve, as expected at higher LR), step-2000 loss 1.5502 (vs Run 9's 1.7102). Dev bpb **1.8798** vs Run 9's 1.9863 — **-0.1065**.
- Conclusion: **win, confirmed, and the second-largest single win of the entire run — larger than the tokenizer (-0.0646) and width (-0.0488) changes combined.** The Phase 1 conclusion that "most of the baseline's 2.37 is closer to real capacity limits than mediocre trainer tax" is now falsified: a single hyperparameter that was never tested was worth more than all of Phase 2b and Phase 3. No divergence at 3.3x the LR, and the late slope did flatten meaningfully (-0.0288/250 steps vs Run 9's -0.0518) — the model is closer to converged but **still not converged**, and 1e-3 did not turn the curve over, so the optimum is likely higher still. **Shipped as `ckpt.pt`.**
- Open, out of time: the LR optimum is bracketed only from below (3e-4 < 1e-3 ≤ ?). 2e-3 and a warmup+cosine schedule with peak >1e-3 — the hypothesis Phase 1 never actually tested — are the obvious next runs. Also untested: block_size 128→256 (costs only n_embd 184→180 by the param formula) and seed variance.
