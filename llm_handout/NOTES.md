# NOTES

Best configuration: char-level tokenizer with byte fallback (vocab 913), 4 layers /
4 heads / n_embd=184 / block=128, untied embeddings, plain Adam at lr=1e-3, gradient
clipping at max_norm=1.0, batch=32, 2000 steps — dev bpb **1.8798**, 1,994,560 params
(99.7% of the 2M cap), down from the 2.3718 baseline (−20.7%).

Five changes survived isolation testing, each stacked on the recipe below it: batch
8→32 (−0.2529), learning rate 3e-4→1e-3 (−0.1065), the char+byte tokenizer (−0.0646),
width n_embd 160→184 (−0.0488), and gradient clipping (−0.0192); everything else
regressed and was reverted. The tokenizer wins because the corpus uses only 657
distinct codepoints while Devanagari costs 3 bytes/char, so dropping to 1.283
bytes/token directly cuts the number of NLL terms that bpb divides by, and width was
then sized by solving the exact parameter formula `4·(12e²+13e) + 2·913·e + 128·e + 2e`
against the cap rather than by trial and error.

The unifying explanation is that at 2000 steps this model is data-starved and
under-trained rather than capacity-limited, which predicts both the wins and the
losses: batch size and learning rate win biggest because both buy more learning per
step, while LR decay, warmup and GPT-2 scaled init all lose because each trades early
progress for a late-training benefit the run never survives long enough to collect,
and weight decay had nothing to regularise.

The most instructive result was the last one: the learning rate was inherited from the
baseline and never varied for nine runs. Phase 1's three "LR schedule" runs had all
held peak LR equal to that baseline constant, so they unintentionally tested LR
*magnitude* rather than schedule *shape*, and sorted by mean LR they form a monotonic
curve whose best point is the highest LR ever tried. Acting on that reading, one run
at lr=1e-3 delivered −0.1065 — more than the tokenizer and width changes combined —
falsifying Phase 1's conclusion that the baseline sat near its capacity limit rather
than paying a mediocre-trainer tax. It neither diverged nor turned the loss curve
over, so the optimum is bracketed only from below and is likely higher still. The most
promising untried levers are a schedule with peak above 1e-3 — the hypothesis Phase 1
never actually tested — and block_size 128→256, which the formula prices at only
n_embd 184→180.
