# adapt — when is fine-tuning actually worth it?

A controlled ablation on Qwen2.5-1.5B-Instruct and Banking77 intent
classification (77 fine-grained, frequently confusable intents — e.g.
`card_payment_not_recognised` vs `reverted_card_payment` — on short, informal
customer messages). The question: does LoRA fine-tuning beat a well-engineered
few-shot prompt on the same base model, or is prompting good enough?

Every method — zero-shot, few-shot, LoRA — is scored through the exact same
path: the model generates text, a label is parsed out of it, and it's graded
exact-match against one of the 77 known labels. That keeps every configuration
directly comparable and removes any LLM-judge ambiguity from the measurement.

## Results

### Baseline: in-context learning curve (local, 693-item stratified dev slice)

| n_shots | mean accuracy | vs. previous |
|---|---|---|
| 0 (zero-shot) | 40.7% | — |
| 2 | 48.4% | +6.9pp, **significant** (Holm-corrected) |
| 4 | 50.1% | +0.7pp, not significant |
| 8 | 51.5% | +4.5pp, **significant** |

Every 8-shot seed and the 0→8 shot gain are real, not sampling noise, at this
eval-set size (paired bootstrap, Holm-corrected across comparisons — see
`scripts/baseline_report.py`). Best single 8-shot run: **52.8%**.

### LoRA fine-tuning (Kaggle T4, full 3,080-item test set)

| config | mean accuracy | seeds |
|---|---|---|
| LoRA, rank 4 | **78.2%** (std 0.0014) | 2 |
| LoRA, rank 8 | **81.6%** | 1 |
| LoRA, rank 16 | **83.9%** | 1 |
| LoRA, rank 32 | **87.0%** | 1 |

**Rank matters here, all the way up to 32.** Rank 8 beats rank 4 by
**+3.3 points** (78.3% vs. 81.6%, aggregate two-proportion z-test, z=3.23,
p=0.0013 — the true per-item array for the original rank=4 Kaggle run was
lost to a session ending before download, so only an aggregate comparison
is possible there). Rank 16 beats rank 8 by **+2.3 points** [+1.3, +3.3],
and rank 32 beats rank 16 by a further **+3.2 points** [+2.2, +4.1] — both
Holm-corrected paired bootstraps on real per-item data, genuinely
significant, using the project's actual designed methodology. Accuracy is
still climbing at rank 32; rank 64 remains untested.

### Headline comparison: fine-tuning vs. the best prompting baseline

| | accuracy | n | 95% CI |
|---|---|---|---|
| LoRA rank 4 (pooled, 2 seeds) | 78.2% | 6,160 | [77.2%, 79.2%] |
| Best 8-shot prompt | 52.8% | 693 | [49.1%, 56.5%] |

**Δ = +25.4 points** (two-proportion z-test, z=14.76, p≈0). Fine-tuning wins
decisively here — not a close call.

*Caveat*: this comparison is a two-independent-proportions test on aggregate
counts, not the project's designed paired-bootstrap methodology (which needs
matched per-item arrays over the same eval items — see `stats.py`). The LoRA
and baseline numbers also come from different-sized eval sets (3,080 vs. 693
items). The gap is large enough that this doesn't change the conclusion, but
it's a simpler test than the rest of the project holds itself to, and it's
labeled as such rather than presented as equivalent.

The best available config, rank 32, widens this further: **+34.2 points**
over the best 8-shot prompt (87.0% vs. 52.8%, aggregate two-proportion
z-test, z=20.65, p≈0). The conclusion — fine-tuning wins decisively — holds
at every rank tested; the gap only grows as rank increases.

### What's not done yet

The full designed sweep (5 ranks × 3 seeds, plus a 5-fraction data-efficiency
curve — 30 LoRA runs total) has not completed; ranks 4 and 8 are real
(above), ranks 16/32/64 and the data-efficiency curve are not. Real cause: a
single run costs **~5.3 GPU-hours** on a Kaggle T4, not the ~15-60 seconds
originally budgeted — the OOM fix needed to fit a 1.5B model on a T4
(`per_device_batch_size=1`, `gradient_accumulation_steps=16`) trades away
GPU parallelism to stay under 15GB, so the real per-run cost is roughly
300x the original estimate. QLoRA (4-bit, frees weight memory for a bigger
batch) was tried as a speedup and **measured no improvement** (~5.5h, same
as unquantized) — the OOM was activation memory, not weight memory, so that
path is a dead end, not just untried. The remaining lever is parallelizing
across Kaggle (30 free GPU-hr/week) and Colab (a separate quota pool that
writes results straight to Google Drive rather than local session disk, so
a session dying mid-run no longer loses progress — see `colab/README.md`)
— `kaggle/run_sweep.py` and `colab/run_sweep.py` each target one
`TARGET_RANK` per session for exactly this.


## Architecture

`task.py` (loads Banking77, stratified subsampling) → `model.py`
(centralizes model/dtype/device loading and LoRA attachment, including
`prepare_model_for_kbit_training` for the QLoRA/4-bit path) → `train.py`
(the training loop — label-masked loss, gradient checkpointing, restores
`use_cache=True` after training since every caller runs `.generate()`
immediately for eval) → `evaluate.py` (robust label parsing + per-item
scoring — every downstream statistic operates on the per-item array, never
a bare aggregate) → `runner.py` (orchestrates one config end-to-end, or a
whole sweep; resumable — skips configs already present in the results CSV)
→ `analyze.py` (rank/data-efficiency tables, Holm-corrected significance,
Pareto frontier). `stats.py` is the shared statistical machinery
(bootstrapped CIs, paired comparisons, multiple-testing correction)
everything else is built on.

**Local MPS training is confirmed broken for this project** — see
`MPS_NOTE.md`. A LoRA backward pass on Apple Silicon MPS gets *slower* step
over step rather than settling into a steady rate, ruling out one-time
warmup. Zero-shot/few-shot inference (no backward pass) works fine locally
and produced `results/sweep_local.csv`. All LoRA/QLoRA training runs on a
real CUDA GPU via `kaggle/run_sweep.py` or `colab/run_sweep.py`.

## Running it

```bash
.venv/bin/python -m pytest tests/ -q                # unit tests
.venv/bin/python scripts/baseline_report.py          # baseline report (local)
.venv/bin/python scripts/sweep_report.py             # LoRA sweep report (after downloading results)
```

Training requires a real GPU (see "Local MPS training" above) — follow
`kaggle/README.md` or `colab/README.md` to run the sweep and pull results
back into `results/`.
