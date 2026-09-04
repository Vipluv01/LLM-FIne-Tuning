# Running the real sweep on Kaggle

Local MPS training has a confirmed pathology (see `../MPS_NOTE.md`) — LoRA
backward passes get *slower* step over step rather than settling into a
steady rate. Every result that needs a training step (LoRA rank sweep, data-
efficiency curve) has to run on Kaggle's T4 instead. Zero/few-shot inference
work stays local; that's already done (`results/sweep_local.csv`).

## One-time setup

1. On Kaggle: **New Dataset** → upload the entire `src/adapt/` folder,
   name it `adapt-src`.
2. **New Notebook** → **Add Data** → attach the `adapt-src` dataset.
3. Settings → Accelerator → **GPU T4 x2** (one is enough; having two doesn't
   help since this project doesn't parallelize training across them, but
   costs the same quota either way — pick the single-T4 option if Kaggle
   offers it separately).
4. Settings → confirm phone verification is done (required for GPU quota).

## Every session after that

Paste `run_sweep.py`'s contents into a notebook cell (or `%run` it) and
execute. It:

- Installs the exact dependency set from `pyproject.toml`
- Puts the uploaded `adapt-src` package on `sys.path`
- Runs the full rank sweep (5 ranks × 3 seeds) and data-efficiency sweep
  (5 fractions × 3 seeds) against the **full 3,080-item test set** — the
  real, final numbers, not the 693-item local dev slice
- Writes `results/sweep_kaggle.csv` and the per-item `.npy` arrays the
  statistics module needs, exactly like the local runner does

**Download `results/sweep_kaggle.csv` and `results/per_item/*.npy` back to
this repo when done** — that's what `analyze.py` and the final report read
from. Kaggle's filesystem doesn't persist across sessions by default, so
don't skip this step.

## Budget

30 free GPU-hours/week. **The original "~15-60s/run" estimate below this
line was wrong — never validated against real hardware.** A real run
measured **~5.3 GPU-hours per config** (mostly training: `per_device_batch_size=1`
+ `gradient_accumulation_steps=16`, the OOM fix needed to fit a 1.5B model
on a T4, trades away GPU parallelism to stay under 15GB). At that rate the
full 30-run sweep costs ~150+ GPU-hours — 5x the weekly quota, and more than
a single 12h session even for the rank sweep alone.

Two things worth trying before assuming the full grid isn't feasible:
**QLoRA** (`kaggle/run_sweep.py`'s commented-out 4-bit path) frees enough
memory to raise the batch size back up, which should meaningfully cut
per-run time. And **splitting runs across Kaggle and Colab** (`colab/README.md`)
roughly doubles the real weekly budget, since Colab draws from a separate
quota pool — and it writes results straight to Google Drive rather than
session-local disk, so a session dying mid-run no longer loses progress
the way a Kaggle session does.

If a session gets interrupted, `run_sweep.py` resumes automatically
(`runner.completed_run_ids` skips finished runs) — **but only if
`results/sweep_kaggle.csv` and `results/per_item/*.npy` are still present**.
Kaggle wipes `/kaggle/working` between sessions, so download those files
every time you stop the notebook, not just at the very end, or a restart
loses everything since the last download.
