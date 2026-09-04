# Running the real sweep on Google Colab (alternate to Kaggle)

Same sweep as `kaggle/run_sweep.py` (see `../MPS_NOTE.md` for why this needs
a real GPU at all), on Colab's free T4 instead of Kaggle's. Use this when
Kaggle's 30h/week pool is tight or a session already burned hours on a
partial run — Colab's free-tier quota is a completely separate pool from
Kaggle's, not shared with it.

## One-time setup

1. Upload the entire `adapt/` project folder to your Google Drive (anywhere
   under `My Drive` — the script finds it automatically, the exact path
   doesn't matter).
2. Open a new notebook at [colab.research.google.com](https://colab.research.google.com).
3. **Runtime → Change runtime type → T4 GPU**.

## Every session after that

Paste `run_sweep.py`'s contents into a cell and run it. It:

- Mounts your Google Drive (you'll get a one-time auth prompt)
- Finds the uploaded project directory automatically, regardless of exactly
  where under Drive it landed
- Installs the exact dependency set from `pyproject.toml`
- `chdir`s into the project directory on Drive **before** running anything,
  so every result this produces — `results/sweep_kaggle.csv` and
  `results/per_item/*.npy` — is written directly to Drive, not to Colab's
  local disk. There is nothing to remember to download before the runtime
  disconnects; Drive already has it.
- Runs the full rank sweep (5 ranks × 3 seeds) and data-efficiency sweep
  (5 fractions × 3 seeds) against the full 3,080-item test set

Results land at `<wherever you uploaded adapt/ on Drive>/results/`. Sync
your Drive locally (Google Drive desktop app) or download the `results/`
folder from Drive's web UI, and drop it into this repo's `results/`.

## A real open question, not just a Kaggle quirk

A prior Kaggle session running this same sweep hit that session's ~12h cap
without finishing — far past the original "15-60s/run" estimate. The
likely cause: the OOM fix (`per_device_batch_size=1`,
`gradient_accumulation_steps=16`) trades a lot of GPU parallelism away to
stay under the T4's ~15GB. If a Colab run also runs unexpectedly long,
that's the same code path, not a platform difference — worth profiling a
single run's wall-clock time early rather than assuming it'll finish inside
a session just because Colab's GPU is nominally the same T4.

## Budget

Colab's free tier has no published hard weekly cap the way Kaggle does, but
sessions disconnect after a period of inactivity and there's a rolling
usage limit that varies by account activity — treat it the same way as
Kaggle's 30h/week: real but not unlimited, and `run_sweep`'s
`completed_run_ids` resume logic means an interrupted session never loses
progress either way.
