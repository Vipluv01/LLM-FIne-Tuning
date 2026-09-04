# Google Colab driver: the full LoRA rank sweep and data-efficiency sweep,
# on a real T4, against the full 3,080-item test set. A genuine alternate to
# kaggle/run_sweep.py -- Colab's free-tier GPU quota is a SEPARATE pool from
# Kaggle's, so running here doesn't draw against the same 30h/week budget.
# Same sweep, same fixes, just a different upload/path-finding mechanism
# (Google Drive instead of a Kaggle Dataset).
#
# Paste into a Colab notebook cell and run, after the one-time setup in
# this folder's README.md (upload the adapt/ project folder to Drive once).

# ---- mount Drive ----
from google.colab import drive

drive.mount("/content/drive")

# ---- install ----
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "bitsandbytes", "accelerate", "torchao"], check=True)
# torchao pinned/upgraded explicitly: peft checks its version eagerly during LoRA
# injection even when torchao itself is unused, and the base image can ship an
# older one than current peft expects (the exact ImportError kaggle/run_sweep.py
# hit for real -- same fix applies here since Colab's base image has the same
# kind of version drift risk).

# ---- path ----
import os

# Same robust fix as kaggle/run_sweep.py, adapted for Drive instead of a
# Kaggle Dataset mount: "from adapt.config import X" requires a directory
# literally named "adapt" on sys.path, which an arbitrary Drive upload path
# won't be by default. Find wherever runner.py + config.py actually live
# under Drive, then symlink THAT directory to a folder literally named
# "adapt" inside /content (writable, unlike Drive when mounted read-mostly
# for some account types), and put /content on sys.path. Works regardless
# of exactly where under MyDrive the project folder was uploaded.
found_dir = None
for root, dirs, files in os.walk("/content/drive/MyDrive"):
    if "runner.py" in files and "config.py" in files:
        found_dir = root
        break
if found_dir is None:
    raise RuntimeError(
        "Could not find runner.py + config.py anywhere under /content/drive/MyDrive. "
        "Upload the adapt/ project folder to your Drive first -- see this folder's README.md."
    )

link_path = "/content/adapt"
if os.path.islink(link_path) or os.path.exists(link_path):
    os.remove(link_path) if os.path.islink(link_path) else None
os.symlink(found_dir, link_path)
sys.path.insert(0, "/content")
print(f"linked {found_dir} -> {link_path}")

import torch

assert torch.cuda.is_available(), "No CUDA GPU visible -- check Runtime > Change runtime type > T4 GPU"
print("CUDA device:", torch.cuda.get_device_name(0))

# ---- sweep ----
from adapt.config import RunConfig, SweepConfig
from adapt.runner import run_sweep

# run_sweep writes results_path exactly where it's told, but the per-item
# .npy arrays (runner.py: Path("results/per_item")) are hardcoded RELATIVE
# to the process's own CWD, not derived from results_path -- so pointing
# results_path at Drive alone is NOT enough; without also chdir-ing into
# the Drive-mounted project dir, those .npy files would land under
# /content/results/per_item instead and vanish when the runtime recycles,
# exactly the kind of silent data loss this project has already had to
# clean up once (a stale/duplicate .npy mixup earlier this session -- see
# results/per_item/_stale_local_not_kaggle/). chdir-ing here means every
# relative path this run produces resolves under Drive consistently, with
# nothing left to remember to download.
os.chdir(found_dir)
os.makedirs("results", exist_ok=True)
results_path = "results/sweep_kaggle.csv"
# Deliberately still named sweep_kaggle.csv, not sweep_colab.csv: this is
# the same schema, the same downstream readers (scripts/sweep_report.py,
# analyze.py), and the same real full-test-set numbers -- just produced by
# a different GPU provider. Two result files with different names for the
# identical thing would just be an extra branch every downstream script
# and every future session has to remember to check.

sw = SweepConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct", seeds=(0, 1, 2))

# ---- ONE rank per session, parallelized across Kaggle + Colab ----
# QLoRA was tried here as a speedup (bigger batch via 4-bit weights) and
# measured NO improvement (~5.5h, same as unquantized) -- confirms the OOM
# was activation memory, not weight memory, exactly as train.py's comment
# predicted. So this stays on the proven (if slow, ~5.3h/run) non-quantized
# config: per_device_batch_size=1, gradient_accumulation_steps=16, same as
# the real completed lora_r4_seed0/seed1 rows already in sweep_kaggle.csv.
#
# Real per-run cost means the only way to cover the remaining ranks
# (8/16/32/64) in the time available is running TWO sessions AT THE SAME
# TIME on the two separate quota pools: kaggle/run_sweep.py (same
# TARGET_RANK mechanism) on Kaggle, and this script on Colab, each
# targeting a DIFFERENT rank. Change TARGET_RANK below, run, repeat with
# the next rank once each session finishes -- 2 ranks/day instead of 1.
TARGET_RANK = 16  # change to 8, 32, or 64 for the next session -- pick
                   # whichever rank ISN'T running on Kaggle right now

rank_config = RunConfig(
    sw.model_id, "lora", seed=0,
    lora_rank=TARGET_RANK, lora_alpha=TARGET_RANK * 2, train_fraction=1.0,
    per_device_batch_size=1, gradient_accumulation_steps=16,
)
print(f"targeting rank={TARGET_RANK}, run_id={rank_config.run_id()}")
run_sweep([rank_config], results_path=results_path, dev_run=False)

print("RUN COMPLETE")
print(f"Results already on Drive at {found_dir}/results/ (sweep_kaggle.csv + per_item/*.npy) -- nothing to download from the Colab runtime itself.")
print("Sync your Drive locally (or drag the results/ folder out of Drive's web UI) to pull them into this repo's results/ folder.")
