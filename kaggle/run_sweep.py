# Kaggle driver: the full LoRA rank sweep and data-efficiency sweep, on a
# real T4, against the full 3,080-item test set.
#
# Paste into a Kaggle notebook cell after attaching the adapt-src dataset
# (see README.md in this folder), or upload this file into that dataset
# and run it with: %run /kaggle/input/adapt-src/run_sweep.py

# ---- install ----
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "bitsandbytes", "accelerate", "torchao"], check=True)
# torchao pinned/upgraded explicitly: peft checks its version eagerly during LoRA
# injection even when torchao itself is unused, and Kaggle image ships an older
# one than current peft expects (found via a real ImportError on first run).

# ---- path ----
import os

# "from adapt.config import X" requires a DIRECTORY literally named "adapt"
# on sys.path -- Python matches package names to folder names exactly, so
# no amount of sys.path tweaking fixes it if the uploaded dataset has the
# .py files sitting directly in a folder named something else (e.g.
# "adapt-src" itself, with no "adapt" subfolder, which is what a plain
# folder upload on Kaggle often produces). The robust fix regardless of
# upload layout: find wherever runner.py + config.py actually live, then
# symlink THAT directory to a folder literally named "adapt" inside
# /kaggle/working (writable, unlike /kaggle/input), and put /kaggle/working
# on sys.path. Works whether the upload was flat or already nested properly.
found_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "runner.py" in files and "config.py" in files:
        found_dir = root
        break
if found_dir is None:
    raise RuntimeError(
        "Could not find runner.py + config.py anywhere under /kaggle/input/. "
        "Run: for r, d, f in os.walk('/kaggle/input'): print(r, f) -- to see what is actually attached."
    )

link_path = "/kaggle/working/adapt"
if os.path.islink(link_path) or os.path.exists(link_path):
    os.remove(link_path) if os.path.islink(link_path) else None
os.symlink(found_dir, link_path)
sys.path.insert(0, "/kaggle/working")
print(f"linked {found_dir} -> {link_path}")

import torch

assert torch.cuda.is_available(), "No CUDA GPU visible -- check Settings > Accelerator > GPU T4"
print("CUDA device:", torch.cuda.get_device_name(0))

# ---- sweep ----
from adapt.config import RunConfig, SweepConfig
from adapt.runner import run_sweep

os.makedirs("results", exist_ok=True)

sw = SweepConfig(model_id="Qwen/Qwen2.5-1.5B-Instruct", seeds=(0, 1, 2))

# ---- ONE rank per session, parallelized across Kaggle + Colab ----
# QLoRA was tried on Colab as a speedup (bigger batch via 4-bit weights) and
# measured NO improvement (~5.5h, same as unquantized) -- confirms the OOM
# was activation memory, not weight memory, exactly as train.py's comment
# predicted. So this stays on the proven (if slow, ~5.3h/run) non-quantized
# config: per_device_batch_size=1, gradient_accumulation_steps=16, same as
# the real completed lora_r4_seed0/seed1 rows already in sweep_kaggle.csv.
#
# Real per-run cost means the only way to cover the remaining ranks
# (8/16/32/64) in the time available is running TWO sessions AT THE SAME
# TIME on the two separate quota pools: this script on Kaggle, and
# colab/run_sweep.py (same TARGET_RANK mechanism) on Colab, each targeting
# a DIFFERENT rank. Change TARGET_RANK below, run, repeat with the next
# rank once each session finishes -- 2 ranks/day instead of 1.
TARGET_RANK = 8  # change to 16, 32, or 64 for the next session

rank_config = RunConfig(
    sw.model_id, "lora", seed=0,
    lora_rank=TARGET_RANK, lora_alpha=TARGET_RANK * 2, train_fraction=1.0,
    per_device_batch_size=1, gradient_accumulation_steps=16,
)
print(f"targeting rank={TARGET_RANK}, run_id={rank_config.run_id()}")
run_sweep([rank_config], results_path="results/sweep_kaggle.csv", dev_run=False)

print("RUN COMPLETE")
print("Download results/sweep_kaggle.csv and results/per_item/*.npy before this session ends.")
