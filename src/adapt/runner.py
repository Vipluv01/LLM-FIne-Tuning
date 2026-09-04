"""Executes one RunConfig end-to-end and writes a result row to disk.

This is the only place that turns a config into a result -- every run in the
sweep, whatever its method, goes through `run_one`. That's what guarantees the
resulting CSV is actually comparable across rows: same eval set, same scoring,
same schema, regardless of whether the row came from a zero-shot baseline or a
rank-64 LoRA run three days later.

Results are appended one line at a time, not batched at the end. A sweep of
40+ runs on a free-tier GPU WILL be interrupted -- by a Kaggle session limit if
nothing else -- and losing already-completed runs to a crash on run 38 would be
a bad way to spend a GPU-hour budget that doesn't come back.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from adapt.config import RunConfig
from adapt.evaluate import EvalResult, evaluate_zero_or_few_shot
from adapt.model import attach_lora, load_base, trainable_param_count
from adapt.task import TaskData, load_task, subsample_train
from adapt.train import LabelMaskedDataset, train_supervised

RESULTS_SCHEMA = [
    "run_id", "method", "lora_rank", "train_fraction", "n_shots", "seed",
    "accuracy", "unparseable_rate", "mean_latency_ms", "p99_latency_ms",
    "trainable_params", "total_params", "peak_memory_mb", "train_wall_time_s",
    "final_train_loss", "eval_n_items", "timestamp",
]


def _eval_limit_for(dev_run: bool) -> int | None:
    # Full 3,080-item test set for real runs; a small fixed slice while
    # iterating on the pipeline itself, so a bug is caught in seconds rather
    # than after a full eval pass.
    return 60 if dev_run else None


def stratified_eval_subset(task: TaskData, n: int, *, seed: int = 0) -> TaskData:
    """A stratified slice of the TEST set for fast local iteration.

    Full-set evaluation (3,080 items) takes ~25-30 minutes per config on this
    machine's GPU, which makes a 12-config baseline sweep a 5+ hour local run.
    That is the wrong place to spend wall-clock time: the final, reported
    numbers belong on Kaggle's faster GPU, and local time is better spent
    getting the pipeline and every intermediate result right first.

    Stratified rather than a random slice for the same reason as
    subsample_train: at n=300 over 77 classes, a plain random slice risks
    losing rare classes entirely, which would silently change what's being
    measured. `min_paired_samples` in stats.py is what ultimately decides
    whether a given n can resolve the effect sizes this project cares about --
    this function only produces the slice, it doesn't justify its size.
    """
    by_label: dict[int, list[int]] = {}
    for i, label in enumerate(task.test["label"]):
        by_label.setdefault(label, []).append(i)

    import random
    rng = random.Random(seed)
    keep: list[int] = []
    per_class = max(1, n // len(by_label))
    for idxs in by_label.values():
        keep.extend(rng.sample(idxs, min(per_class, len(idxs))))
    rng.shuffle(keep)

    return TaskData(labels=task.labels, train=task.train, test=task.test.select(keep))


def run_one(cfg: RunConfig, *, task: TaskData | None = None, dev_run: bool = False) -> dict:
    task = task or load_task(seed=cfg.seed)
    limit = _eval_limit_for(dev_run)

    loaded = load_base(cfg.model_id, load_in_4bit=cfg.load_in_4bit)
    trainable_params, total_params = trainable_param_count(loaded.model)
    train_stats = None

    if cfg.method in ("zero_shot", "few_shot"):
        result: EvalResult = evaluate_zero_or_few_shot(
            loaded, task, n_shots=cfg.n_shots, seed=cfg.seed, limit=limit,
        )
    elif cfg.method in ("lora", "qlora"):
        if cfg.lora_rank is None or cfg.lora_alpha is None:
            raise ValueError(f"{cfg.method} requires lora_rank and lora_alpha")
        loaded = attach_lora(
            loaded, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules, is_quantized=cfg.load_in_4bit,
        )
        trainable_params, total_params = trainable_param_count(loaded.model)

        train_examples = subsample_train(task, cfg.train_fraction, seed=cfg.seed)
        ds = LabelMaskedDataset(task, train_examples, loaded.tokenizer)
        train_stats = train_supervised(
            loaded.model, loaded.tokenizer, ds,
            device=loaded.device,
            learning_rate=cfg.learning_rate,
            per_device_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_train_steps,
        )
        loaded.model.eval()
        result = evaluate_zero_or_few_shot(loaded, task, n_shots=0, seed=cfg.seed, limit=limit)
    else:
        raise NotImplementedError(f"method {cfg.method!r} not yet wired into run_one")

    lat_ms = result.latency_s * 1000.0
    row = {
        "run_id": cfg.run_id(),
        "method": cfg.method,
        "lora_rank": cfg.lora_rank,
        "train_fraction": cfg.train_fraction,
        "n_shots": cfg.n_shots,
        "seed": cfg.seed,
        "accuracy": float(result.correct.mean()),
        "unparseable_rate": result.unparseable_rate,
        "mean_latency_ms": float(np.mean(lat_ms)),
        "p99_latency_ms": float(np.percentile(lat_ms, 99)),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "peak_memory_mb": (train_stats.peak_memory_bytes / 1e6) if train_stats else None,
        "train_wall_time_s": train_stats.wall_time_s if train_stats else None,
        "final_train_loss": train_stats.final_loss if train_stats else None,
        "eval_n_items": int(result.correct.shape[0]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Per-item correctness is what the statistics module actually needs later
    # (paired comparisons require the raw array, not the aggregate) -- stored
    # alongside the summary row rather than folded into the CSV.
    per_item_dir = Path("results/per_item")
    per_item_dir.mkdir(parents=True, exist_ok=True)
    np.save(per_item_dir / f"{cfg.run_id()}.npy", result.correct)

    return row


def append_result(row: dict, path: str = "results/sweep.csv") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_SCHEMA)
        if is_new:
            w.writeheader()
        w.writerow(row)


def completed_run_ids(path: str = "results/sweep.csv") -> set[str]:
    """So a resumed sweep skips runs that already finished."""
    p = Path(path)
    if not p.exists():
        return set()
    with p.open() as f:
        return {row["run_id"] for row in csv.DictReader(f)}


def run_sweep(
    configs: list[RunConfig],
    *,
    results_path: str = "results/sweep.csv",
    dev_run: bool = False,
    task: TaskData | None = None,
) -> None:
    task = task or load_task()
    done = completed_run_ids(results_path)
    todo = [c for c in configs if c.run_id() not in done]
    print(f"{len(done)} runs already complete, {len(todo)} remaining", flush=True)

    for i, cfg in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {cfg.run_id()}", flush=True)
        t0 = time.time()
        row = run_one(cfg, task=task, dev_run=dev_run)
        append_result(row, results_path)
        print(f"  -> acc={row['accuracy']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

        # Each run_one() call loads a fresh model. Python's own garbage
        # collector reclaims the CPU-side object once this loop iteration
        # ends, but PyTorch's CUDA caching allocator does NOT automatically
        # return freed GPU memory to the pool other allocations can use --
        # it holds it in a per-process cache for reuse. Across a sweep that
        # varies LoRA rank (different tensor shapes each run), that cache
        # fragments, and a real run on a 15GB T4 hit a CUDA OOM on the very
        # NEXT run's model load despite each individual run needing only a
        # few GB -- explicit cleanup after every run is what prevents that
        # accumulation, not a one-time fix at the end of the sweep.
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
