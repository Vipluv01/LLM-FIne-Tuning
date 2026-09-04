"""Experiment configuration.

One dataclass per run, so every result on disk is traceable back to the exact
settings that produced it -- model, method, rank, data fraction, seed. This is
what makes the sweep reproducible rather than a pile of numbers nobody can
regenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Method = Literal["few_shot", "zero_shot", "lora", "qlora", "full_ft"]


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    method: Method
    seed: int = 0

    # LoRA / QLoRA only.
    lora_rank: int | None = None
    lora_alpha: int | None = None
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    load_in_4bit: bool = False

    # Data-efficiency axis.
    train_fraction: float = 1.0

    # Few-shot only.
    n_shots: int = 0

    max_train_steps: int | None = None
    learning_rate: float = 2e-4
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    def run_id(self) -> str:
        bits = [self.method]
        if self.lora_rank is not None:
            bits.append(f"r{self.lora_rank}")
        if self.train_fraction != 1.0:
            bits.append(f"frac{self.train_fraction:g}")
        if self.n_shots:
            bits.append(f"shots{self.n_shots}")
        bits.append(f"seed{self.seed}")
        return "_".join(bits)


@dataclass(frozen=True)
class SweepConfig:
    """The full grid this project runs, expanded lazily by `.runs()`."""

    model_id: str
    ranks: tuple[int, ...] = (4, 8, 16, 32, 64)
    train_fractions: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
    seeds: tuple[int, ...] = (0, 1, 2)
    n_shots_grid: tuple[int, ...] = (0, 2, 4, 8)

    def baseline_runs(self) -> list[RunConfig]:
        runs = []
        for shots in self.n_shots_grid:
            for seed in self.seeds:
                method: Method = "zero_shot" if shots == 0 else "few_shot"
                runs.append(RunConfig(self.model_id, method, seed=seed, n_shots=shots))
        return runs

    def rank_sweep_runs(self, *, train_fraction: float = 1.0) -> list[RunConfig]:
        # Seed is the OUTER loop deliberately: run_sweep() processes this
        # list in order, and Kaggle's free-tier session/quota limits mean
        # the sweep can genuinely stop partway through. Every rank at
        # seed=0 first (a real, if wider-CI, "does rank matter" answer)
        # beats finishing all 3 seeds of rank=4 while ranks 8-64 never ran
        # at all -- a real run hit exactly that risk (12h session cap,
        # paused after ~2.3 configs). Extra seeds for tighter confidence
        # intervals are the next priority once every rank has a first pass.
        runs = []
        for seed in self.seeds:
            for rank in self.ranks:
                runs.append(
                    RunConfig(
                        self.model_id, "lora", seed=seed,
                        lora_rank=rank, lora_alpha=rank * 2,
                        train_fraction=train_fraction,
                    )
                )
        return runs

    def data_efficiency_runs(self, *, rank: int) -> list[RunConfig]:
        runs = []
        for frac in self.train_fractions:
            for seed in self.seeds:
                runs.append(
                    RunConfig(
                        self.model_id, "lora", seed=seed,
                        lora_rank=rank, lora_alpha=rank * 2,
                        train_fraction=frac,
                    )
                )
        return runs
