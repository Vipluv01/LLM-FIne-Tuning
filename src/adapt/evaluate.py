"""Turning model output into per-item scores.

Every function in `adapt.stats` takes a per-item array as input -- that
requirement drives everything here. Nothing in this file reports an aggregate
number directly; it always returns the full array, because the aggregate is
computed downstream once, by the statistics module, so there is exactly one
code path that turns scores into a claim.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from adapt.model import LoadedModel
from adapt.task import TaskData, few_shot_examples, few_shot_prompt

_LABEL_RE = re.compile(r"[a-z_]+")


def normalize_label(raw: str, valid_labels: tuple[str, ...]) -> str | None:
    """Map raw model text to one of the known labels, or None if it can't be resolved.

    Models under evaluation -- especially the unfine-tuned baselines -- do not
    reliably emit a bare label. They add punctuation, repeat the prompt, or
    paraphrase ("card is not working" instead of "card_not_working"). Treating
    every non-exact match as simply wrong would conflate two very different
    failures: "picked the wrong intent" and "picked the right intent but
    formatted it differently". Only the first should count against a method.

    Resolution order: exact match, then a close-match cutoff via difflib. The
    cutoff is deliberately tight (0.85) so this never silently accepts a
    genuinely different intent -- it exists to normalize formatting, not to be
    generous with correctness.
    """
    raw = raw.strip().lower()
    if raw in valid_labels:
        return raw

    first_line = raw.splitlines()[0] if raw else ""
    tokens = _LABEL_RE.findall(first_line)
    candidate = "_".join(tokens) if tokens else first_line.replace(" ", "_")
    if candidate in valid_labels:
        return candidate

    close = difflib.get_close_matches(candidate, valid_labels, n=1, cutoff=0.85)
    return close[0] if close else None


@dataclass
class EvalResult:
    correct: np.ndarray          # bool, one per test item
    predicted: list[str | None]  # resolved label, or None if unparseable
    unparseable_rate: float
    latency_s: np.ndarray        # wall time per item, for the systems side


@torch.inference_mode()
def evaluate_zero_or_few_shot(
    loaded: LoadedModel,
    task: TaskData,
    *,
    n_shots: int = 0,
    seed: int = 0,
    max_new_tokens: int = 12,
    limit: int | None = None,
) -> EvalResult:
    examples = few_shot_examples(task, n_shots, seed=seed) if n_shots else ()
    test = task.test if limit is None else task.test.select(range(limit))

    correct = np.zeros(len(test), dtype=bool)
    predicted: list[str | None] = []
    latency = np.zeros(len(test), dtype=np.float64)

    import time
    for i, row in enumerate(tqdm(test, desc=f"eval n_shots={n_shots}")):
        prompt = (
            few_shot_prompt(task, row["text"], examples)
            if n_shots
            else task.prompt(row["text"])
        )
        inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)

        t0 = time.perf_counter()
        out = loaded.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy: required for the token-identity
                                       # determinism check elsewhere in the project
            pad_token_id=loaded.tokenizer.pad_token_id,
        )
        latency[i] = time.perf_counter() - t0

        gen = out[0][inputs["input_ids"].shape[1]:]
        text = loaded.tokenizer.decode(gen, skip_special_tokens=True)

        label = normalize_label(text, task.labels)
        predicted.append(label)
        correct[i] = (label == task.target(row["label"]))

    unparseable = sum(p is None for p in predicted) / len(predicted)
    return EvalResult(correct=correct, predicted=predicted, unparseable_rate=unparseable, latency_s=latency)
