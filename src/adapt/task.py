"""The task: Banking77 intent classification.

Chosen deliberately over a generation task. Scoring is exact-match against one
of 77 known labels, which means every one of the ~40+ runs in the sweep can be
graded in milliseconds with zero ambiguity -- no LLM-judge, no ROUGE, no
argument about what counts as "close enough". That matters because the project
runs the same eval set through dozens of configurations; anything judge-based
would burn the GPU budget on grading rather than training.

It is also genuinely hard: 77 fine-grained, frequently confusable intents
("card_payment_not_recognised" vs "reverted_card_payment") on short, informal
customer messages. A model that just pattern-matches keywords will not do well,
which is what makes "does fine-tuning help" a real question here instead of a
foregone conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from datasets import Dataset, DatasetDict, load_dataset

DATASET_ID = "legacy-datasets/banking77"

PROMPT_TEMPLATE = """You are a customer support intent classifier for a bank. \
Read the customer message and reply with EXACTLY ONE intent label from the list below \
-- nothing else, no explanation.

Intent labels:
{label_list}

Customer message: {text}
Intent:"""


@dataclass(frozen=True)
class TaskData:
    labels: tuple[str, ...]
    train: Dataset
    test: Dataset

    def label_list_str(self) -> str:
        return "\n".join(self.labels)

    def prompt(self, text: str) -> str:
        return PROMPT_TEMPLATE.format(label_list=self.label_list_str(), text=text)

    def target(self, label_id: int) -> str:
        return self.labels[label_id]


def load_task(seed: int = 0) -> TaskData:
    raw: DatasetDict = load_dataset(DATASET_ID)
    labels = tuple(raw["train"].features["label"].names)
    train = raw["train"].shuffle(seed=seed)
    test = raw["test"]
    return TaskData(labels=labels, train=train, test=test)


def subsample_train(task: TaskData, fraction: float, *, seed: int) -> Dataset:
    """Take a stratified fraction of the training set.

    Stratified, not a plain random slice: a plain slice at 5% (~500 items over
    77 classes) risks dropping rare classes to zero, which would make the
    data-efficiency curve measure "did the class disappear" instead of "does
    this method need less data". Stratifying keeps every class represented in
    proportion at every point on the curve, so the curve is actually about
    sample efficiency.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return task.train

    by_label: dict[int, list[int]] = {}
    for i, label in enumerate(task.train["label"]):
        by_label.setdefault(label, []).append(i)

    import random
    rng = random.Random(seed)
    keep: list[int] = []
    for label, idxs in by_label.items():
        n = max(1, round(len(idxs) * fraction))
        keep.extend(rng.sample(idxs, n))
    rng.shuffle(keep)
    return task.train.select(keep)


def few_shot_examples(task: TaskData, n_shots: int, *, seed: int) -> Sequence[tuple[str, str]]:
    """Pick n_shots examples spread across distinct classes, not the same one."""
    if n_shots == 0:
        return ()
    import random
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, label in enumerate(task.train["label"]):
        by_label.setdefault(label, []).append(i)

    chosen_labels = rng.sample(list(by_label.keys()), min(n_shots, len(by_label)))
    out = []
    for label in chosen_labels:
        idx = rng.choice(by_label[label])
        row = task.train[idx]
        out.append((row["text"], task.target(row["label"])))
    return out


def few_shot_prompt(task: TaskData, text: str, examples: Sequence[tuple[str, str]]) -> str:
    shots = "\n\n".join(f"Customer message: {ex_text}\nIntent: {ex_label}" for ex_text, ex_label in examples)
    header = (
        "You are a customer support intent classifier for a bank. "
        "Read the customer message and reply with EXACTLY ONE intent label from the list below "
        "-- nothing else, no explanation.\n\n"
        f"Intent labels:\n{task.label_list_str()}\n\n"
        f"Examples:\n{shots}\n\n"
    )
    return header + f"Customer message: {text}\nIntent:"
