"""Supervised fine-tuning for the classification task.

The task is framed as next-token generation of the label, which is what keeps
zero-shot, few-shot and every fine-tuned variant directly comparable -- all of
them are scored by the exact same `evaluate.py` path, generating text and
parsing a label out of it. A fine-tuned classifier head would be a faster,
cleaner way to do intent classification, and is deliberately NOT what this
does, because it would answer a different, easier question than the one this
project asks. The point is measuring how much fine-tuning helps a generative
model at a task it can already attempt zero-shot.

Loss is masked to the label tokens only. Training the model to reproduce the
prompt (which is identical, or near-identical, across every example) would let
loss go down while learning nothing about the actual task -- masking is what
keeps the loss numbers meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

from adapt.task import TaskData


class LabelMaskedDataset(TorchDataset):
    """Wraps (prompt, label) pairs, masking every token except the label's."""

    def __init__(self, task: TaskData, examples, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.task = task
        self.examples = examples  # a datasets.Dataset with 'text' and 'label'
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.examples[idx]
        prompt = self.task.prompt(row["text"])
        target = " " + self.task.target(row["label"]) + self.tok.eos_token

        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tok(target, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids  # -100 = ignored by the loss

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            # Left-pad: for a decoder-only model doing causal generation, the
            # real tokens must sit at the end of the sequence so position ids
            # and causal masking line up the same way at train and eval time.
            input_ids[i, max_len - n:] = b["input_ids"]
            labels[i, max_len - n:] = b["labels"]
            attention_mask[i, max_len - n:] = 1

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class TrainStats:
    steps: int
    final_loss: float
    peak_memory_bytes: int
    wall_time_s: float
    trainable_params: int
    total_params: int


def train_supervised(
    model,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: TorchDataset,
    *,
    device: str,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_steps: int | None = None,
    epochs: float = 1.0,
    log_every: int = 20,
) -> TrainStats:
    import math
    import time

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=Collator(tokenizer.pad_token_id),
    )

    steps_per_epoch = math.ceil(len(loader) / gradient_accumulation_steps)
    total_steps = max_steps or int(steps_per_epoch * epochs)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))

    # Gradient checkpointing: trades ~20-30% more compute time for a large
    # cut in activation memory, by not storing every decoder layer's
    # intermediate output for the backward pass -- instead recomputing them
    # during backward. This is the standard fix for a model that OOMs deep
    # inside forward/backward despite comfortably fitting in isolation (a
    # 1.5B model's weights are ~3GB in bf16; a T4's full 15GB going to
    # activations for a batch of ONE example, as happened on Kaggle, means
    # the activation graph -- not the weights -- was the actual problem).
    # use_cache=False is required alongside it: KV-caching (built for
    # inference, where you want to REUSE past activations) is incompatible
    # with checkpointing (which deliberately DISCARDS them to recompute
    # later) -- leaving it on with checkpointing silently wastes the memory
    # checkpointing was just freed to save.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # Required whenever gradient checkpointing is combined with a model
        # that has FROZEN input embeddings -- which describes every LoRA
        # run, not just the quantized (QLoRA) ones. Without this, backward
        # through a checkpointed segment has no differentiable input to
        # anchor the recomputed graph to (the base model's embeddings are
        # frozen; only the tiny adapter matrices train), and depending on
        # the exact torch/transformers versions this manifests as anything
        # from silently-zero adapter gradients to an outright NaN loss a
        # few steps in -- which is exactly what a real Kaggle run hit: loss
        # was finite through step ~20, then NaN, on a plain (non-quantized)
        # LoRA config. attach_lora() already calls this for the QLoRA path
        # via prepare_model_for_kbit_training(); this call is what was
        # missing for everything else, and is idempotent to call twice.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    step = 0
    running_loss = 0.0
    running_loss_count = 0
    micro = 0
    last_logged_avg = float("nan")

    while step < total_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / gradient_accumulation_steps
            loss.backward()
            # An example whose entire label span got truncated away by
            # LabelMaskedDataset's max_length cutoff (Qwen2.5's prompts list
            # all 77 Banking77 labels every time, so occasionally the prompt
            # alone eats the whole budget) has zero non-ignored targets --
            # cross-entropy's mean over zero elements is 0/0 = NaN in the
            # forward pass. That's harmless to training itself: verified
            # directly that CrossEntropyLoss's backward returns an exact
            # ZERO gradient (not NaN) for a fully-masked target, so it
            # can't corrupt the accumulated gradient or optimizer state.
            # It WAS corrupting the printed diagnostic though -- a single
            # NaN float in this running sum poisons the whole logged
            # average for that window, showing "loss=nan" and looking like
            # real divergence when nothing was actually wrong. Excluding
            # only the NaN contribution (not the real ones) keeps the
            # printed number an honest reflection of the examples that
            # actually produced a gradient.
            loss_value = out.loss.item()
            if not math.isnan(loss_value):
                running_loss += loss_value
                running_loss_count += 1
            micro += 1

            if micro % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % log_every == 0 or step == total_steps:
                    last_logged_avg = running_loss / running_loss_count if running_loss_count else float("nan")
                    print(f"  step {step}/{total_steps}  loss={last_logged_avg:.4f}", flush=True)
                    running_loss = 0.0
                    running_loss_count = 0
                if step >= total_steps:
                    break

    # Restore use_cache before returning. This is not optional cleanup: every
    # caller of train_supervised (runner.run_one, specifically) immediately
    # follows training with model.generate() calls to evaluate the just-
    # trained adapter, and generate() with use_cache still False recomputes
    # full attention from scratch for every new token instead of reusing
    # cached keys/values -- turning a few-minute eval pass over the test set
    # into something an order of magnitude slower, silently, with no error
    # to signal it. The function that turned caching off is the one
    # responsible for turning it back on once its own job (training) is done.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_disable()
        model.config.use_cache = True

    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    trainable_n = sum(p.numel() for p in trainable)
    total_n = sum(p.numel() for p in model.parameters())

    return TrainStats(
        steps=step,
        final_loss=last_logged_avg,
        peak_memory_bytes=peak_mem,
        wall_time_s=time.time() - t0,
        trainable_params=trainable_n,
        total_params=total_n,
    )
