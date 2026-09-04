"""Tests for the training loop's state management around gradient
checkpointing and use_cache.

Uses a minimal fake model rather than a real HF model + real backward pass:
this machine has a confirmed MPS pathology where LoRA backward passes get
SLOWER step over step rather than settling into a steady rate (see
MPS_NOTE.md) -- running a real training loop here risks repeating the exact
40-minute hang that note exists to prevent. The fake model has just enough
surface (gradient_checkpointing_enable/disable, config.use_cache,
parameters(), a forward pass cheap enough to actually backward through) to
exercise the real state-management bug this test guards, on CPU, in
milliseconds.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapt.train import Collator, train_supervised


class _FakeConfig:
    def __init__(self):
        self.use_cache = True  # HF models default to True; matches reality


class _FakeModel(nn.Module):
    """Minimal stand-in for a HF CausalLM: takes input_ids/labels/
    attention_mask, returns an object with .loss, and exposes the same
    gradient-checkpointing toggle surface train_supervised checks for."""

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)
        self.config = _FakeConfig()
        self._checkpointing_enabled = False
        self.gradient_checkpointing_enable_calls = 0
        self.gradient_checkpointing_disable_calls = 0

    def gradient_checkpointing_enable(self):
        self._checkpointing_enabled = True
        self.gradient_checkpointing_enable_calls += 1

    def gradient_checkpointing_disable(self):
        self._checkpointing_enabled = False
        self.gradient_checkpointing_disable_calls += 1

    def forward(self, input_ids, labels, attention_mask=None):
        logits = self.head(self.embed(input_ids))
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss)


class _FakeTokenizer:
    pad_token_id = 0


class _FakeDataset:
    """Yields a handful of tiny fixed examples -- enough for a few real
    optimizer steps without needing any actual text data."""

    def __init__(self, n=8, seq_len=6, vocab_size=32):
        self.examples = []
        gen = torch.Generator().manual_seed(0)
        for _ in range(n):
            ids = torch.randint(1, vocab_size, (seq_len,), generator=gen)
            labels = ids.clone()
            self.examples.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def test_use_cache_and_checkpointing_restored_after_training():
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    ds = _FakeDataset()

    assert model.config.use_cache is True  # sanity: starts enabled, like a real HF model

    stats = train_supervised(
        model, tokenizer, ds, device="cpu",
        per_device_batch_size=2, gradient_accumulation_steps=2,
        max_steps=2, log_every=1,
    )

    assert stats.steps == 2
    # The actual bug being guarded: after training returns, generation-time
    # state must be back to what it was before training touched it.
    assert model.config.use_cache is True, \
        "use_cache must be restored to True after training, or every " \
        "subsequent .generate() call loses KV-caching silently"
    assert model._checkpointing_enabled is False, \
        "gradient checkpointing must be disabled after training completes"
    assert model.gradient_checkpointing_enable_calls == 1
    assert model.gradient_checkpointing_disable_calls == 1


def test_use_cache_is_false_during_training_not_just_before_and_after():
    """Guards against a fix that merely restores the value at the end
    without actually disabling it during training -- checked via a forward
    hook that records use_cache at the moment a batch is processed."""
    model = _FakeModel()
    observed_during_training = []

    orig_forward = model.forward
    def spying_forward(*args, **kwargs):
        observed_during_training.append(model.config.use_cache)
        return orig_forward(*args, **kwargs)
    model.forward = spying_forward

    train_supervised(
        model, _FakeTokenizer(), _FakeDataset(), device="cpu",
        per_device_batch_size=2, gradient_accumulation_steps=2,
        max_steps=2, log_every=1,
    )

    assert len(observed_during_training) > 0
    assert all(v is False for v in observed_during_training), \
        "use_cache must actually be False WHILE training runs, not just restored after"


class _FakeDatasetWithOneFullyMaskedExample:
    """Reproduces a real condition seen on Kaggle: LabelMaskedDataset's
    max_length truncation can occasionally cut off an example's ENTIRE
    label span (Qwen2.5's prompts list all 77 Banking77 labels every time,
    so a long prompt alone can eat the whole budget) -- leaving a row with
    zero non-ignored targets, which cross_entropy scores as loss=NaN in the
    forward pass. One row here is exactly that case; the rest are normal."""

    def __init__(self, seq_len=6, vocab_size=32):
        gen = torch.Generator().manual_seed(0)
        self.examples = []
        for _ in range(3):
            ids = torch.randint(1, vocab_size, (seq_len,), generator=gen)
            self.examples.append({"input_ids": ids, "labels": ids.clone()})
        fully_masked_ids = torch.randint(1, vocab_size, (seq_len,), generator=gen)
        self.examples.append({
            "input_ids": fully_masked_ids,
            "labels": torch.full((seq_len,), -100, dtype=torch.long),
        })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def test_a_fully_masked_example_does_not_poison_the_logged_loss_or_the_model():
    """Regression test for a real, if cosmetic, bug seen live on Kaggle: one
    NaN-loss microbatch (from a fully-truncated-away label span) made the
    ENTIRE logging window's printed average show "loss=nan", which reads as
    real training divergence even though it isn't -- verified separately
    that CrossEntropyLoss's backward returns an exact zero gradient (not
    NaN) for a fully-masked target, so it can't actually corrupt the
    optimizer state. This test checks both halves of that claim end to end:
    the reported loss stays finite, AND the model's real parameters stay
    finite after a step that included the degenerate example.
    """
    model = _FakeModel()
    ds = _FakeDatasetWithOneFullyMaskedExample()

    stats = train_supervised(
        model, _FakeTokenizer(), ds, device="cpu",
        per_device_batch_size=1, gradient_accumulation_steps=4,
        max_steps=1, log_every=1,
    )

    assert not math.isnan(stats.final_loss), \
        "one degenerate (fully-masked-label) example must not poison the whole logged average"
    for p in model.parameters():
        assert torch.isfinite(p).all(), "a fully-masked example's zero-gradient must not corrupt real parameters"


def test_enable_input_require_grads_called_for_plain_lora_not_just_qlora():
    """Regression test for a real bug: enable_input_require_grads() was
    only called in the QLoRA (is_quantized=True) branch of attach_lora(),
    but gradient checkpointing -- which NEEDS this call -- is enabled
    unconditionally in train_supervised for every LoRA run, quantized or
    not. A real Kaggle run on a plain (non-quantized) LoRA config hit NaN
    loss around step 20 as a direct result. The fix moved the call into
    train_supervised itself, alongside gradient_checkpointing_enable(), so
    it fires for every run that enables checkpointing -- verified here via
    a call-count check, since the fake model's embeddings (unlike a real
    frozen LoRA base model's) aren't actually frozen, so this can't
    reproduce the NaN dynamics directly, only confirm the required call
    happens.
    """
    model = _FakeModel()
    model.enable_input_require_grads_calls = 0
    model.enable_input_require_grads = lambda: setattr(
        model, "enable_input_require_grads_calls", model.enable_input_require_grads_calls + 1
    )

    train_supervised(
        model, _FakeTokenizer(), _FakeDataset(), device="cpu",
        per_device_batch_size=2, gradient_accumulation_steps=2,
        max_steps=1, log_every=1,
    )

    assert model.enable_input_require_grads_calls == 1
