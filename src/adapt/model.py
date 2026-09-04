"""Model loading, one place, for every method in the sweep.

Centralised for a concrete reason: zero-shot, few-shot, LoRA, QLoRA and full
fine-tuning must all load the SAME base checkpoint the SAME way, or a
difference in outcomes could be an artifact of e.g. one path loading in fp16
and another in fp32 rather than the method itself. This module is the one
place that decides dtype and device, so every run shares it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def best_dtype(device: str) -> torch.dtype:
    # MPS's bf16 support is inconsistent across ops as of this torch release;
    # fp16 is the reliable choice there. CUDA gets bf16, which has the wider
    # dynamic range and is what the quantized (4-bit) paths expect downstream.
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    dtype: torch.dtype


def load_base(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    load_in_4bit: bool = False,
    device: str | None = None,
) -> LoadedModel:
    device = device or best_device()
    dtype = best_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # Decoder-only models routinely ship without a pad token; reusing EOS
        # is the standard fix and is safe because we always mask padding out
        # of the loss.
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"dtype": dtype}
    if load_in_4bit:
        if device != "cuda":
            raise ValueError("4-bit quantization requires CUDA (bitsandbytes); "
                              f"got device={device!r}. Run this path on Kaggle's T4.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()

    return LoadedModel(model=model, tokenizer=tokenizer, device=device, dtype=dtype)


def attach_lora(
    loaded: LoadedModel, *, rank: int, alpha: int, target_modules: tuple[str, ...],
    is_quantized: bool = False,
) -> LoadedModel:
    from peft import LoraConfig, get_peft_model

    if is_quantized:
        # Required before wrapping a 4-bit (QLoRA) model with LoRA, not
        # optional. prepare_model_for_kbit_training casts layer norms and
        # the output logits layer to fp32 for numerical stability against
        # quantized weights, and -- the part that actually bites without
        # it -- calls enable_input_require_grads() so the backward graph
        # connects from the (frozen, quantized) input embeddings through to
        # the LoRA adapter weights at all. Skipping this on a quantized
        # model is a well-known way to hit "element 0 of tensors does not
        # require grad" the moment training starts, especially combined
        # with gradient checkpointing (train.py enables it unconditionally),
        # which depends on the same input-requires-grad wiring to recompute
        # activations correctly during backward.
        from peft import prepare_model_for_kbit_training
        loaded.model = prepare_model_for_kbit_training(loaded.model)

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    loaded.model = get_peft_model(loaded.model, cfg)
    return loaded


def trainable_param_count(model: PreTrainedModel) -> tuple[int, int]:
    """Returns (trainable, total). The ratio is the headline number for the
    rank sweep -- it's what "1/8th the parameters" in the results actually
    means, computed rather than asserted."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
