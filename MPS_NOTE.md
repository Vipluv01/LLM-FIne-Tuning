# Local MPS training: not viable for LoRA on this model

Diagnosed directly rather than assumed, after a LoRA sanity run stalled for
40+ minutes on 15 training steps.

## What was measured

Instrumented every micro-step of a real LoRA fine-tune (rank 8, Qwen2.5-1.5B,
`q_proj`/`v_proj` targets) on this machine's MPS backend, timing data
loading, forward, backward, and optimizer step separately:

- **Forward pass: 0.19s** — fine, in line with expectations for a 1.5B model.
- **Backward pass: 22.6s on step 0**, and **step 1 took longer, not
  shorter** — ruling out one-time MPS kernel-compilation warmup as the
  explanation. A warmup cost pays once; this got worse.

That rules out "just slow" and points to a genuine pathology in how PyTorch's
MPS backend handles the backward graph for this adapter configuration —
plausibly memory fragmentation or an inefficient fallback path for LoRA's
specific autograd pattern. Not investigated further; MPS backward-pass
internals are outside this project's scope, and the practical answer is the
same regardless of the exact mechanism.

## What this means for the project

**Zero-shot and few-shot inference work fine locally** (no backward pass
needed) — that's how the baseline sweep in `results/sweep_local.csv` was
produced, and those numbers stand.

**Any LoRA/QLoRA training must run on Kaggle's T4 (real CUDA), not locally.**
This was always the plan for the final reported numbers; this note is here
so a future run doesn't waste another 40 minutes rediscovering the same
thing by trying local MPS training again.
