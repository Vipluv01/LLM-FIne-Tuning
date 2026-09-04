"""adapt: when is fine-tuning actually worth it?

A controlled ablation comparing prompting, LoRA at several ranks, QLoRA, and
full fine-tuning -- evaluated on the same held-out items, with the statistical
apparatus in `adapt.stats` used to say which differences are real.
"""
