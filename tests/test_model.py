"""Tests for model loading and LoRA attachment.

The is_quantized path (attach_lora's call to
peft.prepare_model_for_kbit_training) can only be exercised end-to-end with
real 4-bit weights, which requires bitsandbytes + CUDA -- not available on
this machine (see MPS_NOTE.md). This test verifies the code-path SELECTION
is correct via a mock instead: that is_quantized=True actually calls
prepare_model_for_kbit_training, and is_quantized=False (the default, used
by every existing LoRA test) does not -- catching a regression where that
call gets accidentally removed or the flag stops being threaded through,
without needing a CUDA environment to catch it.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapt.model import LoadedModel, attach_lora


def _fake_loaded() -> LoadedModel:
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    return LoadedModel(model=fake_model, tokenizer=fake_tokenizer, device="cpu", dtype=None)


def test_quantized_path_calls_prepare_for_kbit_training():
    loaded = _fake_loaded()
    with patch("peft.prepare_model_for_kbit_training") as mock_prep, \
         patch("peft.get_peft_model") as mock_get_peft:
        mock_prep.return_value = loaded.model
        mock_get_peft.return_value = loaded.model
        attach_lora(loaded, rank=8, alpha=16, target_modules=("q_proj", "v_proj"), is_quantized=True)
        mock_prep.assert_called_once()


def test_non_quantized_path_skips_kbit_prep():
    loaded = _fake_loaded()
    with patch("peft.prepare_model_for_kbit_training") as mock_prep, \
         patch("peft.get_peft_model") as mock_get_peft:
        mock_get_peft.return_value = loaded.model
        attach_lora(loaded, rank=8, alpha=16, target_modules=("q_proj", "v_proj"), is_quantized=False)
        mock_prep.assert_not_called()


def test_is_quantized_defaults_to_false():
    """Every existing call site that doesn't pass is_quantized (all of the
    plain LoRA sweep) must keep getting the non-quantized path -- this
    guards the default itself, not just the explicit-False case above."""
    loaded = _fake_loaded()
    with patch("peft.prepare_model_for_kbit_training") as mock_prep, \
         patch("peft.get_peft_model") as mock_get_peft:
        mock_get_peft.return_value = loaded.model
        attach_lora(loaded, rank=8, alpha=16, target_modules=("q_proj", "v_proj"))
        mock_prep.assert_not_called()
