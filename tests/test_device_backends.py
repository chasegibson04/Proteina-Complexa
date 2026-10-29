"""Tests for the multi-backend device layer.

These run on whatever backend is active (CUDA, ROCm, DirectML, MPS or CPU) and
are written so the whole file is meaningful on a CPU-only machine -- GPU-only
assertions are skipped rather than failed.

Run the backend-selection tests against a specific backend with, e.g.::

    COMPLEXA_DEVICE=cpu pytest tests/test_device_backends.py
    COMPLEXA_DEVICE=directml pytest tests/test_device_backends.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from proteinfoundation.utils.device_utils import (
    BACKEND_CPU,
    BACKEND_CUDA,
    BACKEND_DIRECTML,
    BACKEND_MPS,
    BACKEND_ROCM,
    autocast_dtype,
    cpu_fallback,
    describe_device,
    empty_cache,
    get_backend,
    get_device,
    is_gpu_available,
    resolve_precision,
    supports_bfloat16,
    synchronize,
    to_device,
)
from proteinfoundation.utils.device_utils import eye as device_eye
from proteinfoundation.utils.device_utils import one_hot as device_one_hot
from proteinfoundation.utils.scatter_utils import scatter_mean, scatter_sum

ALL_BACKENDS = {BACKEND_CUDA, BACKEND_ROCM, BACKEND_DIRECTML, BACKEND_MPS, BACKEND_CPU}


# ---------------------------------------------------------------- backend ---


def test_backend_is_known():
    assert get_backend() in ALL_BACKENDS


def test_device_is_usable():
    """A tensor must actually allocate on the reported device."""
    x = torch.zeros(4, 4, device=get_device())
    assert x.sum().item() == 0.0


def test_describe_device_is_informative():
    text = describe_device()
    assert isinstance(text, str) and text


def test_is_gpu_available_matches_backend():
    assert is_gpu_available() == (get_backend() != BACKEND_CPU)


def test_empty_cache_and_synchronize_never_raise():
    """These are called in cleanup paths and must be safe on every backend."""
    empty_cache()
    synchronize()


def test_explicit_cpu_override_in_subprocess():
    """COMPLEXA_DEVICE=cpu must force the CPU backend regardless of hardware."""
    env = dict(os.environ, COMPLEXA_DEVICE="cpu")
    code = (
        "from proteinfoundation.utils.device_utils import get_backend, get_device;"
        "assert get_backend() == 'cpu', get_backend();"
        "assert get_device().type == 'cpu';"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_unknown_backend_rejected_in_subprocess():
    """A typo'd backend must fail loudly, not silently fall back to the CPU."""
    env = dict(os.environ, COMPLEXA_DEVICE="definitely-not-a-backend")
    code = "from proteinfoundation.utils.device_utils import get_backend; get_backend()"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode != 0
    assert "Unknown" in out.stderr or "ValueError" in out.stderr


# ---------------------------------------------------------------- dtypes ---


def test_directml_never_reports_bf16_support():
    """bf16 aborts the DirectML process at the C++ level, so this gate must hold."""
    if get_backend() == BACKEND_DIRECTML:
        assert supports_bfloat16() is False


def test_autocast_dtype_never_bf16_when_unsupported():
    if not supports_bfloat16():
        assert autocast_dtype() != torch.bfloat16


def test_autocast_dtype_defaults_outside_autocast():
    """Outside an autocast block the caller's dtype is preserved."""
    assert autocast_dtype(default=torch.float64) == torch.float64


@pytest.mark.parametrize("requested", ["bf16-mixed", "bf16", "bf16-true"])
def test_resolve_precision_rejects_bf16_when_unsupported(requested):
    resolved = resolve_precision(requested)
    if not supports_bfloat16():
        assert "bf16" not in str(resolved)
    else:
        assert resolved == requested


def test_resolve_precision_passes_through_fp32():
    assert resolve_precision("32-true") == "32-true"


def test_bf16_is_never_materialised_on_directml():
    """The real safety property: no bf16 tensor is ever created on DirectML.

    Run in a subprocess because a violation aborts the interpreter rather than
    raising, which would otherwise take the whole test session down.
    """
    if get_backend() != BACKEND_DIRECTML:
        pytest.skip("DirectML-specific guard")
    code = (
        "import torch;"
        "from proteinfoundation.utils.device_utils import autocast_dtype, supports_bfloat16;"
        "assert supports_bfloat16() is False;"
        "assert autocast_dtype() is not torch.bfloat16;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# ------------------------------------------------------------ cpu_fallback ---


def test_cpu_fallback_returns_to_origin_device():
    # Note: device_eye, not torch.eye -- see test_eye_is_a_real_identity_matrix.
    a = device_eye(3).unsqueeze(0).repeat(2, 1, 1) * 2.0
    out = cpu_fallback(torch.linalg.inv, a)
    assert out.device == a.device
    expected = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 0.5
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-5)


def test_cpu_fallback_handles_tuple_results():
    a = device_eye(3).unsqueeze(0).repeat(2, 1, 1)
    vals, vecs = cpu_fallback(torch.linalg.eigh, a)
    assert vals.device == a.device
    assert vecs.device == a.device
    torch.testing.assert_close(vals.cpu(), torch.ones(2, 3), rtol=1e-4, atol=1e-5)


def test_cpu_fallback_passes_non_tensors_through():
    assert cpu_fallback(lambda x, k=1: x + k, 2, k=3) == 5


# ------------------------------------------------- broken-creation-op shims ---


def test_eye_is_a_real_identity_matrix():
    """Regression: torch.eye(device=dml) returns an EMPTY tensor on DirectML.

    The next op on that tensor aborts the process with std::system_error, which
    took out Kabsch alignment (align_utils) before device_utils.eye existed.
    """
    got = device_eye(3)
    assert got.shape == (3, 3), f"expected 3x3 identity, got shape {tuple(got.shape)}"
    assert got.numel() == 9
    torch.testing.assert_close(got.cpu(), torch.eye(3))


def test_eye_is_usable_in_arithmetic():
    """The real failure mode was a crash on first use, not a bad value."""
    scaled = device_eye(3) * 2.0
    assert scaled.shape == (3, 3)
    torch.testing.assert_close(scaled.cpu(), torch.eye(3) * 2.0)


def test_eye_repeat_matches_kabsch_usage():
    """Exactly the expression align_utils builds its reflection matrix from."""
    batched = device_eye(3).repeat(4, 1, 1)
    assert batched.shape == (4, 3, 3)
    torch.testing.assert_close(batched.cpu(), torch.eye(3).repeat(4, 1, 1))


def test_eye_non_square_and_dtype():
    got = device_eye(3, 4, dtype=torch.float64)
    assert got.shape == (3, 4)
    assert got.dtype == torch.float64


def test_eye_lands_on_active_device():
    assert device_eye(3).device == get_device()


def test_one_hot_matches_reference():
    """aten::one_hot is broken on DirectML; the shim must match F.one_hot."""
    idx = torch.tensor([0, 2, 1, 2], device=get_device())
    got = device_one_hot(idx, num_classes=3)
    assert got.shape == (4, 3)
    expected = torch.nn.functional.one_hot(torch.tensor([0, 2, 1, 2]), 3)
    torch.testing.assert_close(got.cpu().long(), expected)


def test_one_hot_is_multipliable():
    """feature_utils does `one_hot(...) * 1.0`, which crashed on the broken op."""
    idx = torch.tensor([0, 1], device=get_device())
    got = device_one_hot(idx, num_classes=2) * 1.0
    torch.testing.assert_close(got.cpu().float(), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_one_hot_preserves_batch_shape():
    idx = torch.zeros(2, 5, dtype=torch.long, device=get_device())
    assert device_one_hot(idx, num_classes=7).shape == (2, 5, 7)


def test_one_hot_stays_on_device():
    idx = torch.tensor([0, 1], device=get_device())
    assert device_one_hot(idx, num_classes=2).device == get_device()


# --------------------------------------------------------------- to_device ---


def test_to_device_moves_nested_structures():
    device = get_device()
    payload = {
        "a": torch.zeros(2),
        "b": [torch.zeros(2), torch.ones(2)],
        "c": ("text", 7, torch.zeros(1)),
    }
    moved = to_device(payload, device)
    assert moved["a"].device == device
    assert moved["b"][1].device == device
    assert moved["c"][0] == "text" and moved["c"][1] == 7
    assert moved["c"][2].device == device


# ----------------------------------------------------------------- scatter ---


def _reference_scatter_mean(src, index, dim_size):
    """Straightforward loop implementation to check the vectorised version."""
    out = torch.zeros(dim_size, src.shape[1])
    for bucket in range(dim_size):
        rows = src[index == bucket]
        if rows.numel():
            out[bucket] = rows.mean(dim=0)
    return out


def test_scatter_mean_matches_reference():
    torch.manual_seed(0)
    src = torch.randn(10, 4)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    got = scatter_mean(src, index, dim=0, dim_size=4)
    torch.testing.assert_close(got, _reference_scatter_mean(src, index, 4))


def test_scatter_mean_empty_buckets_are_zero_not_nan():
    """Upstream torch_scatter yields 0 for empty buckets; 0/0 would give NaN."""
    src = torch.ones(3, 2)
    index = torch.tensor([0, 0, 2])
    got = scatter_mean(src, index, dim=0, dim_size=4)
    assert not torch.isnan(got).any()
    torch.testing.assert_close(got[1], torch.zeros(2))
    torch.testing.assert_close(got[3], torch.zeros(2))


def test_scatter_sum_matches_index_add():
    torch.manual_seed(0)
    src = torch.randn(12, 5)
    index = torch.randint(0, 4, (12,))
    expected = torch.zeros(4, 5).index_add_(0, index, src)
    torch.testing.assert_close(scatter_sum(src, index, dim=0, dim_size=4), expected)


def test_scatter_mean_infers_dim_size():
    src = torch.ones(4, 2)
    index = torch.tensor([0, 1, 1, 2])
    assert scatter_mean(src, index, dim=0).shape == (3, 2)


def test_scatter_mean_runs_on_active_device():
    """scatter_add_ is the one op this shim needs; it must work on the GPU."""
    device = get_device()
    src = torch.randn(8, 3, device=device)
    index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    got = scatter_mean(src, index, dim=0, dim_size=4)
    assert got.device == src.device
    torch.testing.assert_close(
        got.cpu(),
        scatter_mean(src.cpu(), index.cpu(), dim=0, dim_size=4),
        rtol=1e-4,
        atol=1e-5,
    )


def test_scatter_mean_preserves_dtype():
    src = torch.ones(4, 2, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 1])
    assert scatter_mean(src, index, dim=0, dim_size=2).dtype == torch.float64


# -------------------------------------------------------------- numerics ---


def test_core_model_ops_run_on_active_device():
    """The ops the transformer actually needs must work on this backend."""
    device = get_device()
    b, n, heads, dim = 2, 32, 4, 64
    x = torch.randn(b, n, dim, device=device)

    normed = torch.nn.functional.layer_norm(x, (dim,))
    q = normed.view(b, n, heads, dim // heads).transpose(1, 2)
    bias = torch.randn(b, heads, n, n, device=device)
    attn = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=bias)
    out = attn.transpose(1, 2).reshape(b, n, dim)

    assert out.shape == (b, n, dim)
    assert torch.isfinite(out.cpu()).all()


def test_gpu_matches_cpu_numerically():
    """GPU results must agree with the CPU reference within fp32 tolerance."""
    if not is_gpu_available():
        pytest.skip("no GPU backend active")
    torch.manual_seed(0)
    a_cpu = torch.randn(64, 128)
    b_cpu = torch.randn(128, 64)
    device = get_device()
    got = (a_cpu.to(device) @ b_cpu.to(device)).cpu()
    torch.testing.assert_close(got, a_cpu @ b_cpu, rtol=1e-3, atol=1e-3)


def test_backward_pass_on_active_device():
    device = get_device()
    x = torch.randn(4, 8, device=device, requires_grad=True)
    layer = torch.nn.Linear(8, 8).to(device)
    layer(x).pow(2).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad.cpu()).all()


# -------------------------------------------------------------- lightning ---


def test_make_trainer_builds_for_active_backend():
    from proteinfoundation.utils.lightning_backends import make_trainer

    trainer = make_trainer(logger=False, enable_checkpointing=False)
    assert trainer is not None


def test_make_trainer_downgrades_bf16_when_unsupported():
    from proteinfoundation.utils.lightning_backends import make_trainer

    trainer = make_trainer(precision="bf16-mixed", logger=False, enable_checkpointing=False)
    if not supports_bfloat16():
        assert "bf16" not in str(trainer.precision)


def test_directml_accelerator_registers():
    from proteinfoundation.utils.lightning_backends import DirectMLAccelerator

    assert isinstance(DirectMLAccelerator.auto_device_count(), int)
    if get_backend() == BACKEND_DIRECTML:
        assert DirectMLAccelerator.is_available()
