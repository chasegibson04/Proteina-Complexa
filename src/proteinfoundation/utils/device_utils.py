"""Backend-agnostic device selection for Proteina-Complexa.

Upstream the codebase assumes an NVIDIA CUDA GPU: it hard-asserts
``torch.cuda.is_available()``, calls ``.cuda()`` directly and defaults to
``bf16-mixed`` precision.  This module centralises device handling so the same
code runs on:

==============  ==================================================================
Backend         Notes
==============  ==================================================================
``cuda``        NVIDIA.  Unchanged behaviour, still the default when present.
``rocm``        AMD via HIP.  PyTorch's ROCm build masquerades as ``torch.cuda``,
                so tensors still use ``cuda`` device strings; only capability
                reporting differs.
``directml``    AMD/Intel/Qualcomm via DirectML (Windows and WSL2).  The only
                route to a *consumer* AMD GPU such as an RX 6800 under Windows,
                because AMD's ROCm WSL support covers RDNA3/RDNA4 only.
                **Experimental and never auto-selected** -- see below.
``mps``         Apple silicon.
``cpu``         Always available.
==============  ==================================================================

Selection order is ``cuda/rocm -> mps -> cpu`` and can be overridden with the
``COMPLEXA_DEVICE`` environment variable (``auto``, ``cuda``, ``rocm``,
``directml``, ``mps``, ``cpu``, or an explicit string like ``cuda:1``).

Why DirectML is opt-in
----------------------
It runs the whole pipeline and is 2-5x faster than a 32-thread CPU, but the
structures it produces are measurably worse. On the shipped 7V11 ligand target,
three CPU samples gave mean CA-CA 3.80 A with 0/299 backbone bonds out of
tolerance and no clashes; three DirectML samples gave 4.86-5.55 A with ~63% of
bonds out of tolerance and 3-12 clashes. Radius of gyration is correct in both,
so the global fold is right and the local geometry is not.

Every individual operation tested (~110 of them, including the full attention
module in isolation) agrees with the CPU to fp32 noise, so the cause is not yet
identified. Until it is, auto-detection prefers the CPU: a slow correct answer
beats a fast wrong one, and the failure is silent.

DirectML defects this module absorbs
------------------------------------
All were found by running the model on an RX 6800, not from documentation.
The first is silent corruption rather than a crash, which is why the fix is
applied globally instead of at the call sites.

* **``float * bool`` returns bool and discards the values.**  ``x * mask[...,
  None]`` is the masking idiom used throughout the flow matcher and feature
  factories, so this destroyed coordinates at the first mask with nothing
  raised.  :func:`_install_directml_compat` repairs multiplication itself.
* **``torch.eye(device=...)`` returns an empty tensor**, and the next operation
  on it dies with ``std::system_error``.  Use :func:`eye`.
* **``F.one_hot`` and ``Tensor.scatter_add_`` are broken.**  Use :func:`one_hot`
  and :mod:`proteinfoundation.utils.scatter_utils`.
* **Mixed-dtype ``torch.where`` segfaults**, e.g. guarding an int64 count with a
  float scalar.
* **``torch.inference_mode`` is unsafe** -- operating on a view of a batch
  tensor inside it raises "Cannot set version_counter for inference tensor".
  Use :func:`inference_context`.
* **bfloat16 aborts the process** in native code, so it cannot be caught in
  Python.  :func:`autocast_dtype` and :func:`resolve_precision` never hand bf16
  to DirectML.
* **``linalg.inv``/``eigh``/``solve`` fail.**  Route them through
  :func:`cpu_fallback`.
* **There is no ``torch.cuda`` module**, so cache and synchronise helpers are
  guarded.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

import torch
from loguru import logger

__all__ = [
    "BACKEND_CPU",
    "BACKEND_CUDA",
    "BACKEND_DIRECTML",
    "BACKEND_MPS",
    "BACKEND_ROCM",
    "autocast_dtype",
    "cpu_fallback",
    "describe_device",
    "device_arg_for_lightning",
    "empty_cache",
    "eye",
    "get_backend",
    "get_device",
    "inference_context",
    "is_gpu_available",
    "log_device_banner",
    "one_hot",
    "require_gpu",
    "resolve_precision",
    "supports_bfloat16",
    "synchronize",
    "to_device",
]

BACKEND_CUDA = "cuda"
BACKEND_ROCM = "rocm"
BACKEND_DIRECTML = "directml"
BACKEND_MPS = "mps"
BACKEND_CPU = "cpu"

_ENV_VAR = "COMPLEXA_DEVICE"

# Backends whose tensors are addressed with a "cuda" device string. PyTorch's
# ROCm build reuses the entire torch.cuda namespace, so ROCm belongs here.
_CUDA_LIKE = (BACKEND_CUDA, BACKEND_ROCM)


def _directml_module():
    """Import ``torch_directml`` if installed, else ``None``."""
    try:
        import torch_directml

        return torch_directml
    except Exception:  # pragma: no cover - import guard
        return None


def _torch_cuda_usable() -> bool:
    """True when torch reports a usable CUDA/HIP device.

    ``torch.cuda.is_available()`` can raise on broken driver installs, so this
    is defensive rather than a bare call.
    """
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:  # pragma: no cover - driver weirdness
        return False


def _is_rocm_build() -> bool:
    """True when the installed torch was built against ROCm/HIP."""
    return getattr(torch.version, "hip", None) is not None


def _mps_usable() -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover
        return False


def _resolve_backend() -> str:
    """Resolve the backend to use: one of the ``BACKEND_*`` constants.

    Honours ``COMPLEXA_DEVICE``. An explicit request for an unavailable backend
    is a hard error rather than a silent downgrade -- a benchmark that quietly
    ran on the CPU would be worse than a crash.
    """
    requested = os.environ.get(_ENV_VAR, "auto").strip().lower()

    if requested not in ("", "auto"):
        base = requested.split(":", 1)[0]
        if base in _CUDA_LIKE:
            if not _torch_cuda_usable():
                raise RuntimeError(
                    f"{_ENV_VAR}={requested!r} but torch reports no CUDA/HIP device. "
                    "Check the driver and that torch was built for your GPU."
                )
            return BACKEND_ROCM if _is_rocm_build() else BACKEND_CUDA
        if base == BACKEND_DIRECTML:
            dml = _directml_module()
            if dml is None or dml.device_count() == 0:
                raise RuntimeError(
                    f"{_ENV_VAR}={requested!r} but torch_directml is unavailable or sees no device. "
                    "Install it with `pip install torch-directml` (requires torch 2.4.1)."
                )
            logger.warning(
                "DirectML is EXPERIMENTAL and produces measurably worse structures than CPU or "
                "CUDA/ROCm. On the shipped 7V11 target, CPU samples gave mean CA-CA 3.80 A with "
                "no bad backbone bonds; DirectML gave 4.86-5.55 A with ~63% of bonds out of "
                "tolerance. The cause is not yet identified. Use it for throughput experiments, "
                "not for results you intend to trust -- and check output with "
                "script_utils/gpu_time/validate_backend_outputs.py."
            )
            return BACKEND_DIRECTML
        if base == BACKEND_MPS:
            if not _mps_usable():
                raise RuntimeError(f"{_ENV_VAR}={requested!r} but MPS is unavailable.")
            return BACKEND_MPS
        if base == BACKEND_CPU:
            return BACKEND_CPU
        raise ValueError(f"Unknown {_ENV_VAR} value {requested!r}.")

    if _torch_cuda_usable():
        return BACKEND_ROCM if _is_rocm_build() else BACKEND_CUDA

    if _mps_usable():
        return BACKEND_MPS

    # DirectML is deliberately NOT auto-selected, even when present. It runs the
    # full pipeline and is faster than the CPU, but generated structures are
    # measurably worse: on the shipped 7V11 target, three CPU samples gave mean
    # CA-CA 3.80 A with no bad bonds, while three DirectML samples gave
    # 4.86-5.55 A with ~63% of backbone bonds out of tolerance. The cause is not
    # yet identified, so picking it automatically would hand users quietly
    # degraded science. Opt in with COMPLEXA_DEVICE=directml.
    return BACKEND_CPU


@functools.lru_cache(maxsize=1)
def get_backend() -> str:
    """Return the active backend, installing any backend compatibility fixes.

    Cached wrapper around :func:`_resolve_backend`; see that function for the
    selection rules and the ``COMPLEXA_DEVICE`` override.

    Selecting DirectML also installs the ``float * bool`` promotion fix (see
    :func:`_install_directml_compat`). It is applied here, rather than at import
    or in each entry point, so that no code path can reach a DirectML tensor
    without the repair in place -- the bug it guards against corrupts data
    silently.
    """
    backend = _resolve_backend()
    if backend == BACKEND_DIRECTML:
        _install_directml_compat()
    return backend


@functools.lru_cache(maxsize=8)
def get_device(index: int = 0) -> torch.device:
    """Return the ``torch.device`` for the active backend.

    Args:
        index: GPU ordinal, ignored on CPU.

    Returns:
        A device usable directly in ``.to(...)`` calls.
    """
    backend = get_backend()
    requested = os.environ.get(_ENV_VAR, "auto").strip().lower()

    # An explicit "cuda:1" style override carries its own ordinal.
    if ":" in requested:
        base, _, idx = requested.partition(":")
        if base in _CUDA_LIKE and idx.isdigit():
            index = int(idx)

    if backend in _CUDA_LIKE:
        return torch.device(f"cuda:{index}")
    if backend == BACKEND_DIRECTML:
        return _directml_module().device(index)
    if backend == BACKEND_MPS:
        return torch.device("mps")
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """True when any GPU backend (CUDA, ROCm, DirectML, MPS) is active."""
    return get_backend() != BACKEND_CPU


def require_gpu(feature: str) -> None:
    """Raise a clear error when ``feature`` genuinely cannot run on the CPU.

    Prefer letting things run slowly on the CPU; reserve this for paths that
    would otherwise fail deep inside a third-party library with an unhelpful
    message.

    Args:
        feature: Human-readable name used in the error message.
    """
    if not is_gpu_available():
        raise RuntimeError(
            f"{feature} requires a GPU, but no GPU backend was found. "
            f"Set {_ENV_VAR}=cpu only for pipeline stages that support it, or "
            "install a GPU backend (CUDA, ROCm, or torch-directml)."
        )


def supports_bfloat16() -> bool:
    """True when bf16 is safe on the active backend.

    DirectML aborts the process on bf16 tensors, so this is a safety gate and
    not merely a performance hint.
    """
    backend = get_backend()
    if backend == BACKEND_DIRECTML:
        return False
    if backend == BACKEND_MPS:
        return False
    if backend in _CUDA_LIKE:
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:  # pragma: no cover
            return False
    # CPU bf16 works but is slow and unhelpful for numerics-sensitive sampling.
    return False


def autocast_dtype(default: torch.dtype = torch.float32) -> torch.dtype:
    """Return the dtype autocast should use, respecting backend limits.

    Mirrors the deprecated ``torch.get_autocast_gpu_dtype()`` but is correct on
    non-CUDA backends and never returns bf16 where that would abort.
    """
    if not torch.is_autocast_enabled():
        return default
    backend = get_backend()
    if backend in _CUDA_LIKE:
        dtype = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.float16
    elif backend == BACKEND_CPU:
        dtype = torch.get_autocast_dtype("cpu") if hasattr(torch, "get_autocast_dtype") else torch.float32
    else:
        dtype = torch.float16

    if dtype == torch.bfloat16 and not supports_bfloat16():
        return torch.float16 if backend != BACKEND_CPU else torch.float32
    return dtype


def resolve_precision(requested: str | int) -> str | int:
    """Map a Lightning precision setting onto something the backend can run.

    ``bf16-mixed`` is the upstream default; on DirectML it would abort the
    process, and on CPU it is pathologically slow, so both are redirected.

    Args:
        requested: A Lightning precision value, e.g. ``"bf16-mixed"`` or ``32``.

    Returns:
        A precision value safe for the active backend.
    """
    backend = get_backend()
    text = str(requested).lower()

    if "bf16" in text and not supports_bfloat16():
        replacement = "32-true" if backend in (BACKEND_CPU, BACKEND_DIRECTML) else "16-mixed"
        logger.warning(
            f"Precision {requested!r} is unsupported on backend {backend!r}; using {replacement!r}. "
            "(DirectML aborts on bfloat16; CPU bf16 is far slower than fp32.)"
        )
        return replacement

    if "16" in text and backend == BACKEND_CPU and "32" not in text:
        logger.warning(f"Precision {requested!r} is slow on CPU; using '32-true'.")
        return "32-true"

    return requested


def device_arg_for_lightning() -> tuple[str, Any]:
    """Return ``(accelerator, devices)`` for ``lightning.Trainer``.

    DirectML is not a Lightning accelerator, so the Trainer is told to use the
    CPU while the model itself is moved to the DirectML device by the caller.
    """
    backend = get_backend()
    if backend in _CUDA_LIKE:
        return "gpu", 1
    if backend == BACKEND_MPS:
        return "mps", 1
    return "cpu", 1


def empty_cache() -> None:
    """Release cached GPU memory where the backend supports it."""
    backend = get_backend()
    if backend in _CUDA_LIKE:
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass
    elif backend == BACKEND_MPS:
        try:
            torch.mps.empty_cache()
        except Exception:  # pragma: no cover
            pass
    # DirectML manages its own allocations and exposes no cache API.


def synchronize() -> None:
    """Block until queued GPU work has finished, where supported.

    Needed for honest timing: DirectML and CUDA both dispatch asynchronously.
    """
    backend = get_backend()
    if backend in _CUDA_LIKE:
        try:
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
    elif backend == BACKEND_MPS:
        try:
            torch.mps.synchronize()
        except Exception:  # pragma: no cover
            pass
    elif backend == BACKEND_DIRECTML:
        # torch_directml has no synchronize(); a device->host copy forces a flush.
        try:
            torch.empty(1, device=get_device()).cpu()
        except Exception:  # pragma: no cover
            pass


def cpu_fallback(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` on the CPU and move tensor results back to the input device.

    DirectML does not implement ``linalg.inv``, ``linalg.eigh`` or
    ``linalg.solve``; rather than scatter try/except blocks through the maths
    code, call those through this helper.

    Args:
        fn: Callable to invoke.
        *args: Positional arguments; tensors are moved to the CPU.
        **kwargs: Keyword arguments; tensors are moved to the CPU.

    Returns:
        ``fn``'s result with tensors restored to the original device.
    """
    origin: torch.device | None = None
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            origin = value.device
            break

    def _down(value: Any) -> Any:
        return value.cpu() if isinstance(value, torch.Tensor) else value

    def _up(value: Any) -> Any:
        if origin is None or not isinstance(value, torch.Tensor):
            return value
        return value.to(origin)

    result = fn(*[_down(a) for a in args], **{k: _down(v) for k, v in kwargs.items()})

    if isinstance(result, torch.Tensor):
        return _up(result)
    if isinstance(result, tuple):
        return tuple(_up(r) for r in result)
    if isinstance(result, list):
        return [_up(r) for r in result]
    return result


def to_device(obj: Any, device: torch.device | None = None) -> Any:
    """Recursively move tensors in ``obj`` to ``device`` (default: active device).

    Handles tensors, modules, dicts, lists and tuples; other values pass through.
    """
    target = device if device is not None else get_device()

    if isinstance(obj, torch.Tensor | torch.nn.Module):
        return obj.to(target)
    if isinstance(obj, dict):
        return {k: to_device(v, target) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, target) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, target) for v in obj)
    return obj


def describe_device() -> str:
    """Return a one-line human-readable description of the active device."""
    backend = get_backend()
    if backend in _CUDA_LIKE:
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            label = "ROCm/HIP" if backend == BACKEND_ROCM else "CUDA"
            return f"{label}: {name} ({total:.1f} GiB)"
        except Exception:  # pragma: no cover
            return f"{backend} (properties unavailable)"
    if backend == BACKEND_DIRECTML:
        dml = _directml_module()
        try:
            return f"DirectML: {dml.device_name(0)}"
        except Exception:  # pragma: no cover
            return "DirectML (name unavailable)"
    if backend == BACKEND_MPS:
        return "MPS: Apple silicon"
    try:
        return f"CPU: {os.cpu_count()} logical cores"
    except Exception:  # pragma: no cover
        return "CPU"


_directml_compat_installed = False


def _install_directml_compat() -> None:
    """Repair DirectML defects that corrupt results without raising.

    Two patches, both applied globally rather than at call sites because the
    failures are silent and a missed site leaves wrong numbers behind.

    **1. ``float * bool`` returns bool and discards the values.**

    This is the most dangerous DirectML defect found: multiplying a float
    tensor by a bool tensor returns a **bool** tensor and discards the values,
    where every other backend promotes to float. ::

        x = torch.randn(2, 4, 3, device=dml)
        m = torch.ones(2, 4, 1, dtype=torch.bool, device=dml)
        (x * m).dtype   # torch.bool, and x's values are gone

    ``x * mask[..., None]`` is the standard masking idiom here and appears in
    about forty places across the flow matcher and feature factories. Patching
    each call site would leave silent corruption behind the first one missed --
    and the corruption is silent precisely because bool is a valid dtype, so
    the error only surfaces later (or never, just as wrong coordinates).

    Patching multiplication itself fixes every site at once, including the
    vendored openfold and colabdesign code. Scalars are untouched, and
    ``bool * bool`` stays bool so pair masks still work.

    **2. ``torch.cdist`` returns the wrong shape**: ``[b, 1, n, 1]`` instead of
    ``[b, n, m]``, again with no error, so the result broadcasts into plausible
    nonsense wherever pairwise distances are used.

    Idempotent, and a no-op on every backend other than DirectML.
    """
    global _directml_compat_installed
    if _directml_compat_installed:
        return
    _directml_compat_installed = True

    original_mul = torch.Tensor.mul
    original_imul = torch.Tensor.mul_

    def _promote(a: Any, b: Any) -> tuple[Any, Any]:
        """Cast a lone bool operand up to the other operand's dtype."""
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.dtype == torch.bool and b.dtype != torch.bool:
                a = a.to(b.dtype)
            elif b.dtype == torch.bool and a.dtype != torch.bool:
                b = b.to(a.dtype)
        return a, b

    def patched_mul(self: Any, other: Any, *args: Any, **kwargs: Any) -> Any:
        self, other = _promote(self, other)
        return original_mul(self, other, *args, **kwargs)

    def patched_mul_(self: Any, other: Any, *args: Any, **kwargs: Any) -> Any:
        # In-place cannot change self's dtype, so only `other` is adjusted.
        if isinstance(other, torch.Tensor) and other.dtype == torch.bool and self.dtype != torch.bool:
            other = other.to(self.dtype)
        return original_imul(self, other, *args, **kwargs)

    torch.Tensor.mul = patched_mul
    torch.Tensor.__mul__ = patched_mul
    torch.Tensor.__rmul__ = patched_mul
    torch.Tensor.mul_ = patched_mul_
    torch.Tensor.__imul__ = patched_mul_

    # torch.cdist returns the wrong SHAPE on DirectML -- [b, 1, n, 1] instead of
    # [b, n, m] -- with no error, so the result broadcasts into plausible
    # nonsense. Compute it from ops that are known good on this backend.
    def patched_cdist(x1: torch.Tensor, x2: torch.Tensor, p: float = 2.0, **kwargs: Any) -> torch.Tensor:
        diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
        if p == 2.0:
            return torch.sqrt((diff * diff).sum(-1).clamp(min=0.0))
        return diff.abs().pow(p).sum(-1).pow(1.0 / p)

    torch.cdist = patched_cdist

    logger.debug("Installed DirectML compatibility fixes (float*bool promotion, cdist)")


def inference_context() -> Any:
    """Return the strictest no-grad context the active backend can survive.

    ``torch.inference_mode()`` is the right choice almost everywhere: it skips
    version-counter bookkeeping and is faster than ``torch.no_grad()``.

    DirectML cannot support it. Multiplying by a view of a batch tensor inside
    inference mode raises::

        RuntimeError: Cannot set version_counter for inference tensor

    which kills sampling in the flow matcher. The same code is fine under
    ``torch.no_grad()``, so DirectML gets that instead -- correctness over the
    small bookkeeping saving.

    Usable as either a decorator or a context manager::

        @inference_context()
        def predict_step(self, batch, batch_idx): ...
    """
    if get_backend() == BACKEND_DIRECTML:
        return torch.no_grad()
    return torch.inference_mode(mode=True)


def eye(
    n: int,
    m: int | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``torch.eye`` that is correct on every backend.

    On DirectML ``torch.eye(n, device=dml)`` returns an **empty** tensor rather
    than an identity matrix, and the next op on it aborts the process with
    ``std::system_error``. Building on the CPU and copying across is correct
    everywhere and costs nothing -- identity matrices here are 3x3.

    Args:
        n: Number of rows.
        m: Number of columns; defaults to ``n``.
        device: Target device; defaults to the active device.
        dtype: Result dtype.

    Returns:
        An identity matrix on ``device``.
    """
    target = device if device is not None else get_device()
    host = torch.eye(n, m if m is not None else n, dtype=dtype)
    return host.to(target)


def one_hot(indices: torch.Tensor, num_classes: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """``F.one_hot`` that is correct on every backend.

    ``aten::one_hot`` is broken on DirectML. Comparing against ``arange`` uses
    only ops that work there and keeps the result on-device, avoiding a
    host round-trip on the model's hot path.

    Args:
        indices: Integer tensor of class indices.
        num_classes: Width of the one-hot dimension.
        dtype: Result dtype; defaults to ``torch.long``, matching ``F.one_hot``.

    Returns:
        Tensor shaped ``(*indices.shape, num_classes)``.
    """
    if get_backend() != BACKEND_DIRECTML:
        out = torch.nn.functional.one_hot(indices.long(), num_classes)
        return out if dtype is None else out.to(dtype)

    classes = torch.arange(num_classes, device=indices.device)
    out = indices.long().unsqueeze(-1) == classes
    return out.to(dtype if dtype is not None else torch.long)


def log_device_banner() -> None:
    """Log the active backend once, at startup."""
    logger.info(f"Compute backend: {get_backend()} | {describe_device()}")
    if get_backend() == BACKEND_DIRECTML:
        logger.info(
            "DirectML notes: bfloat16 is unavailable (fp32/fp16 only) and "
            "linalg.inv/eigh/solve run on the CPU via cpu_fallback()."
        )
