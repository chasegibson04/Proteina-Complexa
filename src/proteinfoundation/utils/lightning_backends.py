"""Lightning glue for the non-CUDA backends.

Lightning ships accelerators for CUDA, MPS, XPU and CPU but not DirectML, so a
``Trainer(accelerator="gpu")`` cannot reach an AMD RX 6800 under Windows/WSL2.
This module registers a minimal :class:`DirectMLAccelerator` and exposes
:func:`make_trainer`, which builds a correctly-configured ``Trainer`` for
whichever backend :mod:`proteinfoundation.utils.device_utils` selected.

Only single-device inference is supported on DirectML -- there is no collective
backend behind it, so distributed training stays CUDA/ROCm-only.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.strategies import SingleDeviceStrategy
from loguru import logger

from proteinfoundation.utils.device_utils import (
    BACKEND_DIRECTML,
    get_backend,
    get_device,
    resolve_precision,
)

__all__ = ["DirectMLAccelerator", "make_trainer"]


class DirectMLAccelerator(Accelerator):
    """Lightning accelerator backed by ``torch_directml``.

    DirectML exposes devices through the ``privateuseone`` dispatch key. The
    accelerator contract is small, so this wires up device setup and leaves
    telemetry empty -- DirectML publishes no memory counters.
    """

    @staticmethod
    def name() -> str:
        """Accelerator name used by Lightning's registry."""
        return "directml"

    def setup_device(self, device: torch.device) -> None:
        """No-op: DirectML devices need no explicit activation."""

    def teardown(self) -> None:
        """No-op: DirectML frees its allocations with the process."""

    def get_device_stats(self, device: torch.device | str) -> dict[str, Any]:
        """Return an empty dict -- DirectML exposes no usable device counters."""
        return {}

    @staticmethod
    def parse_devices(devices: Any) -> int:
        """Normalise the ``devices`` Trainer argument to a count (always 1)."""
        if isinstance(devices, list):
            return len(devices)
        if isinstance(devices, str):
            return 1 if devices == "auto" else int(devices)
        return int(devices) if devices is not None else 1

    @staticmethod
    def get_parallel_devices(devices: Any) -> list[torch.device]:
        """Return the single DirectML device; multi-device is unsupported."""
        return [get_device()]

    @staticmethod
    def auto_device_count() -> int:
        """Return the number of DirectML devices present."""
        try:
            import torch_directml

            return torch_directml.device_count()
        except Exception:  # pragma: no cover - import guard
            return 0

    @staticmethod
    def is_available() -> bool:
        """True when at least one DirectML device exists."""
        return DirectMLAccelerator.auto_device_count() > 0

    @classmethod
    def register_accelerators(cls, accelerator_registry: Any) -> None:
        """Register under the name ``directml``."""
        accelerator_registry.register(
            "directml",
            cls,
            description="DirectML (AMD/Intel/Qualcomm GPUs on Windows and WSL2)",
        )


def make_trainer(precision: str | int | None = None, **kwargs: Any) -> L.Trainer:
    """Build a ``Trainer`` wired to the active backend.

    Replaces hard-coded ``Trainer(accelerator="gpu", devices=1, ...)`` calls so
    the same code path works on CUDA, ROCm, DirectML, MPS and CPU.

    Args:
        precision: Lightning precision. Passed through
            :func:`~proteinfoundation.utils.device_utils.resolve_precision`, which
            downgrades settings the backend cannot run (notably bf16 on
            DirectML, where it would abort the process).
        **kwargs: Forwarded to ``lightning.Trainer``. Any ``accelerator``,
            ``devices`` or ``strategy`` given here wins over the defaults.

    Returns:
        A configured ``lightning.Trainer``.
    """
    backend = get_backend()

    if precision is not None:
        kwargs["precision"] = resolve_precision(precision)

    if backend == BACKEND_DIRECTML:
        device = get_device()
        # The strategy carries the accelerator; passing `accelerator` as well
        # makes Lightning raise "accelerator set through both strategy class
        # and accelerator flag".
        kwargs.setdefault("devices", 1)
        kwargs.setdefault(
            "strategy",
            SingleDeviceStrategy(device=device, accelerator=DirectMLAccelerator()),
        )
        logger.debug(f"Trainer on DirectML device {device}")
    elif backend == "mps":
        kwargs.setdefault("accelerator", "mps")
        kwargs.setdefault("devices", 1)
    elif backend in ("cuda", "rocm"):
        kwargs.setdefault("accelerator", "gpu")
        kwargs.setdefault("devices", 1)
    else:
        kwargs.setdefault("accelerator", "cpu")
        kwargs.setdefault("devices", 1)

    return L.Trainer(**kwargs)
