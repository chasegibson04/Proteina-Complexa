"""Pure-PyTorch replacements for the ``torch_scatter`` operations used here.

``torch_scatter`` is a compiled extension published only as CUDA/CPU wheels
built against one exact torch version.  It has no ROCm wheels and no DirectML
support, which made it a hard blocker for AMD GPUs.

The codebase uses exactly one function from it -- ``scatter_mean`` -- which is
expressible with built-in ops, so this module removes the dependency entirely
rather than trying to build it per backend.

The aggregation is written with ``Tensor.index_add_`` rather than the more
obvious ``Tensor.scatter_add_``: ``scatter_add_`` aborts the process on
DirectML for every input shape tested, while ``index_add_`` is correct on CUDA,
ROCm, DirectML, MPS and CPU. ``index_add_`` also takes the 1-D index these call
sites already have, with no broadcasting.

The signatures match ``torch_scatter`` so call sites need no changes beyond the
import.
"""

from __future__ import annotations

import torch

__all__ = ["scatter_add", "scatter_mean", "scatter_sum"]


def _flat_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a 1-D int64 index addressing ``src`` along ``dim``.

    ``torch_scatter`` accepts either a 1-D index (one entry per slice) or one
    already shaped like ``src``; ``index_add_`` only takes the 1-D form.

    Args:
        index: Index tensor, 1-D or shaped like ``src``.
        src: Source tensor being aggregated.
        dim: Dimension being reduced.

    Returns:
        A contiguous 1-D int64 index of length ``src.shape[dim]``.
    """
    index = index.to(torch.int64)
    if index.dim() == 1:
        return index.contiguous()

    # Shaped like src: every entry along the other axes is identical by
    # construction, so collapse back to one id per slice.
    permuted = index.movedim(dim, 0)
    return permuted.reshape(permuted.shape[0], -1)[:, 0].contiguous()


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Sum ``src`` into buckets given by ``index`` along ``dim``.

    Args:
        src: Values to aggregate.
        index: Bucket id per slice along ``dim``.
        dim: Dimension to reduce over.
        dim_size: Size of the output along ``dim``. Inferred from ``index`` when
            omitted.

    Returns:
        Tensor matching ``src`` except along ``dim``, which has length
        ``dim_size``.
    """
    dim = dim if dim >= 0 else src.dim() + dim
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    # index_add_, not scatter_add_: the latter aborts the process on DirectML.
    return out.index_add_(dim, _flat_index(index, src, dim), src.contiguous())


# torch_scatter exposes this name as well; keep both so imports can be swapped
# one-for-one.
scatter_add = scatter_sum


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Average ``src`` within buckets given by ``index`` along ``dim``.

    Drop-in replacement for ``torch_scatter.scatter_mean``. Empty buckets yield
    zero, matching upstream behaviour (rather than NaN from a 0/0 divide).

    Args:
        src: Values to aggregate.
        index: Bucket id per slice along ``dim``.
        dim: Dimension to reduce over.
        dim_size: Size of the output along ``dim``. Inferred when omitted.

    Returns:
        Bucket means, shaped like ``src`` except along ``dim``.
    """
    dim = dim if dim >= 0 else src.dim() + dim
    totals = scatter_sum(src, index, dim=dim, dim_size=dim_size)

    ones = torch.ones_like(src)
    counts = scatter_sum(ones, index, dim=dim, dim_size=totals.shape[dim])

    return totals / counts.clamp(min=1)
