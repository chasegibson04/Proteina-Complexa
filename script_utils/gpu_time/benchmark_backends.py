"""Benchmark the compute backends on this machine's actual hardware.

Times the pair-bias attention block -- the dominant cost in Proteina-Complexa's
denoiser -- across the backends available here, so a user can see what their GPU
is worth before launching a long design job.

Usage::

    python -m script_utils.gpu_time.benchmark_backends
    python -m script_utils.gpu_time.benchmark_backends --lengths 128 256 512
    python -m script_utils.gpu_time.benchmark_backends --backends cpu directml

Each backend runs in its own subprocess: DirectML aborts the process on some
inputs, and one backend's crash must not take the whole comparison down.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CHILD_FLAG = "--_child-backend"


def _bench_one(seq_len: int, dim: int, heads: int, layers: int, iters: int, warmup: int) -> float:
    """Time one forward pass of a stack of pair-bias attention blocks.

    Returns:
        Mean milliseconds per full forward pass.
    """
    import torch
    import torch.nn.functional as F

    from proteinfoundation.utils.device_utils import get_device, synchronize

    device = get_device()
    torch.manual_seed(0)

    qkv = [torch.nn.Linear(dim, 3 * dim).to(device) for _ in range(layers)]
    proj = [torch.nn.Linear(dim, dim).to(device) for _ in range(layers)]
    norm = [torch.nn.LayerNorm(dim).to(device) for _ in range(layers)]
    to_bias = [torch.nn.Linear(32, heads).to(device) for _ in range(layers)]

    x = torch.randn(1, seq_len, dim, device=device)
    pair = torch.randn(1, seq_len, seq_len, 32, device=device)

    def forward() -> torch.Tensor:
        h = x
        for i in range(layers):
            normed = norm[i](h)
            q, k, v = qkv[i](normed).chunk(3, dim=-1)
            q, k, v = (
                t.view(1, seq_len, heads, dim // heads).transpose(1, 2) for t in (q, k, v)
            )
            bias = to_bias[i](pair).permute(0, 3, 1, 2)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            o = o.transpose(1, 2).reshape(1, seq_len, dim)
            h = h + proj[i](o)
        return h

    with torch.no_grad():
        for _ in range(warmup):
            forward()
        synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            forward()
        synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / iters * 1000.0


def _run_child(args: argparse.Namespace) -> int:
    """Benchmark inside a subprocess pinned to one backend; emit JSON on stdout."""
    from proteinfoundation.utils.device_utils import describe_device, get_backend

    out: dict[str, object] = {
        "backend": get_backend(),
        "device": describe_device(),
        "results": {},
    }
    for seq_len in args.lengths:
        try:
            ms = _bench_one(seq_len, args.dim, args.heads, args.layers, args.iters, args.warmup)
            out["results"][str(seq_len)] = round(ms, 2)
        except Exception as exc:  # pragma: no cover - reported, not raised
            out["results"][str(seq_len)] = f"FAILED: {type(exc).__name__}: {exc}"
    print("BENCHJSON" + json.dumps(out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backends", nargs="+", default=None,
                        help="Backends to compare (default: cpu plus any GPU present).")
    parser.add_argument(CHILD_FLAG, dest="child_backend", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_backend:
        return _run_child(args)

    from proteinfoundation.utils.device_utils import get_backend

    backends = args.backends
    if backends is None:
        backends = ["cpu"]
        detected = get_backend()
        if detected != "cpu":
            backends.append(detected)

    rows: list[dict] = []
    for backend in backends:
        env = dict(os.environ, COMPLEXA_DEVICE=backend)
        cmd = [sys.executable, "-m", "script_utils.gpu_time.benchmark_backends",
               CHILD_FLAG, backend,
               "--lengths", *[str(x) for x in args.lengths],
               "--dim", str(args.dim), "--heads", str(args.heads),
               "--layers", str(args.layers), "--iters", str(args.iters),
               "--warmup", str(args.warmup)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("BENCHJSON"):
                payload = json.loads(line[len("BENCHJSON"):])
        if payload is None:
            tail = (proc.stderr or "").strip().splitlines()
            reason = tail[-1][:80] if tail else f"exit {proc.returncode}"
            print(f"  {backend}: unavailable ({reason})")
            continue
        rows.append(payload)

    if not rows:
        print("No backend produced a result.")
        return 1

    print()
    print(f"pair-bias attention: {args.layers} layers, dim={args.dim}, heads={args.heads}")
    print("=" * 78)
    header = f"{'backend':12s} {'device':34s} " + " ".join(f"{f'L={n}':>10s}" for n in args.lengths)
    print(header)
    print("-" * 78)
    for row in rows:
        cells = []
        for n in args.lengths:
            val = row["results"].get(str(n))
            cells.append(f"{val:>10.1f}" if isinstance(val, int | float) else f"{'fail':>10s}")
        print(f"{row['backend']:12s} {str(row['device'])[:34]:34s} " + " ".join(cells))
    print("-" * 78)
    print("milliseconds per forward pass (lower is better)")

    baseline = next((r for r in rows if r["backend"] == "cpu"), None)
    if baseline and len(rows) > 1:
        print()
        for row in rows:
            if row["backend"] == "cpu":
                continue
            parts = []
            for n in args.lengths:
                a = baseline["results"].get(str(n))
                b = row["results"].get(str(n))
                if isinstance(a, int | float) and isinstance(b, int | float) and b > 0:
                    parts.append(f"L={n}: {a / b:.2f}x")
            if parts:
                print(f"  {row['backend']} vs cpu -> " + ",  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
