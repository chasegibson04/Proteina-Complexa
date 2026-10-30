# Running Proteina-Complexa on AMD GPUs and on CPU

Upstream, Proteina-Complexa requires an NVIDIA CUDA GPU. It hard-asserts
`torch.cuda.is_available()`, calls `.cuda()` directly, hard-codes
`Trainer(accelerator="gpu")`, defaults to `bf16-mixed` precision, and depends on
`torch_scatter`, a compiled extension published only as CUDA/CPU wheels.

This document describes the multi-backend support added on top of that, how to
choose a backend, and the backend-specific limits that are worth knowing before
you commit to a long design run.

## Supported backends

| Backend | Hardware | Status |
|---|---|---|
| `cuda` | NVIDIA | Unchanged from upstream; still preferred when present. |
| `rocm` | AMD GPUs on ROCm-supported Linux (gfx90a, gfx942, gfx1100/1101/1102, gfx1200/1201, and gfx1030 on bare-metal Linux) | Supported. PyTorch's ROCm build presents as `torch.cuda`, so the CUDA path is reused. |
| `directml` | AMD / Intel / Qualcomm GPUs on Windows and WSL2 | Supported; output verified against CPU. Use `batch_size >= 8` — see [Measured throughput](#measured-throughput). The only route to a consumer AMD GPU under Windows. |
| `mps` | Apple silicon | Selected automatically; lightly tested. |
| `cpu` | Anything | Always available. Verified correct — see [Validation](#validation). |

Selection order is `cuda/rocm -> directml -> mps -> cpu`.

## Validation

Shipped `39_7V11_LIGAND` target, shipped 400 sampling steps, single-pass
search, released ligand checkpoint. Geometry checked with
`script_utils/gpu_time/validate_backend_outputs.py`:

| Backend | mean CA–CA (ideal 3.80 Å) | backbone bonds out of tolerance | clashes |
|---|---|---|---|
| CPU | 3.807 / 3.798 / 3.796 Å | 0 of 299, all three | 0 |
| DirectML | 3.807 / 3.798 / 3.797 Å | 0 of 299, all three | 0 |

The two backends agree sample for sample. A further eight DirectML samples at
batch size 8 all passed (3.785–3.803 Å).

Getting there required finding a defect that is worth knowing about if you port
anything else to DirectML: **in-place writes through a view made by mixed
integer+slice indexing are silently dropped**. The model builds its coordinate
tensor with `x_t_coord[:, :, 1, :] = bb_ca`, so every pairwise-distance feature
saw an all-zero structure. The run still completed and still wrote
plausible-looking PDBs — the global fold was right, because the flow integrates
the CA trace directly — but local geometry was wrong (CA–CA 4.86–5.55 Å, ~63%
of backbone bonds out of tolerance). See the caveats below.

## Measured throughput

AMD Radeon RX 6800 (16 GB, gfx1030) under WSL2 against an Intel Xeon Gold
5218R using 32 threads. 100-residue ligand binder, 400 steps, fp32, generation
time per sample:

| Batch size | CPU | DirectML | DirectML vs CPU |
|---|---|---|---|
| 1 | 244 s | 477 s | 0.51x (slower) |
| 3 | — | 257 s | — |
| 8 | 189 s | 134 s | **1.41x faster** |

**Use `batch_size >= 8` on DirectML.** It has no graph capture, so per-operation
dispatch cost dominates small batches — at batch size 1 the GPU is half the
speed of the CPU, and only batching makes it worthwhile. Set it with:

```bash
complexa generate ... ++generation.dataloader.batch_size=8                       ++generation.search.max_batch_size=8
```

On isolated pair-bias attention blocks DirectML is 2–5x faster than the CPU
(see `benchmark_backends.py`), so the end-to-end figure is dispatch-bound
rather than compute-bound. This is still well short of the hardware's
potential — the RX 6800 is roughly 16 TFLOP/s fp32 against roughly 1.3 for the
Xeon — but DirectML is a compatibility layer, not a tuned compute stack. If you
have a ROCm-supported GPU, use ROCm.

### Keep checkpoints off `/mnt`

Reading the 4 GB autoencoder checkpoint from a Windows drive through WSL's
drvfs mount ran at **8 MB/s (498 s)**; the same file from the WSL ext4
filesystem reads at **514 MB/s (7.6 s)**. Model loading dominated total runtime
until the checkpoints were moved. Copy them into the WSL filesystem and point
the config there:

```bash
cp -r /mnt/<drive>/.../ckpts /opt/complexa-ckpts
complexa generate ... ++ckpt_path=/opt/complexa-ckpts     ++autoencoder_ckpt_path=/opt/complexa-ckpts/complexa_ligand_ae.ckpt
```

## Choosing a backend

Detection is automatic. To pin one:

```bash
export COMPLEXA_DEVICE=cpu        # or cuda, rocm, directml, mps, cuda:1
```

Asking for a backend that is not present raises an error rather than quietly
falling back, so a benchmark can never silently run on the CPU.

To check what you have:

```bash
python -c "from proteinfoundation.utils.device_utils import log_device_banner; log_device_banner()"
```

To measure what it is worth:

```bash
python -m script_utils.gpu_time.benchmark_backends --lengths 128 256 384
```

## Why DirectML on an RX 6800, and not ROCm

If you have an RDNA2 card (RX 6800/6800 XT/6900 XT, gfx1030) on Windows, ROCm
is not an option and DirectML is. This is a driver limitation, not a
configuration mistake, and it is worth stating precisely because the failure
mode is silent.

ROCm *does* support gfx1030 on bare-metal Linux. Under WSL2 it does not. AMD's
WSL compatibility matrix lists only RDNA3 (RX 7700/7800/7900, PRO W7000) and
RDNA4 (RX 9000) cards. The measured consequence on this hardware:

- `/dev/dxg` exists and `libdxcore.so` is present, so WSL GPU passthrough itself
  is working.
- Installing `hsa-runtime-rocr4wsl-amdgpu` (AMD's WSL ROCr runtime) succeeds, and
  `rocminfo` reports `WSL environment detected`.
- `rocminfo` then enumerates **one agent: the CPU**. The Windows driver exposes
  no GPU agent to the WSL ROCm stack.
- `rocm-smi` reports `Driver not initialized (amdgpu not found in modules)`,
  which is expected under WSL and not itself the problem.

A detail worth recording if you try this yourself: the
`hsa-runtime-rocr4wsl-amdgpu` package is published for Ubuntu 24.04 in ROCm
6.3.4, 6.4, 6.4.1, 6.4.2 and 6.4.4, but **not** in 6.4.3. Picking 6.4.3 produces
a confusing "Unable to locate package" error that looks like a distro problem.

DirectML reaches the same card through Direct3D 12 instead, and works.

## DirectML caveats

These were found by testing each operation on an RX 6800; they are not
documented upstream. All are handled by `proteinfoundation.utils.device_utils`,
but they shape what is fast and what is possible.

### In-place writes through a sliced view are silently dropped

The worst defect found, and the one that produced wrong structures while every
individual operation still looked correct:

```python
t = torch.zeros(2, 4, 6, 3, device=dml)
t[:, :, 1, :] = src
t.abs().max()      # 0.0 -- the write vanished, nothing raised
```

`__getitem__` with an integer index followed by further slicing returns a
**copy** rather than a view there, so the write lands on a temporary and is
discarded. `select()` and `narrow()` do return real views whose in-place ops
work, so `_install_directml_compat()` rewrites `__setitem__` in terms of them,
and routes advanced indexing (`t[i, j] = v`) through `index_put_`. Anything
more exotic falls back to the CPU; `directml_fallback_stats()` reports how
often that happened, so a hot one can be found by measurement.

If you write DirectML code yourself, assume a slice is not a view.

### bfloat16 aborts the process

A bf16 tensor on a DirectML device calls `abort()` in native code:

```
[F] dml_util.cc:118] Invalid or unsupported data type BFloat16.
```

This is not a Python exception and cannot be caught with `try`/`except`. Since
the upstream default precision is `bf16-mixed`, `resolve_precision()` rewrites
it to `32-true` on DirectML and `supports_bfloat16()` returns `False`. Use fp32
or fp16.

### `float * bool` silently corrupts data

This is the most dangerous defect found, and the one most worth knowing about
if you are porting any other model to DirectML.

```python
x = torch.randn(2, 4, 3, device=dml)
m = torch.ones(2, 4, 1, dtype=torch.bool, device=dml)
(x * m).dtype      # torch.bool  -- on CPU and CUDA this is torch.float32
```

The result is a **bool tensor and the values are gone**. Every other backend
promotes to float. `x * mask[..., None]` is the standard masking idiom in this
codebase and appears in roughly forty places across the flow matcher and the
feature factories, so on DirectML the model was destroying its own coordinates
at the first mask.

It is dangerous because it is silent: bool is a legal dtype, so nothing raises
at the point of corruption. The only reason it surfaced at all is that a later
subtraction refuses bool operands. Had the next operation been another
multiply, the run would have completed and produced confident nonsense.

Rather than patch forty call sites — where the first one missed would leave
silent corruption behind — `device_utils` repairs multiplication itself when
the DirectML backend is selected, via `_install_directml_compat()`. That fixes
every site at once, including the vendored openfold and colabdesign code.
Scalars are untouched, and `bool * bool` stays bool so pair masks still work.
The patch is installed automatically by `get_backend()`, so no code path can
reach a DirectML tensor without it.

### `torch.cdist` returns the wrong shape

Also silent. `torch.cdist(a, b)` on `[2, 7, 3]` and `[2, 5, 3]` returns
`[2, 1, 7, 1]` instead of `[2, 7, 5]`, with no error, so the result broadcasts
into plausible nonsense wherever pairwise distances feed the pair
representation. `_install_directml_compat()` replaces it with an equivalent
built from ops that are correct on this backend.

### Three operations are broken outright

| Operation | Behaviour | Workaround |
|---|---|---|
| `torch.eye(n, device=...)` | Returns an **empty** tensor; the next operation on it dies with `std::system_error` | `device_utils.eye()` builds on the CPU and copies across |
| `torch.nn.functional.one_hot` | Fails on device | `device_utils.one_hot()` compares against `arange` |
| `Tensor.scatter_add_` | Aborts the process for every shape tested | `scatter_utils` aggregates with `index_add_` instead |

The `torch.eye` defect is the nastiest of the three: it is silent at the point
of creation, and it sat directly in the Kabsch alignment path
(`utils/align_utils.py`), which runs on every generated structure.

### Mixed-dtype `torch.where` segfaults

`torch.where(int64_tensor == 0, torch.tensor(1.0), int64_tensor)` — guarding a
divide by replacing zero counts — segfaults on DirectML because the operands
have different dtypes. `align_utils.mean_w_mask` now uses `clamp(min=1)`, which
says the same thing without the mixed-dtype branch.

### `torch.inference_mode` is unsafe

Operating on a view of a batch tensor inside `torch.inference_mode()` raises
`RuntimeError: Cannot set version_counter for inference tensor` on DirectML,
which kills sampling in the flow matcher. `device_utils.inference_context()`
returns `torch.inference_mode()` on backends that support it and
`torch.no_grad()` on DirectML.

### linalg routines fall back to the CPU

`torch.linalg.inv`, `eigh` and `solve` fail on DirectML with
`Unknown error -2147024809` (`E_INVALIDARG`). Call them through
`device_utils.cpu_fallback()`, which runs them on the CPU and returns the result
to the original device. `svd`, `det` and `cross` all work natively, so Kabsch
alignment itself stays on the GPU.

### Autocast

`torch.autocast(device_type="privateuseone")` fails on `aten::normal_`. Use
explicit `.half()` modules instead, or stay in fp32. `autocast_dtype()` never
returns bf16 on a backend that cannot survive it.

### Operation coverage

Of 47 operations exercised from this model's hot path, 43 run natively on
DirectML: `matmul`, `bmm`, `einsum`, `scaled_dot_product_attention` (including
with an additive attention bias), `softmax`, `layer_norm`, `gelu`, `silu`,
`embedding`, `gather`, `index_select`, `cdist`, `svd`, `det`, `cross`, `topk`,
`sort`, `cumsum`, `triu`, fp16 matmul, float64, and backward passes.

Unsupported operations print a warning and fall back to the CPU automatically
rather than failing, so coverage gaps cost performance, not correctness. The
exceptions are the three hard aborts listed above, which is why they are
intercepted rather than left to fall back.

## CPU-only operation

Everything runs on the CPU with `COMPLEXA_DEVICE=cpu`. Expect generation to be
slow but correct; precision is forced to fp32 because CPU bf16 is far slower.

Some pipeline stages are CPU-bound regardless of backend:

- **AlphaFold2 / ColabDesign** uses JAX. There is no ROCm JAX build for gfx1030
  and no DirectML JAX at all, so AF2 refolding and the AF2 reward run on the CPU
  on AMD hardware.
- **Amber relaxation** (`metric_utils.relax_protein`) only has a CUDA platform in
  OpenFold's OpenMM wrapper, so non-CUDA backends relax on the CPU.
- **tmol** force-field rewards ship CUDA kernels.
- **RoseTTAFold3** (`rc-foundry`) is NVIDIA-only.

## Dependency notes

`torch_scatter` is no longer required. It was used only for `scatter_mean`,
which `proteinfoundation.utils.scatter_utils` now implements with `index_add_`.
This removes a compiled extension that had no ROCm or DirectML wheels.

For DirectML specifically, `torch-directml` pins `torch==2.4.1`. Several
dependencies will happily upgrade torch and silently break GPU support; after
installing anything large, confirm with:

```bash
python -c "import torch, torch_directml; print(torch.__version__, torch_directml.device_count())"
```

## Line endings

`.gitattributes` now forces LF for `*.sh` and other text files. Without it, a
clone on Windows produces CRLF shell scripts that fail under bash, WSL and
Docker with `$'\r': command not found` — including `env/download_startup.sh`,
which means model weights cannot be downloaded at all.

## Testing

```bash
pytest tests/test_device_backends.py                        # active backend
COMPLEXA_DEVICE=cpu pytest tests/test_device_backends.py    # CPU
```

A DirectML abort takes down the whole pytest process, so when debugging a
suspected backend crash, run tests one at a time rather than trusting a single
session's first failure.
