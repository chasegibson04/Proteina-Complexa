"""Check that generated structures are chemically sane, and compare backends.

A GPU backend can run to completion and still emit nonsense -- DirectML's
``float * bool`` promotion bug did exactly that, silently. This script reads
generated PDBs and verifies the geometry a real protein backbone must satisfy,
then compares two runs so a CPU and a GPU result can be held side by side.

Usage::

    python -m script_utils.gpu_time.validate_backend_outputs RUN_A [RUN_B]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Ideal backbone geometry (Engh & Huber), in angstroms.
IDEAL_BONDS = {
    ("N", "CA"): 1.458,
    ("CA", "C"): 1.525,
    ("C", "N"): 1.329,  # peptide bond, residue i -> i+1
}
BOND_TOLERANCE = 0.25


def parse_backbone(pdb_path: Path) -> list[dict]:
    """Return per-residue backbone atom coordinates from a PDB file."""
    residues: dict[tuple[str, int], dict] = {}
    order: list[tuple[str, int]] = []
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in ("N", "CA", "C", "O"):
            continue
        chain = line[21]
        resseq = int(line[22:26])
        key = (chain, resseq)
        if key not in residues:
            residues[key] = {"chain": chain, "resseq": resseq}
            order.append(key)
        residues[key][name] = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    return [residues[k] for k in order]


def dist(a: tuple, b: tuple) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def check(pdb_path: Path) -> dict:
    """Validate one structure; return a dict of metrics and problems."""
    res = parse_backbone(pdb_path)
    report: dict = {"path": pdb_path, "n_residues": len(res), "problems": []}
    if not res:
        report["problems"].append("no backbone atoms parsed")
        return report

    coords = [r["CA"] for r in res if "CA" in r]
    if len(coords) != len(res):
        report["problems"].append(f"{len(res) - len(coords)} residues missing CA")

    # Non-finite coordinates
    flat = [c for r in res for n in ("N", "CA", "C") if (c := r.get(n)) for _ in (0,)]
    if any(not all(map(math.isfinite, xyz)) for xyz in flat):
        report["problems"].append("non-finite coordinates")

    # Intra-residue and peptide bond lengths
    bad_bonds = 0
    total_bonds = 0
    for i, r in enumerate(res):
        for (a, b), ideal in IDEAL_BONDS.items():
            if a == "C" and b == "N":
                if i + 1 >= len(res) or "C" not in r or "N" not in res[i + 1]:
                    continue
                d = dist(r["C"], res[i + 1]["N"])
            else:
                if a not in r or b not in r:
                    continue
                d = dist(r[a], r[b])
            total_bonds += 1
            if abs(d - ideal) > BOND_TOLERANCE:
                bad_bonds += 1
    report["bonds_checked"] = total_bonds
    report["bonds_out_of_range"] = bad_bonds
    if total_bonds:
        frac = bad_bonds / total_bonds
        report["bad_bond_fraction"] = frac
        if frac > 0.10:
            report["problems"].append(f"{frac:.0%} of backbone bonds outside tolerance")

    # Consecutive CA-CA should be ~3.8 A
    ca_steps = [dist(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    if ca_steps:
        mean_step = sum(ca_steps) / len(ca_steps)
        report["mean_ca_ca"] = mean_step
        off = sum(1 for s in ca_steps if abs(s - 3.8) > 0.5)
        report["ca_ca_out_of_range"] = off
        if abs(mean_step - 3.8) > 0.3:
            report["problems"].append(f"mean CA-CA {mean_step:.2f} A (expected ~3.8)")

    # Radius of gyration: a folded domain is compact, not an exploded cloud.
    if coords:
        cx = sum(c[0] for c in coords) / len(coords)
        cy = sum(c[1] for c in coords) / len(coords)
        cz = sum(c[2] for c in coords) / len(coords)
        rg = math.sqrt(sum(dist(c, (cx, cy, cz)) ** 2 for c in coords) / len(coords))
        report["radius_of_gyration"] = rg
        # Empirical: Rg ~ 2.2 * N^0.38 for a compact globular domain.
        expected = 2.2 * len(coords) ** 0.38
        report["expected_rg"] = expected
        if rg > 3 * expected:
            report["problems"].append(f"Rg {rg:.1f} A vs expected ~{expected:.1f} (not compact)")

    # Steric clashes between non-adjacent CA atoms
    clashes = sum(
        1
        for i in range(len(coords))
        for j in range(i + 3, len(coords))
        if dist(coords[i], coords[j]) < 3.0
    )
    report["ca_clashes"] = clashes
    if clashes:
        report["problems"].append(f"{clashes} CA-CA clashes under 3.0 A")

    return report


def show(report: dict) -> None:
    print(f"\n  {report['path'].name}")
    print(f"    residues            {report['n_residues']}")
    if "bonds_checked" in report:
        print(f"    backbone bonds      {report['bonds_checked']} checked, "
              f"{report['bonds_out_of_range']} outside +/-{BOND_TOLERANCE} A")
    if "mean_ca_ca" in report:
        print(f"    mean CA-CA          {report['mean_ca_ca']:.3f} A (ideal 3.80)")
    if "radius_of_gyration" in report:
        print(f"    radius of gyration  {report['radius_of_gyration']:.1f} A "
              f"(expected ~{report['expected_rg']:.1f})")
    print(f"    CA-CA clashes       {report.get('ca_clashes', 0)}")
    if report["problems"]:
        for p in report["problems"]:
            print(f"    PROBLEM: {p}")
    else:
        print("    OK: geometry is consistent with a real protein backbone")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="generation output directories")
    args = ap.parse_args()

    all_reports: dict[Path, list[dict]] = {}
    for run in args.runs:
        pdbs = sorted(run.rglob("*_binder.pdb"))
        if not pdbs:
            pdbs = sorted(run.rglob("*.pdb"))
        print(f"\n{'=' * 70}\n{run}  ({len(pdbs)} structures)\n{'=' * 70}")
        reports = []
        for pdb in pdbs:
            rep = check(pdb)
            reports.append(rep)
            show(rep)
        all_reports[run] = reports

    if len(args.runs) == 2:
        a, b = args.runs
        print(f"\n{'=' * 70}\ncomparison: {a.name}  vs  {b.name}\n{'=' * 70}")
        for ra, rb in zip(all_reports[a], all_reports[b], strict=False):
            print(f"  {ra['path'].name}")
            for key, label, fmt in [
                ("n_residues", "residues", "{}"),
                ("mean_ca_ca", "mean CA-CA", "{:.3f}"),
                ("radius_of_gyration", "Rg", "{:.1f}"),
                ("bad_bond_fraction", "bad bonds", "{:.1%}"),
                ("ca_clashes", "clashes", "{}"),
            ]:
                va, vb = ra.get(key), rb.get(key)
                if va is None or vb is None:
                    continue
                print(f"    {label:12s} {fmt.format(va):>10s}  vs {fmt.format(vb):>10s}")

    total_problems = sum(len(r["problems"]) for reps in all_reports.values() for r in reps)
    print(f"\n{'=' * 70}")
    print("ALL STRUCTURES VALID" if total_problems == 0 else f"{total_problems} problem(s) found")
    return 0 if total_problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
