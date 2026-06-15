"""Independent data-quality validation of a v2 parquet (DLL-suspicion check, Mac-side).

No DLL needed: catches a broken/mis-wired CRACKSIM by checking physics the data must obey —
mass conservation of dYdt, enthalpy self-consistency (NASA7 recompute vs the stored energy
column), composition validity, real conversion along trajectories, and correct rate signs for
the major cracking species. Run: .venv/bin/python runs/validate_pilot.py <parquet>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from scarfs.benchmark.loader import infer_schema
from scarfs.data.generation_v2 import gate_front_resolution, sign_audit
from scarfs.models.thermo import SpeciesThermo

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "pilot.parquet"
ABS = "Reaction heat absorption [J/s/m3]"


def pct(x):
    x = np.asarray(x, float)
    return f"p50={np.median(x):.3g} p95={np.percentile(x,95):.3g} max={np.max(x):.3g}"


def main() -> None:
    print(f"=== validating {PATH.name} ({PATH.stat().st_size/1e6:.0f} MB) ===")
    df = pd.read_parquet(PATH)
    schema = infer_schema(df)
    sp = list(schema.species)
    ycols = schema.y_columns(sp)
    dcols = schema.dydt_columns(sp)
    print(f"rows={len(df)}  species={len(sp)}  Y_cols={len(ycols)}  dYdt_cols={len(dcols)}")
    if "sample_kind" in df:
        print("sample_kind:", df["sample_kind"].value_counts().to_dict())
    if "regime" in df:
        print("regime:", df["regime"].value_counts().to_dict())
    if "CaseID" in df:
        print(f"CaseID: {df['CaseID'].nunique()} unique")

    # 1. finiteness
    Y = df[ycols].to_numpy(float)
    D = df[dcols].to_numpy(float)
    T = df[schema.state["T"]].to_numpy(float)
    rho = df[schema.state["rho"]].to_numpy(float)
    absn = df[ABS].to_numpy(float)
    bad = {"Y": int((~np.isfinite(Y)).sum()), "dYdt": int((~np.isfinite(D)).sum()),
           "T": int((~np.isfinite(T)).sum()), "absorption": int((~np.isfinite(absn)).sum())}
    print(f"\n[1] non-finite counts: {bad}")

    # 2. composition validity
    ysum = Y.sum(axis=1)
    print(f"[2] composition: Y>=0 frac={np.mean(Y>=-1e-12):.4f}  ΣY {pct(np.abs(ysum-1.0))} (|ΣY-1|)")

    # 3. dYdt mass closure (MW-free units check: Σ dYdt must be ~0)
    clo = np.abs(D.sum(axis=1)) / (np.abs(D).sum(axis=1) + 1e-300)
    print(f"[3] dYdt mass closure Σ/Σ|·|: {pct(clo)}  "
          f"(~0 good; ~6e-2 => double-MW units bug)")

    # 4. enthalpy self-consistency: NASA7 recompute Σ ρ h_mass dYdt vs stored absorption
    th = SpeciesThermo.from_mechanism_yaml(REPO / "chem_ForTransport.yaml", sp, missing_ok=True)
    keep = [s for s in sp if s in th.species]
    di = [sp.index(s) for s in keep]
    recompute = (th.h_mass(T) * (rho[:, None] * D[:, di])).sum(axis=1)
    m = np.abs(absn) > 1.0
    rel = np.abs(recompute[m] - absn[m]) / np.abs(absn[m])
    corr = np.corrcoef(recompute[m], absn[m])[0, 1]
    print(f"[4] energy self-consistency (NASA7 recompute vs stored, {len(keep)}/{len(sp)} sp): "
          f"corr={corr:.6f}  relerr {pct(rel)}")
    print(f"    (corr~1 & relerr~0 => dYdt↔energy↔thermo mutually consistent)")

    # 5. absorption physicality
    sa = sign_audit(absn)
    print(f"[5] absorption: {pct(np.abs(absn))} J/m3/s  min={sa['min_value']:.3g}  "
          f"frac_neg={sa['frac_negative']:.2e}  (cracking is endothermic => mostly >0)")

    # 6. real conversion + correct major-species signs (trajectory rows only)
    traj = df[df["sample_kind"] == "trajectory"] if "sample_kind" in df else df
    if len(traj) == 0:
        print("[6] (off-manifold file: no trajectories — conversion/front checks N/A)")
        traj = df.iloc[:0]
    if len(traj) and "CaseID" in traj and "C2H6" in sp:
        ye = traj.groupby("CaseID")["Y_C2H6"]
        conv = (ye.first() - ye.min()) / ye.first().clip(1e-9)
        print(f"[6] ethane conversion per case: {pct(conv.to_numpy())}  "
              f"(>0 => trajectories actually react, not frozen)")
    # sign sanity at the most-reacting rows
    if "C2H6" in keep and "C2H4" in keep:
        react = np.argsort(np.abs(D[:, sp.index("C2H6")]))[::-1][:2000]
        f_c2h6_neg = np.mean(D[react, sp.index("C2H6")] < 0)
        f_c2h4_pos = np.mean(D[react, sp.index("C2H4")] > 0)
        print(f"    at top-reacting rows: dYdt(C2H6)<0 frac={f_c2h6_neg:.3f}  "
              f"dYdt(C2H4)>0 frac={f_c2h4_pos:.3f}  (both ~1 => correct cracking chemistry)")

    # 7. front resolution (Gate C)
    if "tau [s]" in df and "PFR point index" in df:
        c = gate_front_resolution(df, max_frac_jump=0.03)
        print(f"[7] front resolution: policy-jump p95={c['p95_jump_frac']:.3f} "
              f"(≤0.045 good); grid single-step max={c['grid_max_jump_frac']:.3f} "
              f"({c['n_policy_jumps']} policy / {c['n_grid_jumps']} grid jumps)")

    # 8. degenerate-species census (informs sigma_floor)
    sig = Y.std(axis=0)
    print(f"[8] σ<1e-10 species: {int((sig<1e-10).sum())}/{len(sp)} "
          f"(de-activated by composition_sigma_floor=1e-10)")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
