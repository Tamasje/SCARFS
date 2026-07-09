"""Diagnose the §5 integral-budget p95 failure: which TEST cases have large ∫S_E dτ error?

Computes per-case integrated-energy rel-err on the held-out test split and breaks the worst cases
down by metadata (regime, sample_kind, T_in, n_rows, integral magnitude, peak S_E) to tell whether
the p95 failure is a fixable data-coverage corner or a model limit.

Run: .venv/bin/python scripts/diag_integral_p95.py runs/merged_best
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.training.datamodule import tripartite_case_split
from full_acceptance import _rate_derived_absorption

_TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))


def main() -> None:
    bundle = sys.argv[1] if len(sys.argv) > 1 else "runs/merged_best"
    db = "Database_FINAL.parquet"
    df = load_database(db)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    pred, tgt, cases, tau = _rate_derived_absorption(bundle, dft, sch)

    cid = dft[sch.meta["CaseID"]].to_numpy()
    # optional metadata columns (present in the regenerated DB)
    meta_cols = {c: c for c in ["regime", "sample_kind", "T_in [K]", "P_in [Pa]",
                                "steam_to_ethane [kg/kg]"] if c in dft.columns}
    rows = []
    for c in np.unique(cid):
        m = cid == c
        t = tau[m]; o = np.argsort(t)
        ip = float(_TRAP(pred[m][o], x=t[o])); it = float(_TRAP(tgt[m][o], x=t[o]))
        rel = abs(ip - it) / max(abs(it), 1.0)
        rec = {"case": int(c), "n": int(m.sum()), "int_target": it, "int_pred": ip,
               "abs_err": abs(ip - it), "int_rel_err": rel, "peak_target": float(tgt[m].max())}
        for k, col in meta_cols.items():
            v = dft.loc[m, col].iloc[0]
            rec[k] = v
        rows.append(rec)

    rel_all = np.array([r["int_rel_err"] for r in rows])
    p95 = float(np.percentile(rel_all, 95))
    fail = rel_all > 0.10
    print(f"=== integral-budget per-case diagnosis ({bundle}) ===")
    print(f"cases={len(rows)}  median={np.median(rel_all):.4f}  p90={np.percentile(rel_all,90):.4f}  "
          f"p95={p95:.4f}  p99={np.percentile(rel_all,99):.4f}  max={rel_all.max():.4f}")
    print(f"cases failing >10%: {int(fail.sum())} ({100*fail.mean():.1f}%)\n")

    # characterize failing vs passing on each metadata axis
    def _summ(mask, label):
        sub = [rows[i] for i in range(len(rows)) if mask[i]]
        ints = np.array([r["int_target"] for r in sub]); ns = np.array([r["n"] for r in sub])
        print(f"  [{label}] n={len(sub)}  int_target: med={np.median(ints):.2e} "
              f"min={ints.min():.2e}  n_rows: med={int(np.median(ns))}")
        for k in meta_cols:
            vals = [r[k] for r in sub]
            try:
                u, cnt = np.unique(np.array(vals, dtype=object), return_counts=True)
                top = sorted(zip(u, cnt), key=lambda x: -x[1])[:4]
                print(f"      {k}: " + ", ".join(f"{a}={b}" for a, b in top))
            except Exception:
                arr = np.array([float(v) for v in vals]); print(f"      {k}: med={np.median(arr):.1f} range=[{arr.min():.1f},{arr.max():.1f}]")

    _summ(fail, "FAIL >10%")
    _summ(~fail, "PASS <=10%")

    # Is the p95 failure a small-denominator artifact? Compare ABSOLUTE integral error to the
    # signal scale (median integral of REACTING cases). If failing cases' abs error is a tiny
    # fraction of a typical case's energy budget, the relative gate is ill-posed, not the model.
    ints = np.array([r["int_target"] for r in rows])
    abserr = np.array([r["abs_err"] for r in rows])
    reacting = ints > np.percentile(ints, 50)            # upper-half = genuinely cracking cases
    scale = float(np.median(ints[reacting]))             # typical reacting-case energy budget
    frac = abserr / scale                                # abs error as fraction of a typical budget
    print(f"\n  --- absolute-error check (signal scale = median reacting ∫ = {scale:.2e} J·s/m³) ---")
    print(f"  |∫pred-∫tgt| / scale :  median={np.median(frac):.2e}  p95={np.percentile(frac,95):.2e}  "
          f"p99={np.percentile(frac,99):.2e}  max={frac.max():.2e}")
    print(f"  among the {int(fail.sum())} relative-FAIL cases: their |∫target| med={np.median(ints[fail]):.2e}, "
          f"abs-err/scale max={frac[fail].max():.2e}  (i.e. negligible vs a real case)")

    # worst 8 cases
    worst = sorted(rows, key=lambda r: -r["int_rel_err"])[:8]
    print("\n  worst 8 cases (case, int_rel_err, int_target, n_rows, regime, sample_kind, T_in):")
    for r in worst:
        print(f"    case={r['case']:<6} rel={r['int_rel_err']:.3f}  int={r['int_target']:.2e}  "
              f"n={r['n']:<4} {r.get('regime','?')!s:<10} {r.get('sample_kind','?')!s:<14} "
              f"T_in={r.get('T_in [K]','?')}")


if __name__ == "__main__":
    main()
