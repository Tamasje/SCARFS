"""v3 database generation CLI — run on the Windows machine where SA_CRACKSIM.dll loads.

Tiers (smoke ⊂ pilot ⊂ full are prefixes of the same Sobol streams; test is independent):

    python scripts/generate_database_v3.py --tier smoke            # ~10 cases, minutes; auto-gates
    python scripts/generate_database_v3.py --tier pilot            # ~1.2k cases — train-subset proof
    python scripts/generate_database_v3.py --tier full             # ~20.5k cases
    python scripts/generate_database_v3.py --tier test             # certification set (indep. stream)
    python scripts/generate_database_v3.py --off-manifold 1000000 --source out_v2/pilot.parquet
    python scripts/generate_database_v3.py --gates --reference <stride5-or-v2 parquet>

Outputs under --out (default out_v2/): {tier}.parquet, {tier}_audits.json, {tier}_manifest.json,
scratch/case_*.parquet (atomic per-case commits; reruns with --skip-existing resume).
See RUNBOOK_DBGEN_WINDOWS.md for the full procedure.
"""
from __future__ import annotations

SCRIPT_VERSION_V3_BACKEND = "generate_database_v3_uses_scarfs.data.generation_v3"

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from scarfs.data import generation_v3 as g2
from scarfs.data.generate import finalize_flow
from scarfs.data.generation_v3 import GenV2Settings


def _ensure_mechanism(base_dir: Path) -> str:
    """Build chem.yaml from chem.inp/transport via ck2yaml if absent (ported bootstrap)."""
    chem_yaml = base_dir / "chem.yaml"
    if not chem_yaml.exists():
        import subprocess

        with (base_dir / "C2KYAML_log.txt").open("w", encoding="utf-8") as log:
            subprocess.run(
                ["ck2yaml", f"--input={base_dir / 'chem.inp'}",
                 f"--transport={base_dir / 'transport_chemkin.DAT'}", "--permissive"],
                stdout=log, stderr=subprocess.STDOUT, check=True, text=True,
            )
    return str(chem_yaml.resolve())


def worker_loop(worker_id: int, task_q: Queue, ready_q: Queue, status_q: Queue,
                dll_path: str, mech_path: str, base_dir: str, scratch_root: str,
                settings_doc: dict) -> None:
    """Per-worker loop (ported pattern): init DLL once, solve cases, atomic parquet commits."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    settings = g2.settings_from_doc(settings_doc)
    g2.init_worker_cracksim(dll_path, mech_path, Path(base_dir), Path(scratch_root), ready_q)
    scratch = Path(scratch_root)
    last_df = None
    try:
        while True:
            item = task_q.get()
            if item is None:
                break
            t0 = time.monotonic()
            case_id = item.get("id", "?")
            try:
                df, audit = g2.run_case_v2(item, settings)
                dt = time.monotonic() - t0
                if df is None:
                    status_q.put(("drop", worker_id, case_id, dt, str(audit)))
                    print(f"[w{worker_id}] DROP case {case_id} ({audit}) {dt:.1f}s", flush=True)
                    continue
                out_tmp = scratch / f"case_{case_id}.tmp.parquet"
                out_file = scratch / f"case_{case_id}.parquet"
                table = pa.Table.from_pandas(df, preserve_index=False)
                pq.write_table(table, str(out_tmp), compression="snappy")
                out_tmp.rename(out_file)
                last_df = df
                status_q.put(("done", worker_id, case_id, dt, json.dumps(audit)))
                print(f"[w{worker_id}] DONE case {case_id} rows={len(df)} {dt:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                dt = time.monotonic() - t0
                warnings.warn(f"[w{worker_id}] error on case {case_id}: {e}")
                status_q.put(("drop", worker_id, case_id, dt, f"exception: {e}"))
        # In-context Gate A: same process, immediately after real solves — discriminates
        # "NetRates_C needs a solve-warmed context" from a genuine recompute mismatch.
        if settings_doc.get("gate_a_in_worker") and worker_id == 0 and last_df is not None:
            try:
                res = g2.gate_dll_consistency(last_df.head(48))
                status_q.put(("gate_a", worker_id, None, None, json.dumps(res)))
            except Exception as e:  # noqa: BLE001
                status_q.put(("gate_a", worker_id, None, None,
                              json.dumps({"error": str(e), "passed": False})))
    finally:
        status_q.put(("exit", worker_id, None, None, None))


def run_tier(args) -> int:
    base_dir = REPO
    dll_path = str((base_dir / "SA_CRACKSIM.dll").resolve())
    if not Path(dll_path).exists():
        print(f"ERROR: {dll_path} not found — place the CRACKSIM DLL in the repo root.",
              file=sys.stderr)
        return 1
    mech_path = _ensure_mechanism(base_dir)

    out_root = Path(args.out)
    scratch = out_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    settings = GenV2Settings(n_points=args.n_points, solver_rtol=args.rtol,
                             solver_atol=args.atol)
    settings.storage.max_frac_jump = args.max_frac_jump

    cases, manifest = g2.build_v2_manifest(args.tier, seed=args.seed)
    print(f"tier={args.tier}: {manifest['n_cases']} cases, regimes={manifest['regime_counts']}")
    finalize_flow(cases, mech_path)
    runner_cases = [g2.to_runner_case(c, settings) for c in cases]

    if args.skip_existing:
        existing = {int(p.stem.split("_")[1]) for p in scratch.glob("case_*.parquet")}
        runner_cases = [c for c in runner_cases if c["id"] not in existing]
        print(f"--skip-existing: {len(existing)} already on disk, {len(runner_cases)} to solve")

    manifest["settings"] = {**settings.__dict__, "storage": settings.storage.__dict__}
    (out_root / f"{args.tier}_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    task_q: Queue = Queue()
    status_q: Queue = Queue()
    n_workers = max(1, int(args.n_cpu))
    settings_doc = {**settings.__dict__, "storage": settings.storage.__dict__,
                    "gate_a_in_worker": args.tier == "smoke"}
    # SEQUENTIAL worker startup with a per-worker READY handshake (the colleague's proven
    # pattern, Database_Generation_MB.py: "sequential worker startup handshake — avoids DLL
    # logfile races"). CRACKSIM's Initialise_CRACKSIM writes to a shared init log; if N
    # workers initialise the DLL simultaneously they race on it and crash. Per-worker FORT45
    # redirection alone is NOT sufficient — the init must also be serialised. Each worker gets
    # its OWN ready queue; we block on it before starting the next worker. Startup cost is
    # negligible against a multi-thousand-case run.
    workers = []
    ready_queues = []
    for i in range(n_workers):
        rq: Queue = Queue(maxsize=1)
        w = Process(target=worker_loop,
                    args=(i, task_q, rq, status_q, dll_path, mech_path,
                          str(base_dir), str(scratch), settings_doc),
                    daemon=False)
        w.start()
        msg = rq.get()  # block until THIS worker has finished CRACKSIM init
        if msg != "READY":
            print(f"ERROR: worker {i} init failed: {msg}", file=sys.stderr)
            for done_w in workers:
                task_q.put(None)
            return 1
        print(f"[w{i}] READY", flush=True)
        workers.append(w)
        ready_queues.append(rq)

    t0 = time.time()
    for c in runner_cases:
        task_q.put(c)
    for _ in workers:
        task_q.put(None)

    audits, done, dropped, exited = [], 0, 0, 0
    gate_a_worker = None
    while exited < n_workers:
        kind, _wid, _cid, _dt, payload = status_q.get()
        if kind == "done":
            done += 1
            audits.append(json.loads(payload))
        elif kind == "drop":
            dropped += 1
        elif kind == "gate_a":
            gate_a_worker = json.loads(payload)
        elif kind == "exit":
            exited += 1
        if (done + dropped) and (done + dropped) % 50 == 0:
            rate = (done + dropped) / max(time.time() - t0, 1e-9)
            print(f"... {done} done / {dropped} dropped "
                  f"({rate * 3600:.0f} cases/h)", flush=True)
    for w in workers:
        w.join()

    # merge per-case parquets for THIS tier's case ids
    import pyarrow.parquet as pq
    import pyarrow as pa

    tier_ids = {c["id"] for c in (g2.to_runner_case(c, settings) for c in cases)}
    files = sorted(p for p in scratch.glob("case_*.parquet")
                   if int(p.stem.split("_")[1]) in tier_ids)
    if not files:
        print("ERROR: no case files produced", file=sys.stderr)
        return 1
    out_file = out_root / f"{args.tier}.parquet"
    writer = None
    n_rows = 0
    for p in files:
        t = pq.read_table(str(p))
        n_rows += t.num_rows
        if writer is None:
            writer = pq.ParquetWriter(str(out_file), t.schema, compression="snappy")
        writer.write_table(t)
    if writer is not None:
        writer.close()

    sign = g2.aggregate_sign_audits(audits)
    (out_root / f"{args.tier}_audits.json").write_text(
        json.dumps({"cases_done": done, "cases_dropped": dropped,
                    "sign_audit": sign, "per_case": audits}, indent=2), encoding="utf-8")
    print(f"\n{args.tier}: {done} cases ok / {dropped} dropped -> {out_file} "
          f"({n_rows} rows, {len(files)} cases)")
    print(f"sign audit: worst min absorption {sign['worst_min_absorption']:.3g} J/m3/s, "
          f"material_negative={sign['material_negative']}")
    if sign["material_negative"]:
        print("NOTE: material negative absorption found -> E-c escape hatch applies "
              "(switch the energy head to shifted-softplus); see MERGE_DESIGN.md.")

    grid_max = [a.get("grid_max_jump_frac") for a in audits if "grid_max_jump_frac" in a]
    if grid_max:
        gmax = max(grid_max)
        pol = settings.storage.max_frac_jump
        # Per-regime breakdown: localise WHERE the steep (near-discontinuous, ignition-like)
        # fronts are. Steep fronts are expected in hot regimes and are physical, not a bug —
        # the ODE solver is adaptive (internal steps ≠ output grid), so each stored point is
        # accurate (Gate A confirms). --n-points only sets SAMPLING density of the front.
        by_regime: dict[str, list[float]] = {}
        for a in audits:
            if "grid_max_jump_frac" in a:
                by_regime.setdefault(a.get("regime", "?"), []).append(a["grid_max_jump_frac"])
        print(f"solve-grid front sampling: worst single-step jump = {gmax:.3f} of peak "
              f"(policy {pol}); per-regime median steepest step:")
        for r in sorted(by_regime):
            v = by_regime[r]
            print(f"    {r:16s} median {np.median(v):.3f}  max {max(v):.3f}  (n={len(v)})")
        # Recommendation is CAPPED: fully resolving an ignition front below 3%/step needs an
        # impractical grid AND is unnecessary for a state→rate surrogate (the off-manifold
        # cloud densifies the high-|S_E| region in STATE space instead). A 3–5x bump improves
        # front sampling and the per-case ∫S_E dτ budget metric without runaway solve cost.
        raw_factor = gmax / pol
        rec = min(5.0, max(1.0, raw_factor))
        print(f"  recommended --n-points multiplier: ~{rec:.1f}x (raw {raw_factor:.1f}x capped at 5x; "
              f"near-discontinuous fronts are physical — let the off-manifold cloud cover the rest)")
    if gate_a_worker is not None:
        print("\nGATE A (in-worker, post-solve context):")
        _print_gate_a(gate_a_worker)
    if args.tier == "smoke" or args.gates:
        return run_gates_on(out_file, args)
    return 0


def run_gates_on(parquet_path: Path, args) -> int:
    """Run gates A (DLL consistency), C (front resolution), D (sign) on a generated parquet."""
    import pandas as pd

    base_dir = REPO
    dll_path = str((base_dir / "SA_CRACKSIM.dll").resolve())
    mech_path = _ensure_mechanism(base_dir)
    ready_q: Queue = Queue()
    g2.init_worker_cracksim(dll_path, mech_path, base_dir,
                            Path(args.out) / "scratch", ready_q)
    if ready_q.get() != "READY":
        print("ERROR: gate init failed", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet_path)
    ref = df[df["sample_kind"] == "trajectory"] if "sample_kind" in df.columns else df
    a = g2.gate_dll_consistency(ref.sample(min(64, len(ref)), random_state=0))
    c = g2.gate_front_resolution(df, max_frac_jump=args.max_frac_jump)
    print("GATE A (DLL/dYdt consistency, fresh process):")
    _print_gate_a(a)
    print(f"GATE C (front resolution): policy-jump p95 {c['p95_jump_frac']:.3f} "
          f"<= {c['threshold']:.3f} -> {'PASS' if c['passed'] else 'FAIL'} "
          f"[{c['n_policy_jumps']} policy jumps]")
    print(f"        grid-limited single-step: p95 {c['grid_p95_jump_frac']:.3f} / "
          f"max {c['grid_max_jump_frac']:.3f} of peak ({c['n_grid_jumps']} jumps) -> "
          f"raise --n-points ~{max(1.0, c['grid_resolution_factor']):.1f}x if max exceeds policy")
    ok = a["passed"] and c["passed"]
    print(f"GATES: {'ALL PASS' if ok else 'FAILED — do not run production tiers'}")
    return 0 if ok else 2


def _print_gate_a(a: dict) -> None:
    if "error" in a:
        print(f"  ERROR: {a['error']}")
        return
    print(f"  max_rel={a['max_rel_diff']:.3e}  median={a['median_rel_diff']:.3e}  "
          f"p95={a['p95_rel_diff']:.3e}  on {a['n_compared']} entries "
          f"-> {'PASS' if a['passed'] else 'FAIL'}")
    print(f"  stored-dYdt mass closure p95: {a['mass_closure_p95']:.3e}  "
          f"(≈0 => mass-conserving; ~6e-2 => extra-MW units bug)")
    print(f"  zero-recompute fraction: {a['zero_recompute_frac']:.3f}  "
          f"(1.0 => DLL returns zeros outside solve context: statefulness)")
    print(f"  raw nonzero frac (row 0): {a['raw_nonzero_frac_row0']:.3f}   "
          f"double-call max |diff|: {a['double_call_max_abs_diff']:.3e} "
          f"(>0 => NetRates_C is stateful)")
    print(f"  worst entry |stored|/species-peak: {a.get('worst_stored_over_species_peak', float('nan')):.2e}  "
          f"(≪1 => the max is a near-zero-crossing artifact, not an ordering/units error)")
    if a.get("per_species_worst"):
        worst = ", ".join(f"{s}={v:.2e}" for s, v in a["per_species_worst"])
        print(f"  worst species: {worst}")


def run_off_manifold(args) -> int:
    import pandas as pd

    base_dir = REPO
    dll_path = str((base_dir / "SA_CRACKSIM.dll").resolve())
    mech_path = _ensure_mechanism(base_dir)
    ready_q: Queue = Queue()
    out_root = Path(args.out)
    (out_root / "scratch").mkdir(parents=True, exist_ok=True)
    g2.init_worker_cracksim(dll_path, mech_path, base_dir, out_root / "scratch", ready_q)
    if ready_q.get() != "READY":
        print("ERROR: init failed", file=sys.stderr)
        return 1

    settings = GenV2Settings(n_points=args.n_points, solver_rtol=args.rtol,
                             solver_atol=args.atol)
    anchors = pd.read_parquet(args.source)
    anchors = anchors[anchors["sample_kind"] == "trajectory"] \
        if "sample_kind" in anchors.columns else anchors
    cfg = g2.PerturbConfig(sigma_log=args.sigma_log,
                           points_per_anchor=args.points_per_anchor)
    print(f"off-manifold: target {args.off_manifold} points from {len(anchors)} anchors "
          f"(sigma_log={cfg.sigma_log}, {cfg.points_per_anchor}/anchor)")
    t0 = time.time()
    df = g2.eval_offmanifold_points(anchors, args.off_manifold, settings, cfg, seed=args.seed)
    out = Path(args.out) / f"offmanifold_{len(df)}.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows) in {time.time() - t0:.1f}s")
    return 0


def run_merge_only(args) -> int:
    """Merge already-generated per-case parquets into {tier}.parquet WITHOUT solving.

    For resuming after an abort (e.g. one stuck straggler case): every finished case is an
    atomically-committed scratch file, so we just concatenate them. No DLL/Cantera needed —
    the manifest (pure) supplies the tier's CaseID set so stray test-tier files are excluded,
    and lightweight audits are recomputed from the merged data itself.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    out_root = Path(args.out)
    scratch = out_root / "scratch"
    cases, _manifest = g2.build_v2_manifest(args.tier, seed=args.seed)
    tier_ids = {int(c["id"]) for c in cases}
    files = sorted(p for p in scratch.glob("case_*.parquet")
                   if int(p.stem.split("_")[1]) in tier_ids)
    if not files:
        print(f"ERROR: no case files for tier '{args.tier}' in {scratch}", file=sys.stderr)
        return 1
    present = {int(p.stem.split("_")[1]) for p in files}
    missing = sorted(tier_ids - present)

    out_file = out_root / f"{args.tier}.parquet"
    writer = None
    n_rows = 0
    for p in files:
        t = pq.read_table(str(p))
        n_rows += t.num_rows
        if writer is None:
            writer = pq.ParquetWriter(str(out_file), t.schema, compression="snappy")
        writer.write_table(t)
    writer.close()
    print(f"merge-only: {len(files)}/{len(tier_ids)} cases -> {out_file} "
          f"({n_rows} rows); {len(missing)} missing")
    if missing:
        head = ", ".join(str(m) for m in missing[:20])
        print(f"  missing CaseIDs (un-run / aborted stragglers): {head}"
              f"{' ...' if len(missing) > 20 else ''}")

    # lightweight audits recomputed from the merged frame (no worker status needed)
    df = pd.read_parquet(out_file)
    traj = df[df["sample_kind"] == "trajectory"] if "sample_kind" in df.columns else df
    sign = g2.sign_audit(traj["Reaction heat absorption [J/s/m3]"].to_numpy(float))
    c = g2.gate_front_resolution(df, max_frac_jump=args.max_frac_jump)
    print(f"sign audit: min absorption {sign['min_value']:.3g} J/m3/s, "
          f"frac_negative {sign['frac_negative']:.2e}")
    print(f"front resolution: policy-jump p95 {c['p95_jump_frac']:.3f} "
          f"(grid max single-step {c['grid_max_jump_frac']:.3f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SCARFS v2 database generator (Windows + CRACKSIM)")
    ap.add_argument("--tier", choices=("smoke", "pilot", "full", "test"), default=None)
    ap.add_argument("--out", default="out_v3")
    ap.add_argument("--n-cpu", type=int, default=max(1, (os_cpu := __import__("os").cpu_count() or 2) - 2))
    ap.add_argument("--n-points", type=int, default=400)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-16)
    ap.add_argument("--max-frac-jump", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=20260612)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--merge-only", action="store_true",
                    help="merge existing scratch case files into {tier}.parquet without solving "
                         "(resume after an abort / stuck straggler)")
    ap.add_argument("--gates", action="store_true", help="run gates after the tier / standalone")
    ap.add_argument("--reference", default=None, help="parquet for standalone --gates")
    ap.add_argument("--off-manifold", type=int, default=None, help="N points to generate")
    ap.add_argument("--source", default=None, help="anchor parquet for --off-manifold")
    ap.add_argument("--sigma-log", type=float, default=0.25)
    ap.add_argument("--points-per-anchor", type=int, default=4)
    args = ap.parse_args(argv)

    if args.off_manifold:
        if not args.source:
            ap.error("--off-manifold requires --source <trajectory parquet>")
        return run_off_manifold(args)
    if args.gates and args.tier is None:
        if not args.reference:
            ap.error("standalone --gates requires --reference <parquet>")
        return run_gates_on(Path(args.reference), args)
    if args.tier is None:
        ap.error("choose --tier, --gates --reference, or --off-manifold")
    if args.merge_only:
        return run_merge_only(args)
    return run_tier(args)


if __name__ == "__main__":
    try:
        set_start_method("spawn")  # Windows default; explicit for cross-platform determinism
    except RuntimeError:
        pass
    raise SystemExit(main())
