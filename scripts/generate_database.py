"""Build the enriched training-case manifest (F1/F4/E-c) and, on the HPC, finalise the inlet flow.

Run with ``--no-flow`` anywhere (NumPy only) to inspect the sampled coverage; run without it on the
HPC (Cantera available) to fill ``mdot``/``U_in`` from gas properties. The resulting manifest is then
consumed by the existing ``Database_Generation_MB.py`` multiprocessing harness (CRACKSIM) to produce
``Database_FINAL.parquet``.

Examples::

    # Coverage check — no Cantera required (local)
    python scripts/generate_database.py --no-flow

    # Finalise flow (HPC)
    python scripts/generate_database.py --mech chem.yaml

    # Front-adaptive storage with D-sweep
    python scripts/generate_database.py --no-flow --storage-mode front_adaptive --frac-jump 0.03 \\
        --diameters 0.0306 0.05 0.1

    # Stride-5 mode (legacy default)
    python scripts/generate_database.py --no-flow --storage-mode stride --stride 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scarfs.data.config import DataGenConfig, ExportColumnsConfig, StorageConfig
from scarfs.data.generate import write_manifest
from scarfs.data.sampling import build_cases, coverage_summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the enriched SCARFS training-case manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- flow / mechanism ---------------------------------------------------------------
    parser.add_argument("--mech", default="chem.yaml", help="Cantera mechanism for flow finalisation.")
    parser.add_argument("--out", default="case_manifest.json", help="Output manifest path.")
    parser.add_argument(
        "--no-flow", action="store_true",
        help="Skip Cantera flow finalisation (coverage-check mode; local).",
    )

    # -- storage mode -------------------------------------------------------------------
    parser.add_argument(
        "--storage-mode", choices=["stride", "front_adaptive"], default="stride",
        help="Per-case storage policy: 'stride' (every Nth point) or 'front_adaptive' "
             "(store where |ΔS_E| > frac-jump × case-peak, with size cap).",
    )
    parser.add_argument(
        "--stride", type=int, default=5, metavar="N",
        help="Step interval for 'stride' storage mode.",
    )
    parser.add_argument(
        "--frac-jump", type=float, default=0.03, metavar="F",
        help="Fractional |ΔS_E| threshold for 'front_adaptive' mode (e.g. 0.03 = 3%%).",
    )
    parser.add_argument(
        "--min-every-nth", type=int, default=20, metavar="N",
        help="Minimum inter-point spacing cap for 'front_adaptive' mode.",
    )

    # -- D-sweep ------------------------------------------------------------------------
    parser.add_argument(
        "--diameters", nargs="+", type=float,
        default=[0.0306, 0.05, 0.1], metavar="D",
        help="Reactor diameters [m] for the D-sweep (space-separated).",
    )

    # -- enrichment counts --------------------------------------------------------------
    parser.add_argument("--n-body", type=int, default=1800, help="Body-regime case count.")
    parser.add_argument("--n-inlet", type=int, default=240, help="Inlet-seed case count.")
    parser.add_argument("--n-hight", type=int, default=120, help="High-T case count.")
    parser.add_argument("--n-tail", type=int, default=200, help="Tail-enrichment case count.")

    # -- export-column flags ------------------------------------------------------------
    parser.add_argument(
        "--no-dydt", action="store_true",
        help="Omit dYdt_* columns from the export (not recommended; reduces training options).",
    )
    parser.add_argument(
        "--export-wdot", action="store_true",
        help="Include wdot_* (molar rate) columns in the export.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    storage = StorageConfig(
        mode=args.storage_mode,
        every_nth=args.stride,
        max_frac_jump=args.frac_jump,
        min_every_nth=args.min_every_nth,
    )

    export_cols = ExportColumnsConfig(
        dydt=not args.no_dydt,
        wdot=args.export_wdot,
        absorption=True,  # always on
        tau=True,
        z=True,
    )

    cfg = DataGenConfig(
        diameters_m=tuple(args.diameters),
        n_body_cases=args.n_body,
        n_inlet_seed_cases=args.n_inlet,
        n_highT_cases=args.n_hight,
        n_tail_cases=args.n_tail,
        storage=storage,
        export_columns=export_cols,
    )

    cases = build_cases(cfg)

    if not args.no_flow:
        from scarfs.data.generate import finalize_flow
        finalize_flow(cases, mech_path=args.mech)

    write_manifest(cases, args.out)
    summary = coverage_summary(cases)
    print(json.dumps(summary, indent=2))
    print(
        f"Wrote {len(cases)} cases to {args.out} "
        f"(flow {'skipped' if args.no_flow else 'finalised'}, "
        f"storage={args.storage_mode}, "
        f"diameters={args.diameters})."
    )


if __name__ == "__main__":
    main()
