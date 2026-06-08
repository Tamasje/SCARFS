"""Build the enriched training-case manifest (F1/F4) and, on the HPC, finalise the inlet flow.

Run with ``--no-flow`` anywhere (NumPy only) to inspect the sampled coverage; run without it on the
HPC (Cantera available) to fill ``mdot``/``U_in`` from gas properties. The resulting manifest is then
consumed by the existing ``Database_Generation_MB.py`` multiprocessing harness (CRACKSIM) to produce
``Database_FINAL.parquet``.

    python scripts/generate_database.py --no-flow                 # coverage check (local)
    python scripts/generate_database.py --mech chem.yaml          # finalise flow (HPC)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scarfs.data.config import DataGenConfig
from scarfs.data.generate import write_manifest
from scarfs.data.sampling import build_cases, coverage_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the enriched SCARFS training-case manifest.")
    parser.add_argument("--mech", default="chem.yaml", help="Cantera mechanism for flow finalisation.")
    parser.add_argument("--out", default="case_manifest.json", help="Output manifest path.")
    parser.add_argument("--no-flow", action="store_true", help="Skip Cantera flow finalisation (local).")
    args = parser.parse_args(argv)

    cfg = DataGenConfig()
    cases = build_cases(cfg)
    if not args.no_flow:
        from scarfs.data.generate import finalize_flow

        finalize_flow(cases, mech_path=args.mech, diam_m=cfg.diam_m)

    write_manifest(cases, args.out)
    print(json.dumps(coverage_summary(cases), indent=2))
    print(f"Wrote {len(cases)} cases to {args.out} (flow {'skipped' if args.no_flow else 'finalised'}).")


if __name__ == "__main__":
    main()
