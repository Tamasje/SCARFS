"""Validate the stable latent-ODE (DIRECT transport) mechanism — does it break the ∫S_E wall?

Rolls the latent with NO E∘D re-projection: z ← z + Δτ·model.latent_field(z,q), decoding only for
readout. Measures on held-out TEST cases: closed-loop composition drift, ∫S_E rollout error, the
one-step dynamics-map gain (stability), and the reconstruction floor decode(encode(Y_true)) — the
quantity the re-projection approach could not push below ~2.3e-2 while staying stable.

Gate (vs stageBt2: recon 2.3e-2, ∫S_E ~0.25, G_F 0.95): a FAITHFUL reconstruction (< 2.3e-2) WITH a
stable rollout (dynamics gain < 1) and ∫S_E < 0.25 means the stability-fidelity wall is broken.

Run: .venv/bin/python scripts/validate_stable_ode.py runs/stableode_k8 [--n-cases 40]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.models.common import thermo_features
from scarfs.models.neuralcoil import MergedCoil
from scarfs.models.thermo import SpeciesThermo
from scarfs.schema import MAJOR_SPECIES
from scarfs.training.datamodule import tripartite_case_split

MECH = "chem_ForTransport.yaml"
_TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _load(bundle: str):
    b = Path(bundle)
    spec = json.loads((b / "spec.json").read_text())
    sc = pickle.load((b / "scalers.pkl").open("rb"))
    cmod = spec["config_echo"]["model"]
    ea = spec["energy_active"]
    m = MergedCoil(
        n_dry=len(spec["input"]), n_energy_active=len(ea), latent_dim=int(cmod["latent_dim"]), n_thermo=4,
        decoder_hidden=tuple(cmod["decoder_hidden"]), rate_hidden=tuple(cmod["rate_hidden"]),
        latent_source_hidden=tuple(cmod["latent_source_hidden"]), energy_hidden=tuple(cmod["energy_hidden"]),
        n_transport=int(cmod.get("transport_outputs", 0) or 0),
        transport_hidden=tuple(cmod.get("transport_hidden", (64, 64))),
        encoder_hidden=tuple(cmod.get("encoder_hidden", ()) or ()))
    # strict=False: legacy bundles (e.g. stageBt2) predate field_damping_raw/latent_arcsinh_scale;
    # reproject mode doesn't use them (it reads s_z from spec), so defaults are harmless there.
    m.load_state_dict(torch.load(b / "model.pt", map_location="cpu", weights_only=True), strict=False); m.eval()
    tsc = sc["legacy_scalers"].thermo
    rs = sc["rate_scaler"]
    thermo = SpeciesThermo.from_mechanism_yaml(MECH, list(ea))
    return {
        "spec": spec, "model": m, "tsc": tsc, "rs": rs, "thermo": thermo,
        "cm": np.asarray(spec["composition_mean"], float), "cs": np.asarray(spec["composition_scale"], float),
        "input": spec["input"],
    }


def _relrmse_abs(pred, true, idx):
    return float(np.mean([np.sqrt(np.mean((pred[:, j] - true[:, j]) ** 2)) for j in idx]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--mode", choices=["direct", "reproject"], default="direct",
                    help="direct = stable-ODE z+=Δτ·latent_field; reproject = legacy z=P(z)+Δτ·ω_Z(P(z))")
    args = ap.parse_args()

    L = _load(args.bundle)
    m, tsc, rs, thermo = L["model"], L["tsc"], L["rs"], L["thermo"]
    cm, cs, inp = L["cm"], L["cs"], L["input"]
    df = load_database(args.db); sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15, seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    tau_col = sch.state["tau"]; cid_col = sch.meta["CaseID"]; abs_col = sch.energy_target_column()
    in_cols = [f"Y_{s}" for s in inp]
    majors = [s for s in MAJOR_SPECIES if s in inp]; mi = [inp.index(s) for s in majors]

    direct = (args.mode == "direct")
    s_z = torch.tensor(np.asarray(L["spec"]["arcsinh_latent_scale"], float), dtype=torch.float32)

    def qn(T, P):
        qq = thermo_features(np.atleast_1d(T), np.atleast_1d(P))
        return torch.tensor((qq - tsc.mean_) / tsc.scale_, dtype=torch.float32)

    @torch.no_grad()
    def head_z(z, q):  # latent the readout heads see: raw z (direct) or P(z)=E∘D (reproject)
        return z if direct else m.encoder(m.decode(z, q))

    @torch.no_grad()
    def advance(z, q, dt):  # one transport step
        if direct:
            return z + dt * m.latent_field(z, q)
        zp = m.encoder(m.decode(z, q))
        return zp + dt * (torch.sinh(m.latent_source(zp, q).clamp(-20, 20)) * s_z)

    @torch.no_grad()
    def decode_phys(z, q):
        y = m.decode(z, q).numpy()[0] * cs + cm
        y = np.maximum(y, 0.0); s = y.sum()
        return y / s if s > 1.0 else y

    @torch.no_grad()
    def s_e(z, q, T):
        r = rs.inverse_transform(m.rates_from_latent(z, q).numpy())
        return float(np.sum(r[0] * thermo.h_mass(np.atleast_1d(T))))

    cids = dft[cid_col].drop_duplicates().to_numpy()
    pick = cids[:: max(1, len(cids) // args.n_cases)][: args.n_cases]
    comp, eint, recon, gains, blew = [], [], [], [], 0
    for c in pick:
        sub = dft.loc[(dft[cid_col] == c).to_numpy()]
        o = np.argsort(sub[tau_col].to_numpy()); tau = sub[tau_col].to_numpy()[o]
        if len(tau) < 4 or not np.all(np.diff(tau) > 0):
            continue
        T = sub[tc].to_numpy()[o]; P = sub[pc].to_numpy()[o]
        Yt = sub[in_cols].to_numpy(float)[o]
        at = np.clip(sub[abs_col].to_numpy(float)[o], 0.0, None)
        n = len(tau); Yp = np.zeros_like(Yt); ap = np.zeros(n); Yr = np.zeros_like(Yt)
        with torch.no_grad():
            z = m.encode(torch.tensor(((Yt[0] - cm) / cs)[None], dtype=torch.float32))
            for i in range(n):
                q = qn(T[i], P[i])
                zh = head_z(z, q)                       # readout latent (raw z or P(z))
                Yp[i] = decode_phys(zh, q); ap[i] = s_e(zh, q, T[i])
                Yr[i] = decode_phys(m.encode(torch.tensor(((Yt[i] - cm) / cs)[None], dtype=torch.float32)), q)
                if i < n - 1:
                    z = advance(z, qn(T[i + 1], P[i + 1]), tau[i + 1] - tau[i])
        if not np.all(np.isfinite(Yp)):
            blew += 1; continue
        comp.append(_relrmse_abs(Yp, Yt, mi)); recon.append(_relrmse_abs(Yr, Yt, mi))
        Ip, It = float(_TRAP(ap, tau)), float(_TRAP(at, tau))
        eint.append(abs(Ip - It) / max(abs(It), 1.0))
        # one-step dynamics-map gain at sampled states along the true trajectory
        with torch.no_grad():
            zt = m.encode(torch.tensor(((Yt - cm) / cs), dtype=torch.float32))
            qf = torch.tensor((thermo_features(T, P) - tsc.mean_) / tsc.scale_, dtype=torch.float32)
            d = torch.randn_like(zt); d = d / (d.norm(dim=1, keepdim=True) + 1e-12) * 0.1
            dtc = torch.tensor(np.r_[np.diff(tau), np.diff(tau)[-1]], dtype=torch.float32)[:, None]
            g0 = advance(zt, qf, dtc); g1 = advance(zt + d, qf, dtc)
            gains.append(float(((g1 - g0).norm(dim=1) / 0.1).median()))
    g = np.array
    print(f"=== STABLE LATENT-ODE validation (direct transport, no re-projection) — {args.bundle} ===")
    print(f"cases={len(comp)}  blewup={blew}")
    print(f"  reconstruction floor decode(encode(Y))   median={np.median(recon):.4e}   (stageBt2 2.3e-2)")
    print(f"  closed-loop composition drift            median={np.median(comp):.4e}   (stageBt2 2.3e-2)")
    print(f"  ∫S_E rollout rel-err                     median={np.median(eint):.4f}    (stageBt2 ~0.25)")
    print(f"  one-step dynamics-map gain (stability)   median={np.median(gains):.4f}    (<1 ⇒ stable; stageBt2 G_F 0.95)")
    print("\n  WALL BROKEN iff: recon < 2.3e-2 AND gain < 1 AND ∫S_E < 0.25 (faithful + stable together).")


if __name__ == "__main__":
    main()
