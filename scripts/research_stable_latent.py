"""Research: stabilize the latent transport by multi-step PUSHFORWARD fine-tuning of ω_Z ONLY.

The free-running latent transport dZ/dτ = ω_Z(Z) is a-posteriori unstable (off-manifold
amplification). The deployed ENERGY is rate-derived and ω_Z only drives transport, so we can freeze
the encoder/decoder/rate/energy heads (preserving the a-priori 7.75× EXACTLY) and retrain ONLY
latent_source_net with a pushforward objective: roll K steps feeding the model its OWN output and
match the true latent trajectory. Unlike the failed 1-step Lagrangian, this trains the model on the
off-manifold states it actually visits, teaching ω_Z to correct its own drift.

    z_0 = E·y_std(inlet);  for j=1..K:  z_j = project(z_{j-1}) + Δτ_j·ω_Z(project(z_{j-1}))
    loss = mean_j ‖z_j − z_j^true‖²        (encoder/decoder frozen ⇒ project frozen; only ω_Z trains)

Writes runs/merged_stablelatent (spec.json + scalers.pkl copied from the source bundle, new model.pt).
Evaluate with: scripts/aposteriori_rollout.py runs/merged_stablelatent  and  scripts/pfr_1d_diffusion.py.

Run: .venv/bin/python scripts/research_stable_latent.py [--src runs/merged_best] [--K 12] [--epochs 60]
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.models.common import thermo_features
from scarfs.models.neuralcoil import MergedCoil
from scarfs.training.datamodule import tripartite_case_split


def build_model(spec, device):
    m = spec["config_echo"]["model"]
    model = MergedCoil(
        n_dry=len(spec["input"]), n_energy_active=len(spec["target"]),
        latent_dim=int(m["latent_dim"]), n_thermo=4,
        decoder_hidden=tuple(m["decoder_hidden"]), rate_hidden=tuple(m["rate_hidden"]),
        latent_source_hidden=tuple(m["latent_source_hidden"]), energy_hidden=tuple(m["energy_hidden"]),
        activation=m.get("activation", "silu"), spectral_norm=bool(m.get("spectral_norm", False)),
        n_transport=int(m.get("transport_outputs", 0) or 0),
        transport_hidden=tuple(m.get("transport_hidden", (64, 64))),
    )
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/merged_best")
    ap.add_argument("--out", default="runs/merged_stablelatent")
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--K", type=int, default=12, help="pushforward horizon (steps)")
    ap.add_argument("--stride", type=int, default=3, help="window stride within a case")
    ap.add_argument("--max-cases", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fidelity", type=float, default=0.0,
                    help="optional weight to keep ω_Z near its ORIGINAL values (anti-forget)")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    src = Path(args.src)
    spec = json.loads((src / "spec.json").read_text())
    scalers = pickle.load((src / "scalers.pkl").open("rb"))
    cm = np.asarray(spec["composition_mean"], float); cs = np.asarray(spec["composition_scale"], float)
    tsc = scalers["legacy_scalers"].thermo
    s_z = torch.tensor(np.asarray(spec["arcsinh_latent_scale"], float), dtype=torch.float32, device=dev)
    st = spec["export_stats"]
    Tmin, Tmax, Pmin, Pmax = st["T_train_min"], st["T_train_max"], st["P_train_min"], st["P_train_max"]
    env_lo = torch.tensor(np.asarray(st["latent_env_min"], float), dtype=torch.float32, device=dev)
    env_hi = torch.tensor(np.asarray(st["latent_env_max"], float), dtype=torch.float32, device=dev)

    model = build_model(spec, dev)
    model.load_state_dict(torch.load(src / "model.pt", map_location=dev, weights_only=True))
    model.eval()  # freeze BN/dropout (none here); we only grad latent_source_net

    # ---- data: train split, build K-step windows per case --------------------------------
    df = load_database(args.db); sch = infer_schema(df)
    tr, _va, _te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[tr].reset_index(drop=True)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    tau_col = sch.state["tau"]; cid_col = sch.meta["CaseID"]
    insp = spec["input"]
    cids = dft[cid_col].drop_duplicates().to_numpy()[: args.max_cases]
    K = args.K

    q_seqs, ztrue_seqs, dtau_seqs = [], [], []
    Yc = dft[[f"Y_{s}" for s in insp]].to_numpy(float)
    Tc = dft[tc].to_numpy(float); Pc = dft[pc].to_numpy(float); tauc = dft[tau_col].to_numpy(float)
    cidc = dft[cid_col].to_numpy()
    with torch.no_grad():
        for c in cids:
            idx = np.where(cidc == c)[0]
            idx = idx[np.argsort(tauc[idx])]
            if len(idx) < K + 1:
                continue
            ystd = (Yc[idx] - cm) / cs
            Tn = np.clip(Tc[idx], Tmin, Tmax); Pn = np.clip(Pc[idx], Pmin, Pmax)
            q = (thermo_features(Tn, Pn) - tsc.mean_) / tsc.scale_
            zt = model.encode(torch.tensor(ystd, dtype=torch.float32, device=dev))
            zt = torch.clamp(zt, env_lo, env_hi).cpu().numpy()
            tau = tauc[idx]
            for s in range(0, len(idx) - K, args.stride):
                q_seqs.append(q[s:s + K + 1]); ztrue_seqs.append(zt[s:s + K + 1])
                dtau_seqs.append(np.diff(tau[s:s + K + 1]))
    Q = torch.tensor(np.asarray(q_seqs), dtype=torch.float32, device=dev)          # (B,K+1,4)
    ZT = torch.tensor(np.asarray(ztrue_seqs), dtype=torch.float32, device=dev)     # (B,K+1,k)
    DT = torch.tensor(np.asarray(dtau_seqs), dtype=torch.float32, device=dev)      # (B,K)
    B = Q.shape[0]
    print(f"device={dev}  windows={B}  K={K}  (train cases used={len(cids)})", flush=True)

    # ---- freeze everything except latent_source_net --------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.latent_source_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.latent_source_net.parameters(), lr=args.lr)

    def omega(zproj, q):  # physical ω_Z
        return torch.sinh(model.latent_source(zproj, q).clamp(-20, 20)) * s_z

    def _clamp(z):
        return torch.clamp(z, env_lo, env_hi)

    # PUSHFORWARD TRICK (Brandstetter et al.): roll forward WITHOUT grad (clamped to envelope, as
    # deployment does) to reach the drifted states the model actually visits, then take a SINGLE
    # gradient step from each visited (detached) state to the next true latent. Back-propagating
    # through the full unstable unroll explodes (1e28); this bounds the loss and still trains ω_Z
    # to CORRECT its own drift — the thing 1-step Lagrangian (true states only) can't do.
    def pushforward_loss(bs=8192, train=True):
        order = torch.randperm(B, device=dev) if train else torch.arange(B, device=dev)
        tot, n = 0.0, 0
        for st0 in range(0, B, bs):
            sel = order[st0:st0 + bs]
            q = Q[sel]; zt = ZT[sel]; dt = DT[sel]
            with torch.no_grad():                       # collect the visited (drifted) states
                visited = [zt[:, 0, :]]; z = zt[:, 0, :]
                for j in range(1, K):
                    zp = model.project(z, q[:, j, :])
                    z = _clamp(zp + dt[:, j - 1].unsqueeze(1) * omega(zp, q[:, j, :]))
                    visited.append(z)
            loss = 0.0
            for j in range(K):                          # 1-step grad: visited[j] -> true z_{j+1}
                zin = visited[j].detach()
                zp = model.project(zin, q[:, j + 1, :])
                zpred = zp + dt[:, j].unsqueeze(1) * omega(zp, q[:, j + 1, :])
                loss = loss + ((zpred - zt[:, j + 1, :]) ** 2).mean()
            loss = loss / K
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.latent_source_net.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach()) * len(sel); n += len(sel)
        return tot / max(n, 1)

    with torch.no_grad():
        base = pushforward_loss(train=False)
    print(f"pushforward loss (pre-finetune) = {base:.4f}", flush=True)
    for ep in range(args.epochs):
        tr_loss = pushforward_loss(train=True)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep+1}/{args.epochs}] pushforward loss = {tr_loss:.4f}", flush=True)

    # ---- save fine-tuned bundle ----------------------------------------------------------
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "spec.json", out / "spec.json")
    shutil.copy(src / "scalers.pkl", out / "scalers.pkl")
    torch.save(model.state_dict(), out / "model.pt")
    print(f"\nsaved fine-tuned bundle -> {out}\n"
          f"evaluate: .venv/bin/python scripts/aposteriori_rollout.py {out} --n-cases 30 --substeps 1,50\n"
          f"          .venv/bin/python scripts/pfr_1d_diffusion.py {out}\n"
          f"          .venv/bin/python scripts/full_test_eval.py {out}  (a-priori should be unchanged)")


if __name__ == "__main__":
    main()
