#!/usr/bin/env python
# coding: utf-8
#
# =========================
# Database Generation: multi-core, robust, broadened space
# - inlet Reynolds numbers -> compute U_in and mdot from Re, mu_in, rho_in, D
# - vary length to change residence time
# - rich, non-negative heat-flux shapes (no cooling)
# - store full thermochemical state + rates + properties
# - keep CWD at base_dir for DLL; per-worker Fortran unit redirection
# - sequential worker startup handshake (avoids DLL logfile races)
# - ck2yaml once (parent)
# - Parquet streaming + optional merged CSV
#
# Mike Bonheure  - Thesis Louis Bocqué
# V1: 12/11/25
# Modified: 12/11/25
# =========================


import os
import time
import warnings
import tempfile
from pathlib import Path
from multiprocessing import Process, JoinableQueue, Queue

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
np.infty = np.inf

#INSTALLEER PARQUET, is sneller voor het opslaan van data (en ook minder memory intensive)
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ctypes as xt
import cantera as ct

from ideal_reactor_models import customPFR  


fortlib = None            # ctypes.CDLL for CRACKSIM
_gas_cache = None         # per-worker Cantera Solution
REAC_MECH_PATH = None     # absolute path to chem.yaml
gas_id = "gas"

# Helper functies
def _interp_func(x, y):
    """Return a pure-Python callable f(z) = interp(x,y)(z) """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    def f(z):
        return float(np.interp(z, x, y))
    return f

def make_piecewise(z, q):
    """Normalise inputs and return a callable heat-flux function (non-negative)."""
    z = np.asarray(z, dtype=float)
    q = np.maximum(0.0, np.asarray(q, dtype=float))  # enforce no cooling
    return _interp_func(z, q)

def _prop_val(obj, name):
    """Return attribute either by calling (if callable) or by value (if float)."""
    attr = getattr(obj, name)
    return attr() if callable(attr) else float(attr)

# Hieronder definëren we een range aan heat flux profiles, feel free to modify of nog aan toe te voegen
def hf_uniform(L, H):
    z = np.array([0.0, L], dtype=float)
    q = np.full_like(z, H, dtype=float)
    return z, q, {"shape": "uniform", "H": H}

def hf_pulsed(L, H, N, Np_req=None, mode="uniform", jitter=0.35,
              seed=None, w_samples=2, snap_to_grid=False):
    """
    heating pieken langs de reactor (zoals een turboreactor), met N het aantal pieken en H de power input (aka hoogte van de piek)

    - w: bepaalt op hoeveel punten in ons grid we zullen heaten (w=2, heat verdeelt over 2 naburige punten) 
    - mode="uniform": Uniforme verdeling over de lengte
      mode="jitter":  start uniform maar nadien komt er wat irrigulariteit op (bepaal jezelf) zodat er speling is op wanneer die pulses komen
    """
    import numpy as _np

    N = int(max(1, N))
    dz = (L / max(Np_req - 1, 1)) if (Np_req and Np_req > 1) else None

    base_w = 0.02 * L / max(N, 1)
    w = max(base_w, (w_samples * dz) if dz else base_w)
    margin = 0.5 * w
    usable = max(L - 2.0 * margin, 1e-9)
    s_nom = usable / max(N - 1, 1)

    if mode == "uniform" or N == 1:
        centres = margin + _np.arange(N) * s_nom
    elif mode == "jitter":
        rng = _np.random.default_rng(seed)
        centres = margin + _np.arange(N) * s_nom
        jitter_amp = jitter * s_nom
        centres = centres + rng.uniform(-jitter_amp, +jitter_amp, size=N)
        min_gap = 0.60 * w
        centres = _np.sort(_np.clip(centres, margin, L - margin))
        for i in range(1, N):
            if centres[i] - centres[i-1] < min_gap:
                centres[i] = centres[i-1] + min_gap
        overflow = centres[-1] - (L - margin)
        if overflow > 0:
            centres -= overflow
            under = margin - centres[0]
            if under > 0:
                centres += under
        for i in range(1, N):
            if centres[i] - centres[i-1] < min_gap:
                centres[i] = centres[i-1] + min_gap
        centres = _np.clip(centres, margin, L - margin)
    else:
        raise ValueError(f"hf_pulsed: unknown mode '{mode}'")

    if snap_to_grid and dz:
        centres = _np.round(centres / dz) * dz
        centres = _np.clip(centres, margin, L - margin)
        for i in range(1, N):
            if centres[i] <= centres[i-1]:
                centres[i] = min(L - margin, centres[i-1] + max(0.60 * w, dz))

    z = [0.0]; q = [0.0]; half_w = 0.5 * w
    for c in centres:
        z.extend([max(0.0, c - half_w), c, min(L, c + half_w)])
        q.extend([0.0, H, 0.0])

    z = _np.array(z, dtype=float)
    q = _np.maximum(_np.array(q, dtype=float), 0.0)
    idx = _np.argsort(z)
    z, q = z[idx], q[idx]

    meta = {"shape": "pulsed", "H": float(H), "N": int(N),
            "w": float(w), "mode": mode, "jitter": float(jitter),
            "snap_to_grid": bool(snap_to_grid), "w_samples": int(w_samples)}
    return z, q, meta


def hf_front_ramp(L, H, k=3.0, Np_req=None, samples_per_cell=4):
    ncp = max(200, (samples_per_cell * (Np_req or 0)) or 0)
    if ncp <= 0: ncp = 200
    z = np.linspace(0.0, L, ncp)
    s = z / L
    q = H * np.exp(-k * s)
    return z, q, {"shape": "front_ramp", "H": H, "k": float(k)}

def hf_back_ramp(L, H, k=3.0, Np_req=None, samples_per_cell=4):
    ncp = max(200, (samples_per_cell * (Np_req or 0)) or 0)
    if ncp <= 0: ncp = 200
    z = np.linspace(0.0, L, ncp)
    s = z / L
    q = H * np.exp(-k * (1.0 - s))
    return z, q, {"shape": "back_ramp", "H": H, "k": float(k)}

def hf_triangular(L, H, peak_s=0.5):
    # unchanged logic; piecewise linear is already exact with 3 points
    z = np.array([0.0, float(peak_s)*L, L])
    q = np.array([0.0, H, 0.0])
    return z, q, {"shape": "triangular", "H": H, "peak_s": float(peak_s)}

def hf_gaussian_pair(L, H, w_frac=0.12, c1=0.3, c2=0.7, normalise_peak=False):
    z = np.linspace(0.0, L, 400)
    w = max(1e-6, w_frac*L)
    q = (np.exp(-0.5*((z-(c1*L))/w)**2) +
         np.exp(-0.5*((z-(c2*L))/w)**2))
    if normalise_peak and q.max() > 0:
        q = q / q.max()
    q = H * q
    return z, q, {"shape": "gaussian_pair", "H": H, "w_frac": float(w_frac),
                  "c1": float(c1), "c2": float(c2), "normalise_peak": bool(normalise_peak)}

def hf_sinusoidal(L, H, cycles=1, mode="offset",
                  Np_req=None, samples_per_cell=6, phase=0.0):
    """
    Sinus heat flux van 0 tot L, met 'cycles' periodes.
    modi:
      - "offset":     q = H * 0.5 * (1 + sin(...))  
      - "pure":       q = H * sin(...)               
      - "half-wave":  q = H * max(0, sin(...))       
    """
    import numpy as _np

    cycles = max(1, int(cycles))
    if Np_req and Np_req > 1:
        n = int(max(10, samples_per_cell * (Np_req - 1) + 1))
    else:
        n = 400

    z = _np.linspace(0.0, L, n)
    omega = 2.0 * _np.pi * cycles / max(L, 1e-12)
    phi   = float(phase)
    s     = _np.sin(omega * z + phi)

    if mode == "offset":
        q = H * 0.5 * (1.0 + s)
    elif mode == "pure":
        q = H * s
    elif mode == "half-wave":
        q = H * _np.maximum(0.0, s)
    else:
        raise ValueError(f"hf_sinusoidal: unknown mode '{mode}'")

    meta = {
        "shape": "sinusoidal", "H": float(H), "cycles": int(cycles),
        "mode": str(mode), "samples_per_cell": int(samples_per_cell),
        "Np_req": int(Np_req) if Np_req else None, "phase": phi
    }
    return z, q, meta


# Dispatcher
HF_BUILDERS = {
    "uniform": hf_uniform,
    "pulsed": hf_pulsed,
    "front_ramp": hf_front_ramp,
    "back_ramp": hf_back_ramp,
    "triangular": hf_triangular,
    "gaussian_pair": hf_gaussian_pair,
    "sinusoidal": hf_sinusoidal,   
}
def build_heat_profile(L, shape_name, params):
    if shape_name not in HF_BUILDERS:
        raise ValueError(f"Unknown heat profile shape: {shape_name}")
    return HF_BUILDERS[shape_name](L=L, **params)

# CRACKSIM DLL
def CRACKSIM_rates_DLL(gas):
    """Rates callback used by customPFR (runs inside worker)"""
    global fortlib
    if fortlib is None:
        raise RuntimeError("CRACKSIM not initialised in this worker")
    T = gas.T
    C_point = gas.concentrations.ctypes
    T_point = xt.byref(xt.c_double(T))
    status = xt.pointer(xt.c_int(0))
    R_point = (xt.c_double * gas.n_species)()
    _ = fortlib.NetRates_C(T_point, C_point, R_point, status)
    return np.ctypeslib.as_array(R_point)

def init_worker_cracksim(dll_path: str, mech_path: str, base_dir: Path,
                         scratch_root: Path, ready_q: Queue):
    """Intialiseer per CPU"""
    global fortlib, _gas_cache, REAC_MECH_PATH

    os.chdir(base_dir)

    w_scratch = Path(tempfile.mkdtemp(prefix=f"cracksim_w_{os.getpid()}_", dir=str(scratch_root)))
    os.environ['FORT45']  = str(w_scratch / "fort45.log")
    os.environ['FOR45']   = os.environ['FORT45']
    os.environ['FORT100'] = str(w_scratch / "fort100.log")
    os.environ['FOR100']  = os.environ['FORT100']

    ct.suppress_thermo_warnings()

    fort = xt.CDLL(dll_path)
    status = xt.pointer(xt.c_int(0))
    option = (xt.c_int * 20)()
    option[0] = 2
    _ = fort.Initialise_CRACKSIM(status, option)
    if status[0] != 1:
        ready_q.put("ERROR: CRACKSIM initialise failed in worker")
        return

    fortlib = fort
    REAC_MECH_PATH = mech_path
    _gas_cache = ct.Solution(REAC_MECH_PATH)

    ready_q.put("READY")

# Case runnen
def run_case(case):
    """
    case: dict with keys:
      id, seed, L, H_peak, shape, params, mdot, T_in, P_in, X_H2O, N_points
    Returns a pandas DataFrame with state, rates, properties along z, or None to skip.
    """
    try:
        L       = float(case["L"])
        H_peak  = float(case["H_peak"])
        shape   = str(case["shape"])
        params  = dict(case.get("params", {}))
        mdot    = float(case["mdot"])
        T_in    = float(case["T_in"])
        P_in    = float(case["P_in"])
        X_H2O   = float(case["X_H2O"])
        Np_req  = int(case.get("N_points", 200))

        # Heat flux profiles imposen
        if shape == "pulsed":
            params = {
                **params, "H": H_peak, "N": int(params.get("N", 10)),
                "Np_req": Np_req,
                "mode": params.get("mode", "uniform"),
                "jitter": float(params.get("jitter", 0.35)),
                "w_samples": int(params.get("w_samples", 2)),
                "snap_to_grid": bool(params.get("snap_to_grid", False)),
                "seed": int(case.get("seed", 0)),
            }
        elif shape == "sinusoidal":
            params = { **params, "H": H_peak, "Np_req": Np_req }
        else:
            params = { **params, "H": H_peak }
        z_cp, q_cp, meta = build_heat_profile(L, shape, params)
        hf_func = make_piecewise(z_cp, q_cp)

        # Reactor/solver setup 
        diam = 0.5 #merk op dat we dit hier d
        pfr = customPFR(                                                #Adiabaat ook getest, geeft ook resultaten die niet overeenkomen met directe uitrekening
            REAC_MECH_PATH, gas_id, mdot, diam, CRACKSIM_rates_DLL,
            energy_type='heat-flux-profile', heat_flux=hf_func,
            U=None, Tr=None, friction_factors=None
        )
        try:
            pfr.gas = _gas_cache
        except Exception:
            pass

        pfr.gas.TPY = T_in, P_in, {'C2H6': (1.0 - X_H2O), 'H2O': X_H2O}
        states, rates, _ = pfr.solve(L, Np_req)

        T_raw     = np.asarray(states.T, dtype=float).ravel()
        P_raw     = np.asarray(states.P, dtype=float).ravel()
        Y_raw     = np.asarray(states.Y, dtype=float)
        rates_raw = np.asarray(rates, dtype=float)

     
        z_raw = getattr(states, "z", None)
        if z_raw is None:
            z_raw = getattr(states, "grid", None)
        if z_raw is None:
            warnings.warn(f"run_case: CaseID={case.get('id','?')} missing axial grid; dropping case.")
            return None
        z_raw = np.asarray(z_raw, dtype=float).ravel()

        n_common = min(T_raw.size, P_raw.size, z_raw.size, Y_raw.shape[0], rates_raw.shape[0])
        if n_common <= 1:
            warnings.warn(f"run_case: CaseID={case.get('id','?')} invalid array sizes; dropping case.")
            return None
        T_raw     = T_raw[:n_common]
        P_raw     = P_raw[:n_common]
        z_raw     = z_raw[:n_common]
        Y_raw     = Y_raw[:n_common, :]
        rates_raw = rates_raw[:n_common, :]

        # Fysiek cappen van onze temperatuur: drop it like its hot 
        T_MAX_K = 1100.0 + 273.15 
        if np.any(T_raw > T_MAX_K):
            warnings.warn(f"run_case: CaseID={case.get('id','?')} exceeds {T_MAX_K-273.15}°C; dropping case.")
            return None

        # Extra safety om te vermijden dat onze case vroegtijdig stopt (eg crash) en dus onze databse zou poluten
        dz_nom = L / max(Np_req - 1, 1)
        if z_raw[-1] < L - 0.5 * dz_nom:
            warnings.warn(
                f"run_case: CaseID={case.get('id','?')} truncated at z={z_raw[-1]:.4f} m < L={L:.4f} m; dropping case."
            )
            return None

        # Wall heat flux imposed op ons grid
        q_wall = np.array([hf_func(zz) for zz in z_raw], dtype=float)

        # Energy Source term 
        S_energy_raw = getattr(states, "energy_source_term", None)
        if S_energy_raw is None:
            S_energy = np.full(n_common, np.nan, dtype=float)
        else:
            S_energy = np.asarray(S_energy_raw, dtype=float).ravel()
            if S_energy.size != n_common:
                S_energy = (S_energy[:n_common]
                            if S_energy.size > n_common else
                            np.pad(S_energy, (0, n_common - S_energy.size), constant_values=np.nan))

        # Hier controleer je de output die je wilt uitschrijven
        #dfY = pd.DataFrame(states.Y, columns = states.species_names)                           Het dataframe direct uit de states en rates opstellen heeft ook geen invloed op de uitkomst
        #dfY.rename(columns = {name: f"Y_{name}" for name in dfY.columns}, inplace = True)
        #dfR = pd.DataFrame(rates, columns = states.species_names)
        #dfR.rename(columns = {name: f"R_{name}" for name in dfR.columns}, inplace = True)
        dfY = pd.DataFrame(Y_raw, columns=states.species_names).rename(columns=lambda s: f"Y_{s}")
        dfR = pd.DataFrame(rates_raw, columns=states.species_names).rename(columns=lambda s: f"R_{s}")
        df  = pd.concat([dfY, dfR], axis=1)

        df['T [K]']              = T_raw
        df['P [Pa]']             = P_raw
        df['S Energy [J/s/m3]']  = S_energy
        df['Mass flow [kg/s]']   = mdot
        df['z [m]']              = z_raw
        df['Heat input [W/m^2]'] = q_wall

        # Geen simulatie zonder wat transport en thermo in de loop, dus die heb ik hier gegroeppeerd. 
        gas = pfr.gas
        n = n_common
        cp  = np.empty(n); cv = np.empty(n); rho = np.empty(n)
        mu  = np.empty(n); k  = np.empty(n); W   = np.empty(n)

        Y_cols_order = [f"Y_{s}" for s in states.species_names]
        Y_mat = df[Y_cols_order].to_numpy()

        for j in range(n):
            gas.TPY = float(T_raw[j]), float(P_raw[j]), Y_mat[j, :]
            cp[j]  = float(gas.cp_mass)
            cv[j]  = float(gas.cv_mass)
            rho[j] = float(gas.density)
            mu[j]  = _prop_val(gas, "viscosity")
            k[j]   = _prop_val(gas, "thermal_conductivity")
            W[j]   = float(gas.mean_molecular_weight)
            
        #Ook deze parameters worden uitgeschreven
        df["cp_mass [J/kg/K]"] = cp
        df["cv_mass [J/kg/K]"] = cv
        df["rho [kg/m3]"]      = rho
        df["mu [Pa·s]"]        = mu
        df["k [W/m/K]"]        = k
        df["W_mean [kg/kmol]"] = W

        # onze DMP verantwoordelijken aan de ugent zouden dit 'Metadata' noemen. 
        #Gewoon wat parameters om te weten welke case de data toe behoort (alsook gebruikt voor debugging, dus je kan hier wat paremeters verwijderen/commenten)
        df["CaseID"]          = int(case["id"])
        df["L [m]"]           = L
        df["H_peak [W/m^2]"]  = H_peak
        df["shape"]           = shape
        df["shape_params"]    = str(meta)
        df["mdot [kg/s]"]     = mdot
        df["T_in [K]"]        = T_in
        df["P_in [Pa]"]       = P_in
        df["X_H2O"]           = X_H2O
        if "Re_in" in case:
            df["Re_in [-]"]  = float(case["Re_in"])
        if "U_in" in case:
            df["U_in [m/s]"] = float(case["U_in"])

        return df

    except Exception as e:
        warnings.warn(f"run_case failed (CaseID={case.get('id','?')}): {e}")
        return None



# ---------------- Worker loop ----------------
def worker_loop(worker_id: int, task_q: JoinableQueue, ready_q: Queue,
                dll_path: str, mech_path: str, base_dir: Path, scratch_root: Path):
    init_worker_cracksim(dll_path, mech_path, base_dir, scratch_root, ready_q)

    out_parquet = f"worker_{worker_id}.parquet"
    try:
        if os.path.exists(out_parquet):
            os.remove(out_parquet)
    except OSError:
        pass

    writer = None
    schema = None

    try:
        while True:
            item = task_q.get()
            try:
                if item is None:
                    break

                case = item
                df = run_case(case)

                if df is not None and len(df) > 0:
                    table = pa.Table.from_pandas(df, preserve_index=False)

                    if writer is None:
                        schema = table.schema
                        writer = pq.ParquetWriter(out_parquet, schema, compression="snappy")
                    elif not table.schema.equals(schema, check_metadata=False):
                        try:
                            table = table.cast(schema)
                        except Exception:
                            schema = pa.unify_schemas([schema, table.schema])
                            writer.close()
                            writer = pq.ParquetWriter(out_parquet, schema, compression="snappy")
                            table = table.cast(schema)

                    writer.write_table(table)
                    print(f"[Worker {worker_id}] finished CaseID {case['id']}", flush=True)

            except Exception as e:
                warnings.warn(f"[Worker {worker_id}] error on CaseID={getattr(item,'get',lambda k, d=None: d)('id','?')}: {e}")
            finally:
                task_q.task_done()



    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        print(f"[Worker {worker_id}] exiting", flush=True)



if __name__ == "__main__":
    import psutil
    import subprocess
    import shutil

    starttime = time.time()
    print("Python CPU usage:", psutil.Process(os.getpid()).cpu_percent(interval=2), "%")

    base_dir = Path(r"C:\Users\Louis\OneDrive\Bureaublad\ugent\Master2\Thesis\Code_Turboreactor")
    dll_path = str((base_dir / "SA_CRACKSIM.dll").resolve())

    chem_inp      = base_dir / "chem.inp"
    transport_dat = base_dir / "transport_chemkin.DAT"
    chem_yaml     = base_dir / "chem.yaml"

    # run ck2yaml in de parent folder indien deze nog niet bestaat 
    if not chem_yaml.exists():
        with (base_dir / "C2KYAML_log.txt").open("w", encoding="utf-8") as log:
            subprocess.run(
                ["ck2yaml", f"--input={chem_inp}", f"--transport={transport_dat}", "--permissive"],
                stdout=log, stderr=subprocess.STDOUT, check=True, text=True
            )

    REAC_MECH_ABS = str(chem_yaml.resolve())

    # ---- parameter space (Re-based inlet) ----
    D_m            = 0.5                                  # m 
    Re_in_list     = [2.0e5]                       # Reynolds getallen om geen ridicule massflows te imposen (aka vermijd dat we supersonic gaan want Cantera heeft dat niet zo graag)
    T_in_list      = [550 + 273.15]                     # K
    P_in_list      = [2e5]                               # Pa
    X_H2O_list     = [0.30]                                 # /
    L_list         = [8.5]                           # m (om onze mean RTD te bepalen gaan we lengte variëren)
    H_peak_list    = [100000]                       # W/m^2 (non-negative only)
    N_points_list  = [200]                                 # axial resolution

    shapes =    [       ("uniform",       {})]#,
#       ("pulsed", {"N": 15, "mode": "uniform", "w_samples": 1, "snap_to_grid": True}),                       # minder variatie, Q input op 1 punt
#        ("pulsed", {"N": 15, "mode": "jitter", "jitter": 0.45, "w_samples": 2, "snap_to_grid": True}),        # meer variatie, Q input op 2 punten
#        ("pulsed", {"N": 15, "mode": "jitter", "jitter": 0.25, "w_samples": 1, "snap_to_grid": True}),
#        ("sinusoidal", {"cycles": 7, "mode": "offset", "samples_per_cell": 6}), 
#        ("front_ramp",    {"k": 3.0}),
#         ("back_ramp",     {"k": 3.0}),
#        ("triangular",    {}),
#        ("gaussian_pair", {"w_frac": 0.10}),
#    ]

    # Cantera gas structuur maken om onze massflow te berekenen
    gas_parent = ct.Solution(REAC_MECH_ABS)

    # Wat for loopjes om omnze cases te bouwen 
    cases = []
    cid = 1
    A_cross = 0.25*np.pi*D_m**2
    for T_in in T_in_list:
        for P_in in P_in_list:
            for X_H2O in X_H2O_list:
                gas_parent.TPY = T_in, P_in, {'C2H6': (1.0 - X_H2O), 'H2O': X_H2O}
                mu_in  = _prop_val(gas_parent, "viscosity")   # Pa·s
                rho_in = float(gas_parent.density)            # kg/m^3
                for Re_in in Re_in_list:
                    U_in  = Re_in * mu_in / (rho_in * D_m)    # m/s
                    mdot  = rho_in * U_in * A_cross           # kg/s
                    for L in L_list:
                        for H_peak in H_peak_list:
                            for shape_name, params in shapes:
                                for Np in N_points_list:
                                    #gas_parent.TPY = T_in, P_in, {'C2H6': (1.0 - X_H2O), 'H2O': X_H2O}         Deze 5 lijnen naar hier in de lus zetten heeft geen effect op de uitkomst, buiten de rekentijd verhogen
                                    #mu_in  = _prop_val(gas_parent, "viscosity")
                                    #rho_in = float(gas_parent.density)
                                    #U_in  = Re_in * mu_in / (rho_in * D_m)
                                    #mdot  = rho_in * U_in * A_cross

                                    seed = cid
                                    cases.append({
                                        "id": cid,
                                        "seed": seed,
                                        "L": L,
                                        "H_peak": H_peak,
                                        "shape": shape_name,
                                        "params": params,
                                        "mdot": mdot,         
                                        "T_in": T_in,
                                        "P_in": P_in,
                                        "X_H2O": X_H2O,
                                        "N_points": Np,
                                        "Re_in": Re_in,
                                        "U_in": U_in,
                                    })
                                    cid += 1
                                    

    print(f"Total cases: {len(cases)}", flush=True)

    # Extra Controle parameters
    CSV_Database = True
    N_threads    = 1#min(1000, os.cpu_count() or 4, len(cases))

    # Een temp file maken want ik had problemen om alles te callen op m'n OneDrive 
    scratch_root = Path(tempfile.mkdtemp(prefix="cracksim_scratch_root_"))

    # Dynamisch verdelen van de jobs. Voor nu zitten we wel met een seriële allocatie van de DLL aan de nodes. 
    # hier laat m'n programmeer kennis mij wat in de steek. Maar de allocatie time zal voor een grote database constructie verwaarloosbaar zijn
    # Dus zie ik het niet als een pertinent probleem op dit moment. 
    task_q   = JoinableQueue(maxsize=8 * N_threads)
    procs    = []
    ready_qs = []

    for wid in range(N_threads):
        rq = Queue(maxsize=1)
        p  = Process(
            target=worker_loop,
            args=(wid, task_q, rq, dll_path, REAC_MECH_ABS, base_dir, scratch_root),
            daemon=False
        )
        p.start()
        msg = rq.get()
        if isinstance(msg, str) and msg.startswith("ERROR"):
            raise RuntimeError(f"Worker {wid} failed to initialise: {msg}")
        print(f"[Worker {wid}] READY", flush=True)
        procs.append(p)
        ready_qs.append(rq)

    for case in cases:
        task_q.put(case)

    for _ in range(N_threads):
        task_q.put(None)

    task_q.join()
    for p in procs:
        p.join()
    try:
        task_q.close()
        task_q.join_thread()
    except Exception:
        pass

    print("[Parent] All workers joined. Starting merge...", flush=True)
    
    # Achter de berekeningen alles samenvoegen; Voor human readable output (debugging) kunnen we ook een .CSV uitschrijven
    final_parquet = "Database_Validation10.parquet"
    final_csv     = "Database_Validation10.csv" if CSV_Database else None

    try:
        if os.path.exists(final_parquet):
            os.remove(final_parquet)
    except OSError:
        pass
    if CSV_Database and final_csv and os.path.exists(final_csv):
        try:
            os.remove(final_csv)
        except OSError:
            pass

    writer        = None
    final_schema  = None
    wrote_any     = False
    csv_header_on = False
    any_parts = False
    for wid in range(N_threads):
        part = f"worker_{wid}.parquet"
        if not os.path.exists(part):
            continue
        any_parts = True
        print(f"[Parent] Merging {part} ...", flush=True)
        pf = pq.ParquetFile(part)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            if writer is None:
                final_schema = tbl.schema
                writer = pq.ParquetWriter(final_parquet, final_schema, compression="snappy")
            elif not tbl.schema.equals(final_schema, check_metadata=False):
                tbl = tbl.cast(final_schema)

            writer.write_table(tbl)
            wrote_any = True

            if CSV_Database:
                df_chunk = tbl.to_pandas(ignore_metadata=True)
                df_chunk.to_csv(
                    final_csv,
                    mode=('a' if csv_header_on else 'w'),
                    index=False,
                    header=(not csv_header_on)
                )
                csv_header_on = True

        # Alles wat opkuisen 
        try:
            os.remove(part)
        except OSError:
            pass

    if writer is not None:
        writer.close()

    if wrote_any:
        msg = f"Combined results written to {final_parquet}"
        if CSV_Database:
            msg += f" and {final_csv}"
        print(msg)
    else:
        print("No results produced.")

    # optional: clean scratch root (comment out to inspect logs)
    try:
        shutil.rmtree(scratch_root, ignore_errors=True)
    except Exception:
        pass

    endtime = time.time()
    print(f"\nAlle simulaties voltooid in {endtime - starttime:.2f} seconden")
