#!/usr/bin/env python
# coding: utf-8

# # Datbase Generation: Multiple cores

# In[35]:


import numpy as np
np.infty = np.inf
from multiprocessing import Process, JoinableQueue, Queue
import os

# (Optional but recommended) keep heavy libs from spawning their own threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import itertools
import scipy.integrate
import ctypes as xt
import os
from ideal_reactor_models import customESC_BM
from ideal_reactor_models import customPFR
import pandas as pd
import multiprocessing
from multiprocessing import Pool,Process, Manager, cpu_count
from concurrent.futures import ProcessPoolExecutor
import time
import pickle
import warnings
import openpyxl
import subprocess
from pathlib import Path

# In[3]:


def calc_conversion(gas,states, reactant):
    return 100.*(states.Y[0,gas.species_index(reactant)]-states.Y[:,gas.species_index(reactant)])/states.Y[0,gas.species_index(reactant)]
def calc_yield(gas,states, reactant, product):
    return 100.*(states.Y[:,gas.species_index(product)]-states.Y[0,gas.species_index(product)])/states.Y[0,gas.species_index(reactant)]
def calc_selectivity(gas,states, reactant, product):
    return 100.*(states.Y[:,gas.species_index(product)]-states.Y[0,gas.species_index(product)])/(states.Y[0,gas.species_index(reactant)]-states.Y[:,gas.species_index(reactant)])


# In[4]:


def CRACKSIM_rates_DLL(gas):
    # CRACKSIM_rates_C call the rates function in CRACKSIM.DLL
    # concentration bassis 

    # initialize temperature and concentrations
    T= gas.T     # [K]
    C_point=gas.concentrations #[mol/l]
    status = 0              

    # copy values to Ctypes to be used as arguements for the Fortran DLL
    C_point= gas.concentrations.ctypes
    T_point = xt.byref(xt.c_double(T)) 
    status = xt.pointer(xt.c_int(status))


    # initalize a ctype pointer to be used as storage for the calculated rates 
    R_point = (xt.c_double*gas.n_species)()     # mol/(s.L)
  
    _ = fortlib.NetRates_C(T_point,C_point,R_point,status)  # function call of CRACKSIM DLL
    
    # convert Ctype to python array
    rates=np.ctypeslib.as_array(R_point)
    rates=rates         #[mol/l/s]
    return rates
    #return rates


# In[5]:


# initialize kinetics 
fortlib = xt.CDLL(r"C:\Users\mbonheur\OneDrive - UGent\Documenten\GitHub\Thesis_Louis\SA_CRACKSIM.dll") #change this to the location where you stored the dll
status = 0
option = (xt.c_int*20)()
# please note that in python arrays possitions starts counting from 0 
            # option(1) : 
            #      0 use full network
            #      1 use reduced network based on compossition file 
            #      2 use only betanetwork
option[0]= 2
status = xt.pointer(xt.c_int(status)) # setup the pointer to the correct data structure
_ = fortlib.Initialise_CRACKSIM(status,option)            # call the function
status = status[0]
if status==1:
    print("Kinetics and Thermo were read succesfuly")
else:
    print("Errors while reading Kinetics and Thermo")


#Included to suppress the error warnings related to the NASA polynomes
ct.suppress_thermo_warnings()       # currently an issue with Nasapolynomials 
# convert the chem.inp file created by the DLL to yaml file

log_path = Path("C2KYAML_log.txt")
with log_path.open("w", encoding="utf-8") as log:
    subprocess.run(
        [
            "ck2yaml",
            "--input=chem.inp",
            "--transport=transport_chemkin.DAT",
            "--permissive",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,   # raise if ck2yaml fails
        text=True,
    )
    
subprocess.run(
    [
        "ck2yaml",
        "--input=chem.inp",
        "--transport=transport_chemkin.DAT",
        "--permissive"
    ],
    stdout=open("C2KYAML_log.txt", "w"),
    stderr=subprocess.STDOUT
)

# ## Verder met code

# In[12]:


reac_mech_DLL = 'chem.yaml'
gas_id = 'gas'
gas = ct.Solution(reac_mech_DLL)
print('Gas mechanism contains {} species and {} reactions'.format(gas.n_species, gas.n_reactions))


# In[13]:


# massflow = [80,85] #np.linspace(75,100,2) #80.4367 #kg/s
# diam = 0.5 #m
# T_in = [650+273,850+273] #np.linspace(550+273.15,1250+273.15,3)#650 + 273.15 #K
# P_in = [1.5e+5,2.0e+5] #np.linspace(1.0e+5,3.5e+5,3)#2.0e+5 #Pa
# Y_in = {'C2H6':0.77,'H2O':0.23} #massfractions inlet stream
# N_steps = 100 #Aantal integratiestappen
# H_input = [0,25e+6,50e+6] #np.linspace(0,50e+6,3) #50000000 #W/m2, discrete heat input
# NS = [5,15,30] #np.linspace(5,30,3,dtype=int) #20 # Aantal heat inputs, (aantal passages door de blades vd turboreactor)
# H2O_fraction = [0.3,0.5] #np.linspace(0.3,0.5,3)


# In[14]:


# cases = list(itertools.product(H_input, NS, massflow, T_in, P_in, H2O_fraction))


# In[15]:


def make_hf(z_prof, hf_prof):
    def hf_func(z):
        return np.interp(z, z_prof, hf_prof)
    return hf_func


# In[16]:


# TEST NIEUWE CONSTRUCTIE HEAT PROFILE

# z_prof = np.linspace(0,10.5,100).tolist() #Axiale afstand, m
# hf_prof = [] #Heatflux, W/m2
# for i in range(len(z_prof)):
#     if i%(len(z_prof)//10) == 0:
#         hf_prof.append(20e6) #W/m2
#     if i%(len(z_prof)//10) != 0:
#         hf_prof.append(0)
# length = z_prof[-1] - z_prof[0] #m
# hf_func = make_hf(z_prof, hf_prof)
# hf = ct.Func1(hf_func)   # ct.Func1(lambda z: np.interp(z,z_prof,hf_prof))
# y = []
# x = []
# for i in z_prof:
#     y.append(hf(i))
#     x.append(i)
# #y.append(0)
# plt.figure()
# plt.plot(x,y)
# plt.show()


# In[37]:


print(multiprocessing.cpu_count())
print(openpyxl.__version__)


# In[39]:


def run_case(argument):
    # argument: tuple (H, N, M, T, P, X)
    try:
        H, N, M, T, P, X = argument
    
        z_prof = np.linspace(0,10.5,100).tolist() #Axiale afstand, m
        hf_prof = [] #Heatflux, W/m2
        for i in range(len(z_prof)):
            if i%(len(z_prof)//N) == 0:
                hf_prof.append(H) #W/m2
            if i%(len(z_prof)//N) != 0:
                hf_prof.append(0)
        length = z_prof[-1] - z_prof[0] #m
        hf_func = make_hf(z_prof, hf_prof)
        hf = ct.Func1(hf_func)   # ct.Func1(lambda z: np.interp(z,z_prof,hf_prof))
        y = []
        for i in z_prof:
            y.append(hf(i))
        y.append(0)
        diam = 0.5
        PFR_calc = customPFR(reac_mech_DLL,gas_id,M,diam,CRACKSIM_rates_DLL,energy_type = 'heat-flux-profile',heat_flux = hf, U = None, Tr = None, friction_factors = None)
        PFR_calc.gas.TPY = T, P, {'C2H6':(1.0 - X), 'H2O':X}
        states_PFR_calc,rates_PFR_calc,enth_PFR_calc = PFR_calc.solve(length,100)
    
        df_Y = pd.DataFrame(states_PFR_calc.Y, columns = states_PFR_calc.species_names)
        df_Y.rename(columns = {name: f"Y_{name}" for name in df_Y.columns},inplace = True)
        df_rr = pd.DataFrame(rates_PFR_calc,columns = states_PFR_calc.species_names)
        df_rr.rename(columns = {name: f"R_{name}" for name in df_rr.columns}, inplace = True)
        df_species = pd.concat([df_Y, df_rr], axis = 1)
        df_species['T [K]'] = states_PFR_calc.T
        df_species['P [Pa]'] = states_PFR_calc.P
        df_species['S Energy [J/s/m3]'] = states_PFR_calc.energy_source_term
        df_species['Mass flow [kg/s]'] = states_PFR_calc.mass_flow
        df_species['Heat input [W/m^2]'] = y
    
        return df_species
    except Exception as e:
        warnings.warn(f"run_case failed: {e}")
        return None


# In[41]:


import psutil, os
print("Python CPU usage:", psutil.Process(os.getpid()).cpu_percent(interval=2), "%")


# In[47]:


# ----- Dynamic workers using a JoinableQueue (no duplicates, clear logging) -----
# ----- Dynamic workers using a JoinableQueue (Parquet, streaming) -----
from multiprocessing import Process, JoinableQueue, Queue
import os

def worker_dynamic(worker_id: int, task_q: JoinableQueue, ready_q: Queue):
    print(f"[Worker {worker_id} | pid {os.getpid()}] online", flush=True)
    ready_q.put("READY")  # signal parent we’re up

    writer = None
    schema = None
    out_parquet = f"worker_{worker_id}.parquet"
    # if a stale file exists, remove it so we start clean
    try:
        if os.path.exists(out_parquet):
            os.remove(out_parquet)
    except OSError:
        pass

    try:
        while True:
            item = task_q.get()  # blocks
            if item is None:
                task_q.task_done()
                break

            case_id, H, N, M, T, P, X = item
            try:
                df = run_case((H, N, M, T, P, X))
            except Exception as e:
                print(f"[Worker {worker_id}] CaseID {case_id} crashed: {e}", flush=True)
                df = None

            if df is not None:
                # stamp metadata so the final table is fully self-describing
                df["CaseID"] = case_id
                df["H_input [W/m^2]"] = H
                df["N_pulses"] = N
                df["mdot [kg/s]"] = M
                df["T_in [K]"] = T
                df["P_in [Pa]"] = P
                df["X_H2O"] = X

                # Convert to Arrow and stream to a Parquet row group
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out_parquet, schema, compression="snappy")
                else:
                    # enforce consistent column order/types across cases
                    if not table.schema.equals(schema, check_metadata=False):
                        table = table.cast(schema)

                writer.write_table(table)
                print(f"[Worker {worker_id}] finished CaseID {case_id}", flush=True)

            task_q.task_done()

    finally:
        if writer is not None:
            writer.close()
        print(f"[Worker {worker_id}] exiting", flush=True)


# ----- Main script (unchanged scheduling; Parquet combine) -----
if __name__ == "__main__":
    starttime = time.time()

    # --- Build cases (unchanged) ---
    massflow = [80]
    T_in = [650+273]
    P_in = [1.5e5]
    H_input = [0, 25e6]
    NS = [5, 15, 30]
    H2O_fraction = [0.3, 0.5]

    raw_cases = [(H, N, M, T, P, X)
                 for H in H_input
                 for N in NS
                 for M in massflow
                 for T in T_in
                 for P in P_in
                 for X in H2O_fraction]
    cases = [(case_id, *c) for case_id, c in enumerate(raw_cases, start=1)]
    print(f"Total cases: {len(cases)}", flush=True)

    # --- Dynamic scheduling (serial start stays to avoid DLL logfile races) ---
    N_threads = min(12, os.cpu_count() or 4)
    task_q = JoinableQueue(maxsize=8 * N_threads)

    procs, ready_queues = [], []
    for i in range(N_threads):
        rq = Queue(maxsize=1)
        p = Process(target=worker_dynamic, args=(i, task_q, rq), daemon=False)
        p.start()
        rq.get()  # wait until the worker is fully started
        procs.append(p)
        ready_queues.append(rq)

    # feed all work
    for item in cases:
        task_q.put(item)

    # one sentinel per worker
    for _ in range(N_threads):
        task_q.put(None)

    task_q.join()
    for p in procs:
        p.join()

    # --- Concatenate all worker parquet files into a SINGLE parquet (streaming) ---
    final_parquet = "Database_FINAL.parquet"
    # remove old final if present
    try:
        if os.path.exists(final_parquet):
            os.remove(final_parquet)
    except OSError:
        pass

    writer = None
    final_schema = None
    any_written = False

    for i in range(N_threads):
        part = f"worker_{i}.parquet"
        if not os.path.exists(part):
            continue

        pf = pq.ParquetFile(part)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            if writer is None:
                final_schema = tbl.schema
                writer = pq.ParquetWriter(final_parquet, final_schema, compression="snappy")
            else:
                if not tbl.schema.equals(final_schema, check_metadata=False):
                    tbl = tbl.cast(final_schema)
            writer.write_table(tbl)
            any_written = True

        # cleanup the worker file once copied
        try:
            os.remove(part)
        except OSError:
            pass

    if writer is not None:
        writer.close()

    if not any_written:
        print("No results were produced; final parquet not created.")
    else:
        print(f"Combined results written to {final_parquet}")

    endtime = time.time()
    print(f"\nAlle simulaties voltooid in {endtime - starttime:.2f} seconden")




# In[36]:


#run_case(cases[-1])


# In[ ]:


#status = xt.pointer(xt.c_int(status)) # setup the pointer to the correct data structure
#_ = fortlib.terminate_CRACKSIM(status)            # call the function
#status = status[0]
#if status==1:
#  print("Memory released succesfully")
#else:
#  print("Errors while deallocating arrays")

#del fortlib

