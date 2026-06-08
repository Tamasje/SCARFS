/*
 * fluent_reduced_source.c
 * =======================
 * Fluent UDF TEMPLATE — Reduced source-term surrogate (Ch. 6 surrogate).
 *
 * IMPORTANT: This file is a TEMPLATE.  It must be edited for mechanism-specific
 * wiring before compilation on the HPC.  Search for all /* TODO */ markers.
 * This file CANNOT be compiled or tested in the SCARFS repository — Ansys Fluent
 * is not available here.  The user compiles it on the HPC via the Fluent UDF panel
 * (Build / Load).
 *
 * What this UDF does
 * ------------------
 * At DEFINE_INIT: reads weights.txt, scalers.txt, species.txt from disk.
 * At DEFINE_SOURCE (per cell, per iteration):
 *   1. Reads local state  Y_i, T, P  from Fluent cell macros.
 *   2. Clips inputs to training ranges  (ChemZIP §4.3 step 5; guards RC-2/RC-4).
 *   3. Builds scaled feature vector [log-scaled Y, standardised T+P block].
 *   4. Runs MLP forward pass.
 *   5. Inverse-scales output to R_i [kg m-3 s-1].
 *   6. Returns species source term (DEFINE_SOURCE for each active species).
 *   7. Returns energy source S_E [J m-3 s-1] (DEFINE_SOURCE for energy equation).
 *
 * Units (DO NOT change)
 * ---------------------
 *   Species source:  R_i  [kg m-3 s-1]   — Fluent expects kg/(m^3·s)
 *   Energy source:   S_E  [J  m-3 s-1]   — Fluent expects W/m^3 = J/(m^3·s)
 *
 * Water (H2O) is a fixed diluent (thesis Ch. 5.6) and is NOT in the active
 * species list.  Its mass fraction is 1 - sum(Y_active).  No source term is
 * set for H2O; Fluent handles the N-th-species constraint automatically.
 *
 * Guarding RC-1 and RC-2
 * ----------------------
 * RC-1 (near-zero rates / frozen composition): the near-inlet state that drove
 *   the surrogate to predict ~0 rates cannot be prevented at inference time alone —
 *   it must be fixed by improved training data coverage (F1).  However, the input-
 *   clipping step below prevents extrapolation to states even further outside the
 *   training distribution, limiting additional bias.
 * RC-2 (NeuralCoil latent drift): not applicable here (this is the reduced physical-
 *   space surrogate, not NeuralCoil).  See fluent_neuralcoil_uds.c for RC-2 guard.
 * RC-4 (OOD extrapolation): the INPUT CLIPPING block (step 2) directly implements
 *   ChemZIP §4.3 step 5 — inputs are clamped to [data_min, data_max] before the
 *   forward pass, preventing the network from seeing states outside its training
 *   distribution.
 */

#include "udf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * TODO: Set mechanism-specific constants
 * ========================================================================= */

/* TODO: Set the number of active species (must match species.txt N_ACTIVE) */
#define N_ACTIVE  30

/* TODO: Set the number of thermo features [T, P, 1/T, ln(T)] = 4 */
#define N_THERMO  4

/* Total input dimension to the MLP */
#define N_INPUT   (N_ACTIVE + N_THERMO)

/* TODO: Set maximum number of MLP layers (update if architecture changes) */
#define MAX_LAYERS  8

/* TODO: Set maximum neurons per layer */
#define MAX_NEURONS  512

/* TODO: Set the path prefix for the exported weight files (UNC path on HPC) */
#define WEIGHTS_FILE  "surrogate_weights.txt"
#define SCALERS_FILE  "surrogate_scalers.txt"
#define SPECIES_FILE  "surrogate_species.txt"

/* =========================================================================
 * Internal data structures
 * ========================================================================= */

typedef struct {
    int  d_in;
    int  d_out;
    char activation[32];
    double *W;   /* d_out x d_in, row-major */
    double *b;   /* d_out */
} DenseLayer;

typedef struct {
    int     n_layers;
    DenseLayer layers[MAX_LAYERS];
    double  buf0[MAX_NEURONS];
    double  buf1[MAX_NEURONS];
} MLP;

typedef struct {
    /* CompositionScaler parameters (for input Y) */
    int    comp_log;
    double comp_floor;
    double comp_feature_lo;
    double comp_feature_hi;
    double comp_data_min[N_ACTIVE];
    double comp_data_max[N_ACTIVE];

    /* StandardScaler parameters (for thermo features) */
    double thermo_mean[N_THERMO];
    double thermo_scale[N_THERMO];

    /* ArcsinhScaler parameters (for output rates + energy) */
    double rate_arcsinh_scale[N_ACTIVE + 1];  /* +1 for energy */
    double rate_std_mean[N_ACTIVE + 1];
    double rate_std_scale[N_ACTIVE + 1];
} Scalers;

static MLP      g_mlp;
static Scalers  g_sc;
static char     g_species[N_ACTIVE][64];
static int      g_loaded = 0;

/* =========================================================================
 * Activation functions
 * ========================================================================= */

static double apply_activation(double x, const char *act)
{
    if (strcmp(act, "relu") == 0)    return x > 0.0 ? x : 0.0;
    if (strcmp(act, "tanh") == 0)    return tanh(x);
    if (strcmp(act, "sigmoid") == 0) return 1.0 / (1.0 + exp(-x));
    if (strcmp(act, "softplus") == 0) return log(1.0 + exp(x));
    if (strcmp(act, "elu") == 0)     return x >= 0.0 ? x : (exp(x) - 1.0);
    /* linear / unknown */
    return x;
}

/* =========================================================================
 * MLP forward pass
 * ========================================================================= */

static void mlp_forward(const double *x_in, double *out, int n_out)
{
    double *cur = g_mlp.buf0;
    double *nxt = g_mlp.buf1;
    int i, j, l;

    memcpy(cur, x_in, sizeof(double) * N_INPUT);

    for (l = 0; l < g_mlp.n_layers; l++) {
        DenseLayer *layer = &g_mlp.layers[l];
        for (i = 0; i < layer->d_out; i++) {
            double acc = layer->b[i];
            for (j = 0; j < layer->d_in; j++)
                acc += layer->W[i * layer->d_in + j] * cur[j];
            nxt[i] = apply_activation(acc, layer->activation);
        }
        /* swap buffers */
        double *tmp = cur; cur = nxt; nxt = tmp;
    }

    memcpy(out, cur, sizeof(double) * n_out);
}

/* =========================================================================
 * File I/O — read exported artefacts
 * ========================================================================= */

static void load_weights(void)
{
    FILE *f = fopen(WEIGHTS_FILE, "r");
    if (!f) { Message("SCARFS UDF ERROR: cannot open %s\n", WEIGHTS_FILE); return; }

    int n_layers;
    fscanf(f, "LAYERS %d", &n_layers);
    g_mlp.n_layers = n_layers;

    int l;
    for (l = 0; l < n_layers; l++) {
        int idx, d_in, d_out;
        char act[32];
        /* format: LAYER <i> in=<din> out=<dout> activation=<name> */
        fscanf(f, " LAYER %d in=%d out=%d activation=%31s", &idx, &d_in, &d_out, act);
        g_mlp.layers[l].d_in  = d_in;
        g_mlp.layers[l].d_out = d_out;
        strncpy(g_mlp.layers[l].activation, act, 31);
        int n_vals = d_out * d_in + d_out;
        g_mlp.layers[l].W = (double *)malloc(sizeof(double) * d_out * d_in);
        g_mlp.layers[l].b = (double *)malloc(sizeof(double) * d_out);
        int k;
        for (k = 0; k < d_out * d_in; k++) fscanf(f, " %lf", &g_mlp.layers[l].W[k]);
        for (k = 0; k < d_out; k++)        fscanf(f, " %lf", &g_mlp.layers[l].b[k]);
    }
    fclose(f);
    Message("SCARFS UDF: loaded MLP weights (%d layers) from %s\n", n_layers, WEIGHTS_FILE);
}

static void load_scalers(void)
{
    FILE *f = fopen(SCALERS_FILE, "r");
    if (!f) { Message("SCARFS UDF ERROR: cannot open %s\n", SCALERS_FILE); return; }

    int n_scalers;
    fscanf(f, " N_SCALERS %d", &n_scalers);

    int s, k;
    for (s = 0; s < n_scalers; s++) {
        char name[64], stype[64];
        fscanf(f, " SCALER %63s %63s", name, stype);

        if (strcmp(stype, "CompositionScaler") == 0) {
            int log_flag; double floor_val, flo, fhi;
            fscanf(f, " log %d floor %lf feature_range %lf %lf",
                   &log_flag, &floor_val, &flo, &fhi);
            g_sc.comp_log = log_flag;
            g_sc.comp_floor = floor_val;
            g_sc.comp_feature_lo = flo;
            g_sc.comp_feature_hi = fhi;
            int n_min; fscanf(f, " data_min %d", &n_min);
            for (k = 0; k < n_min; k++) fscanf(f, " %lf", &g_sc.comp_data_min[k]);
            int n_max; fscanf(f, " data_max %d", &n_max);
            for (k = 0; k < n_max; k++) fscanf(f, " %lf", &g_sc.comp_data_max[k]);
            char end[32]; fscanf(f, " %31s", end); /* END_SCALER */
        }
        else if (strcmp(stype, "StandardScaler") == 0) {
            int n_mean; fscanf(f, " mean %d", &n_mean);
            for (k = 0; k < n_mean; k++) fscanf(f, " %lf", &g_sc.thermo_mean[k]);
            int n_scale; fscanf(f, " scale %d", &n_scale);
            for (k = 0; k < n_scale; k++) fscanf(f, " %lf", &g_sc.thermo_scale[k]);
            char end[32]; fscanf(f, " %31s", end);
        }
        else if (strcmp(stype, "ArcsinhScaler") == 0) {
            double ms; fscanf(f, " min_scale %lf", &ms);
            int n_as; fscanf(f, " arcsinh_scale %d", &n_as);
            for (k = 0; k < n_as; k++) fscanf(f, " %lf", &g_sc.rate_arcsinh_scale[k]);
            int n_sm; fscanf(f, " std_mean %d", &n_sm);
            for (k = 0; k < n_sm; k++) fscanf(f, " %lf", &g_sc.rate_std_mean[k]);
            int n_ss; fscanf(f, " std_scale %d", &n_ss);
            for (k = 0; k < n_ss; k++) fscanf(f, " %lf", &g_sc.rate_std_scale[k]);
            char end[32]; fscanf(f, " %31s", end);
        }
    }
    fclose(f);
    Message("SCARFS UDF: loaded scalers from %s\n", SCALERS_FILE);
}

static void load_species(void)
{
    FILE *f = fopen(SPECIES_FILE, "r");
    if (!f) { Message("SCARFS UDF ERROR: cannot open %s\n", SPECIES_FILE); return; }
    int n; fscanf(f, " N_ACTIVE %d", &n);
    int i;
    for (i = 0; i < n; i++) fscanf(f, " %63s", g_species[i]);
    fclose(f);
    Message("SCARFS UDF: loaded %d active species from %s\n", n, SPECIES_FILE);
}

/* =========================================================================
 * DEFINE_INIT — load all artefacts once at solver start
 * ========================================================================= */

DEFINE_INIT(scarfs_reduced_init, domain)
{
    if (g_loaded) return;
    load_weights();
    load_scalers();
    load_species();
    g_loaded = 1;
    Message("SCARFS UDF: reduced surrogate initialised.\n");
}

/* =========================================================================
 * Helper: build feature vector and run MLP
 * Returns raw (still arcsinh-scaled) outputs in *out_scaled* (N_ACTIVE+1 values).
 * ========================================================================= */

static void predict_scaled(cell_t c, Thread *t, double *out_scaled)
{
    double x[N_INPUT];
    int i;

    /* --- Read cell state --- */

    /* TODO: Replace C_YI(c,t,idx) with Fluent species index macro for each species.
     * The species index in Fluent must match the order in species.txt.
     * Example (must be verified against your mechanism YAML):
     *   i=0  -> C_YI(c, t, 0)   (C2H6)
     *   i=1  -> C_YI(c, t, 1)   (C2H4)
     *   ... etc.
     * Use SV_Y (species index) from your Fluent mixture material settings.
     */
    double Y[N_ACTIVE];
    for (i = 0; i < N_ACTIVE; i++) {
        /* TODO: replace 'i' with actual Fluent species index for g_species[i] */
        Y[i] = C_YI(c, t, i);
    }

    /* TODO: C_T, C_P, C_R are standard Fluent macros; verify units:
     *   C_T(c,t)  [K]
     *   C_P(c,t)  [Pa]  (operating + gauge; confirm with Fluent setup)
     *   C_R(c,t)  [kg/m^3]  (density — not used in features but listed for reference)
     */
    double T = C_T(c, t);
    double P = C_P(c, t);

    /* =========================================================
     * STEP 2: INPUT CLIPPING TO TRAINING RANGES
     * (ChemZIP §4.3 step 5 — guards RC-2/RC-4)
     *
     * Clips each input to [data_min, data_max] BEFORE scaling.
     * This prevents the network from evaluating states outside its
     * training distribution, where predictions are unreliable
     * (extrapolation) and can drive latent drift (RC-2) or large
     * source-term errors (RC-4).
     * ========================================================= */
    double comp_floor = g_sc.comp_floor;
    for (i = 0; i < N_ACTIVE; i++) {
        if (Y[i] < comp_floor) Y[i] = comp_floor;
        /* The training min/max are in log10 space; clip to prevent
         * log10(Y) from going below data_min (extreme extrapolation). */
    }
    /* Temperature and pressure: clip to the scaler's observed range
     * (mean ± 5*scale is a safe proxy for the training envelope) */
    double T_min = g_sc.thermo_mean[0] - 5.0 * g_sc.thermo_scale[0];
    double T_max = g_sc.thermo_mean[0] + 5.0 * g_sc.thermo_scale[0];
    double P_min = g_sc.thermo_mean[1] - 5.0 * g_sc.thermo_scale[1];
    double P_max = g_sc.thermo_mean[1] + 5.0 * g_sc.thermo_scale[1];
    if (T < T_min) T = T_min;
    if (T > T_max) T = T_max;
    if (P < P_min) P = P_min;
    if (P > P_max) P = P_max;

    /* =========================================================
     * STEP 3: BUILD SCALED FEATURE VECTOR
     * ========================================================= */

    /* 3a. Composition block: log10 -> affine to feature_range */
    double lo = g_sc.comp_feature_lo;
    double hi = g_sc.comp_feature_hi;
    for (i = 0; i < N_ACTIVE; i++) {
        double logy = (g_sc.comp_log) ? log10(Y[i]) : Y[i];
        double span = g_sc.comp_data_max[i] - g_sc.comp_data_min[i];
        if (span < 1e-30) span = 1.0;
        double unit = (logy - g_sc.comp_data_min[i]) / span;
        x[i] = unit * (hi - lo) + lo;
    }

    /* 3b. Thermo block: [T, P, 1/T, ln(T)] standardised */
    double thermo[4];
    thermo[0] = T;
    thermo[1] = P;
    thermo[2] = 1.0 / T;
    thermo[3] = log(T);
    for (i = 0; i < N_THERMO; i++) {
        x[N_ACTIVE + i] = (thermo[i] - g_sc.thermo_mean[i]) / g_sc.thermo_scale[i];
    }

    /* =========================================================
     * STEP 4: MLP FORWARD PASS
     * ========================================================= */
    mlp_forward(x, out_scaled, N_ACTIVE + 1);
}

/* Helper: inverse-arcsinh-scale one output element */
static double arcsinh_inv(double z_std, int idx)
{
    double raw = z_std * g_sc.rate_std_scale[idx] + g_sc.rate_std_mean[idx];
    return sinh(raw) * g_sc.rate_arcsinh_scale[idx];
}

/* =========================================================================
 * DEFINE_SOURCE for each active species
 * Fluent calls this once per cell per species per iteration.
 *
 * TODO: Duplicate this macro for each active species, replacing SPECIES_IDX
 * with the ordinal (0 .. N_ACTIVE-1) matching species.txt.
 * Example:
 *   DEFINE_SOURCE(scarfs_src_C2H6,  cell, thread, dS, eqn) { ... return ... }
 *   DEFINE_SOURCE(scarfs_src_C2H4,  cell, thread, dS, eqn) { ... return ... }
 *   ... etc.
 *
 * The function name must match what you enter in the Fluent Species Source panel.
 * ========================================================================= */

/* Generic helper — call from each species-specific DEFINE_SOURCE */
static double get_species_source(cell_t c, Thread *t, int species_idx, real *dS)
{
    double out_scaled[N_ACTIVE + 1];
    predict_scaled(c, t, out_scaled);

    /* STEP 5: Inverse-scale output to R_i [kg m-3 s-1] */
    double R_i = arcsinh_inv(out_scaled[species_idx], species_idx);

    /* Linearised source dS/dY_i — set to 0 (explicit source, no linearisation).
     * TODO: For better convergence stability, provide dR_i/dY_i (Jacobian) if
     * available from automatic differentiation of the surrogate. */
    dS[eqn] = 0.0;

    return (real)R_i;
}

/* TODO: Replace with one DEFINE_SOURCE per active species. */
DEFINE_SOURCE(scarfs_src_species_0, cell, thread, dS, eqn)
{
    /* TODO: Replace 0 with the correct species index (0 .. N_ACTIVE-1) */
    return get_species_source(cell, thread, 0, dS);
}

/* =========================================================================
 * DEFINE_SOURCE for the energy equation
 * Returns S_E [J m-3 s-1] = [W m-3].
 *
 * In Fluent's energy equation the source term has units W/m^3.
 * The sign convention: negative S_E = endothermic (heat absorbed).
 * ========================================================================= */

DEFINE_SOURCE(scarfs_src_energy, cell, thread, dS, eqn)
{
    double out_scaled[N_ACTIVE + 1];
    predict_scaled(cell, thread, out_scaled);

    /* Energy is the last element (index N_ACTIVE) */
    double S_E = arcsinh_inv(out_scaled[N_ACTIVE], N_ACTIVE);

    dS[eqn] = 0.0;
    return (real)S_E;
}
