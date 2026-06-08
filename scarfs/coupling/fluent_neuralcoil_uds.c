/*
 * fluent_neuralcoil_uds.c
 * =======================
 * Fluent UDF TEMPLATE — NeuralCoil latent-scalar (UDS) surrogate (Ch. 5 surrogate).
 *
 * IMPORTANT: This file is a TEMPLATE.  It must be edited for mechanism-specific
 * wiring before compilation on the HPC.  Search for all /* TODO */ markers.
 * This file CANNOT be compiled or tested in the SCARFS repository — Ansys Fluent
 * is not available here.  The user compiles it on the HPC via the Fluent UDF panel
 * (Build / Load).
 *
 * Architecture (ChemZIP-faithful, thesis Ch. 5)
 * ---------------------------------------------
 *   Encoder E:  Y_sc (N_SC dry species) -> Z (k latent scalars)
 *   Rate net:   (Z, T, P) -> omega_Z (latent source terms [s-1])
 *   Decoder D:  Z -> Y_sc (for manifold projection only, not for rate evaluation)
 *   Energy net: (Z, T, P) -> S_E [J m-3 s-1]
 *
 * Fluent setup
 * ------------
 *   1. Define k User-Defined Scalars (UDS) in the Fluent UDS panel.
 *      They hold the transported latent variables Z_0 ... Z_{k-1}.
 *   2. Hook DEFINE_ADJUST -> scarfs_nc_adjust
 *   3. Hook DEFINE_SOURCE for each UDS -> scarfs_nc_uds_source_<j>
 *   4. Hook DEFINE_SOURCE for the energy equation -> scarfs_nc_energy_source
 *
 * Units
 * -----
 *   UDS source terms:  [kg m-3 s-1]  (Fluent UDS transport equation has this default)
 *   Energy source:     [J  m-3 s-1]  = [W m-3]
 *
 * Guarding RC-2: Manifold Projection (F2 fix)
 * -------------------------------------------
 * The key NeuralCoil stability failure (DIAGNOSIS.md RC-2) is that the transported
 * latent Z drifts off the encoder manifold Z = E·Y as Fluent integrates the UDS
 * transport equations.  Off-manifold residuals grow self-amplifyingly (ε: 7 -> 2777,
 * thesis §5.5.2.3-4) and drive the composition to zero.
 *
 * Fix F2: at the START of each DEFINE_ADJUST call (once per iteration, before any
 * source-term evaluation) we project Z back onto the manifold:
 *
 *     Z_proj = E · D(Z)
 *
 * where E is the linear encoder and D is the decoder network.  The projected Z
 * then replaces the UDS value in cell memory.  This is marked with a prominent
 * MANIFOLD PROJECTION comment below.
 *
 * The rate network then evaluates omega_Z = f(Z_proj, T, P) — operating on the
 * corrected latent state — preventing exponential drift.
 */

#include "udf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * TODO: Set NeuralCoil-specific constants
 * ========================================================================= */

/* TODO: Number of latent dimensions (k=6 in thesis Ch. 5) */
#define K_LATENT  6

/* TODO: Number of transported dry species in the encoder input (74 in thesis) */
#define N_SC  74

/* TODO: Number of thermo features for the rate net [T, P, 1/T, ln(T)] */
#define N_THERMO  4

/* Input to rate / energy net: latent + thermo */
#define N_RATE_INPUT  (K_LATENT + N_THERMO)

/* TODO: Max layers / neurons (adjust to match exported architecture) */
#define MAX_LAYERS   8
#define MAX_NEURONS  512

/* File paths (use absolute paths on HPC) */
#define ENCODER_WEIGHTS_FILE  "nc_encoder_weights.txt"
#define DECODER_WEIGHTS_FILE  "nc_decoder_weights.txt"
#define RATE_WEIGHTS_FILE     "nc_rate_weights.txt"
#define ENERGY_WEIGHTS_FILE   "nc_energy_weights.txt"
#define SCALERS_FILE          "nc_scalers.txt"
#define SPECIES_FILE          "nc_species.txt"

/* =========================================================================
 * Internal data structures (same layout as fluent_reduced_source.c)
 * ========================================================================= */

typedef struct {
    int  d_in;
    int  d_out;
    char activation[32];
    double *W;
    double *b;
} DenseLayer;

typedef struct {
    int       n_layers;
    DenseLayer layers[MAX_LAYERS];
    double    buf0[MAX_NEURONS];
    double    buf1[MAX_NEURONS];
} MLP;

static MLP g_encoder;    /* linear encoder E: Y_sc -> Z  (activation=linear) */
static MLP g_decoder;    /* decoder D: Z -> Y_sc_hat */
static MLP g_rate_net;   /* (Z, T, P) -> omega_Z */
static MLP g_energy_net; /* (Z, T, P) -> S_E */

/* Scaler parameters for composition (Y input to encoder) and output (omega_Z, S_E) */
typedef struct {
    /* CompositionScaler for encoder input Y_sc */
    int    comp_log;
    double comp_floor;
    double comp_feature_lo, comp_feature_hi;
    double comp_data_min[N_SC];
    double comp_data_max[N_SC];

    /* StandardScaler for thermo [T, P, 1/T, ln(T)] */
    double thermo_mean[N_THERMO];
    double thermo_scale[N_THERMO];

    /* ArcsinhScaler for latent sources omega_Z */
    double omegaZ_arcsinh_scale[K_LATENT];
    double omegaZ_std_mean[K_LATENT];
    double omegaZ_std_scale[K_LATENT];

    /* ArcsinhScaler for energy S_E */
    double energy_arcsinh_scale;
    double energy_std_mean;
    double energy_std_scale;
} NCScalers;

static NCScalers g_sc;
static char      g_species[N_SC][64];
static int       g_loaded = 0;

/* Per-cell projected Z storage (Fluent provides per-cell UDS; projection is in-place) */

/* =========================================================================
 * Activation and forward pass (same as reduced template)
 * ========================================================================= */

static double apply_activation(double x, const char *act)
{
    if (strcmp(act, "relu") == 0)     return x > 0.0 ? x : 0.0;
    if (strcmp(act, "tanh") == 0)     return tanh(x);
    if (strcmp(act, "sigmoid") == 0)  return 1.0 / (1.0 + exp(-x));
    if (strcmp(act, "softplus") == 0) return log(1.0 + exp(x));
    if (strcmp(act, "elu") == 0)      return x >= 0.0 ? x : (exp(x) - 1.0);
    return x; /* linear */
}

static void mlp_forward(MLP *net, const double *x_in, int n_in, double *x_out, int n_out)
{
    int i, j, l;
    double *cur = net->buf0;
    double *nxt = net->buf1;
    memcpy(cur, x_in, sizeof(double) * n_in);

    for (l = 0; l < net->n_layers; l++) {
        DenseLayer *layer = &net->layers[l];
        for (i = 0; i < layer->d_out; i++) {
            double acc = layer->b[i];
            for (j = 0; j < layer->d_in; j++)
                acc += layer->W[i * layer->d_in + j] * cur[j];
            nxt[i] = apply_activation(acc, layer->activation);
        }
        double *tmp = cur; cur = nxt; nxt = tmp;
    }
    memcpy(x_out, cur, sizeof(double) * n_out);
}

/* =========================================================================
 * File I/O helpers
 * ========================================================================= */

static void load_mlp(MLP *net, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { Message("SCARFS NC UDF ERROR: cannot open %s\n", path); return; }
    int n_layers; fscanf(f, " LAYERS %d", &n_layers);
    net->n_layers = n_layers;
    int l, k;
    for (l = 0; l < n_layers; l++) {
        int idx, d_in, d_out; char act[32];
        fscanf(f, " LAYER %d in=%d out=%d activation=%31s", &idx, &d_in, &d_out, act);
        net->layers[l].d_in  = d_in;
        net->layers[l].d_out = d_out;
        strncpy(net->layers[l].activation, act, 31);
        net->layers[l].W = (double *)malloc(sizeof(double) * d_out * d_in);
        net->layers[l].b = (double *)malloc(sizeof(double) * d_out);
        for (k = 0; k < d_out * d_in; k++) fscanf(f, " %lf", &net->layers[l].W[k]);
        for (k = 0; k < d_out; k++)        fscanf(f, " %lf", &net->layers[l].b[k]);
    }
    fclose(f);
    Message("SCARFS NC UDF: loaded MLP from %s (%d layers)\n", path, n_layers);
}

static void load_all_weights(void)
{
    load_mlp(&g_encoder,    ENCODER_WEIGHTS_FILE);
    load_mlp(&g_decoder,    DECODER_WEIGHTS_FILE);
    load_mlp(&g_rate_net,   RATE_WEIGHTS_FILE);
    load_mlp(&g_energy_net, ENERGY_WEIGHTS_FILE);
}

static void load_scalers(void)
{
    FILE *f = fopen(SCALERS_FILE, "r");
    if (!f) { Message("SCARFS NC UDF ERROR: cannot open %s\n", SCALERS_FILE); return; }
    int n_scalers, s, k;
    fscanf(f, " N_SCALERS %d", &n_scalers);
    for (s = 0; s < n_scalers; s++) {
        char name[64], stype[64];
        fscanf(f, " SCALER %63s %63s", name, stype);
        if (strcmp(stype, "CompositionScaler") == 0) {
            int lf; double fl, flo, fhi;
            fscanf(f, " log %d floor %lf feature_range %lf %lf", &lf, &fl, &flo, &fhi);
            g_sc.comp_log = lf; g_sc.comp_floor = fl;
            g_sc.comp_feature_lo = flo; g_sc.comp_feature_hi = fhi;
            int nm; fscanf(f, " data_min %d", &nm);
            for (k = 0; k < nm; k++) fscanf(f, " %lf", &g_sc.comp_data_min[k]);
            int nx; fscanf(f, " data_max %d", &nx);
            for (k = 0; k < nx; k++) fscanf(f, " %lf", &g_sc.comp_data_max[k]);
            char end[32]; fscanf(f, " %31s", end);
        }
        else if (strcmp(stype, "StandardScaler") == 0) {
            int nm; fscanf(f, " mean %d", &nm);
            for (k = 0; k < nm; k++) fscanf(f, " %lf", &g_sc.thermo_mean[k]);
            int ns; fscanf(f, " scale %d", &ns);
            for (k = 0; k < ns; k++) fscanf(f, " %lf", &g_sc.thermo_scale[k]);
            char end[32]; fscanf(f, " %31s", end);
        }
        else if (strcmp(stype, "ArcsinhScaler") == 0) {
            /* TODO: distinguish omega_Z from energy scaler by name */
            double ms; fscanf(f, " min_scale %lf", &ms);
            int nas; fscanf(f, " arcsinh_scale %d", &nas);
            for (k = 0; k < nas; k++) fscanf(f, " %lf", &g_sc.omegaZ_arcsinh_scale[k]);
            int nsm; fscanf(f, " std_mean %d", &nsm);
            for (k = 0; k < nsm; k++) fscanf(f, " %lf", &g_sc.omegaZ_std_mean[k]);
            int nss; fscanf(f, " std_scale %d", &nss);
            for (k = 0; k < nss; k++) fscanf(f, " %lf", &g_sc.omegaZ_std_scale[k]);
            char end[32]; fscanf(f, " %31s", end);
        }
    }
    fclose(f);
    Message("SCARFS NC UDF: loaded scalers from %s\n", SCALERS_FILE);
}

/* =========================================================================
 * DEFINE_INIT — load all artefacts once
 * ========================================================================= */

DEFINE_INIT(scarfs_nc_init, domain)
{
    if (g_loaded) return;
    load_all_weights();
    load_scalers();
    g_loaded = 1;
    Message("SCARFS NC UDF: NeuralCoil initialised (k=%d latent scalars).\n", K_LATENT);
}

/* =========================================================================
 * DEFINE_ADJUST — manifold projection + rate evaluation
 *
 * Called ONCE per iteration, before source-term evaluation.
 * This is where the F2 fix (RC-2 guard) lives.
 * ========================================================================= */

DEFINE_ADJUST(scarfs_nc_adjust, domain)
{
    Thread *t;
    cell_t  c;

    thread_loop_c(t, domain) {
        begin_c_loop(c, t) {

            /* Read latent Z from UDS (k scalars stored in UDS 0 .. K_LATENT-1) */
            double Z[K_LATENT];
            int j;
            for (j = 0; j < K_LATENT; j++)
                Z[j] = C_UDSI(c, t, j);

            /* =================================================================
             * MANIFOLD PROJECTION  —  F2 fix for RC-2 (latent drift)
             *
             * Background: when Fluent transports the latent UDS scalars Z_j,
             * truncation errors and boundary conditions cause Z to drift off
             * the encoder manifold  Z = E · Y.  Once off-manifold, the decoded
             * Y is unphysical, the rate network returns large rates, and a
             * self-amplifying divergence follows (off-manifold residual ε grew
             * from 7 to 2777 in 326 iterations, thesis §5.5.2.3).
             *
             * Fix: each iteration, before evaluating omega_Z, project Z back:
             *     Y_hat   = D(Z)          [decoder: latent -> physical species]
             *     Z_proj  = E · Y_hat     [encoder: linear projection]
             *     Z <- Z_proj
             *
             * This re-anchors Z onto the manifold at negligible cost per cell.
             * The rate net then sees a physically consistent latent state.
             * ================================================================= */

            /* Step (i): decode Z -> Y_hat */
            double Y_hat[N_SC];
            mlp_forward(&g_decoder, Z, K_LATENT, Y_hat, N_SC);

            /* Step (ii): re-encode Y_hat -> Z_proj = E · Y_hat
             * The encoder is LINEAR (thesis Ch. 5); its forward pass is:
             *   Z_proj = W_enc · Y_hat + b_enc   (activation=linear)
             * We use the same mlp_forward helper for generality. */
            double Z_proj[K_LATENT];
            mlp_forward(&g_encoder, Y_hat, N_SC, Z_proj, K_LATENT);

            /* Step (iii): write projected Z back to UDS and local array */
            for (j = 0; j < K_LATENT; j++) {
                C_UDSI(c, t, j) = (real)Z_proj[j];
                Z[j] = Z_proj[j];
            }

            /* =================================================================
             * INPUT CLIPPING TO TRAINING RANGES
             * (ChemZIP §4.3 step 5 — guards RC-4, also limits RC-2 re-entry)
             * ================================================================= */
            double T = C_T(c, t);
            double P = C_P(c, t);

            double T_min = g_sc.thermo_mean[0] - 5.0 * g_sc.thermo_scale[0];
            double T_max = g_sc.thermo_mean[0] + 5.0 * g_sc.thermo_scale[0];
            double P_min = g_sc.thermo_mean[1] - 5.0 * g_sc.thermo_scale[1];
            double P_max = g_sc.thermo_mean[1] + 5.0 * g_sc.thermo_scale[1];
            if (T < T_min) T = T_min;
            if (T > T_max) T = T_max;
            if (P < P_min) P = P_min;
            if (P > P_max) P = P_max;

            /* =================================================================
             * Evaluate latent-space rate network omega_Z = f(Z_proj, T, P)
             * Rate net input: [Z_0..Z_{k-1}, T, P, 1/T, ln(T)]
             * ================================================================= */
            double rate_in[N_RATE_INPUT];
            for (j = 0; j < K_LATENT; j++)
                rate_in[j] = Z[j];
            double thermo[N_THERMO] = {T, P, 1.0/T, log(T)};
            for (j = 0; j < N_THERMO; j++)
                rate_in[K_LATENT + j] = (thermo[j] - g_sc.thermo_mean[j]) / g_sc.thermo_scale[j];

            double omega_Z_scaled[K_LATENT];
            mlp_forward(&g_rate_net, rate_in, N_RATE_INPUT, omega_Z_scaled, K_LATENT);

            /* Inverse-scale omega_Z [s-1] — stored in UDS source scratch if needed.
             * The actual source terms are returned by DEFINE_SOURCE below.
             * Here we pre-compute and cache in thread-local memory if available,
             * or recompute in DEFINE_SOURCE (simpler, shown below). */

        } end_c_loop(c, t)
    }
}

/* =========================================================================
 * DEFINE_SOURCE for each UDS (latent scalar transport equation)
 *
 * Fluent calls this for UDS_j = Z_j.  The UDS transport equation is:
 *   d(rho Z_j)/dt + div(rho u Z_j) = S_j   [kg m-3 s-1]
 *
 * The source is: S_j = rho * omega_Z_j   [kg m-3 s-1]
 * where omega_Z_j [s-1] is the latent source from the rate network.
 *
 * TODO: Duplicate for each of k=0..K_LATENT-1 and register in Fluent.
 * ========================================================================= */

static double get_latent_source(cell_t c, Thread *t, int latent_idx)
{
    /* Z was already projected in DEFINE_ADJUST; read it back */
    double Z[K_LATENT];
    int j;
    for (j = 0; j < K_LATENT; j++)
        Z[j] = C_UDSI(c, t, j);

    double T = C_T(c, t);
    double P = C_P(c, t);

    /* Clip T, P */
    double T_min = g_sc.thermo_mean[0] - 5.0 * g_sc.thermo_scale[0];
    double T_max = g_sc.thermo_mean[0] + 5.0 * g_sc.thermo_scale[0];
    if (T < T_min) T = T_min; if (T > T_max) T = T_max;
    double P_min = g_sc.thermo_mean[1] - 5.0 * g_sc.thermo_scale[1];
    double P_max = g_sc.thermo_mean[1] + 5.0 * g_sc.thermo_scale[1];
    if (P < P_min) P = P_min; if (P > P_max) P = P_max;

    double rate_in[N_RATE_INPUT];
    for (j = 0; j < K_LATENT; j++) rate_in[j] = Z[j];
    double thermo[N_THERMO] = {T, P, 1.0/T, log(T)};
    for (j = 0; j < N_THERMO; j++)
        rate_in[K_LATENT + j] = (thermo[j] - g_sc.thermo_mean[j]) / g_sc.thermo_scale[j];

    double omega_scaled[K_LATENT];
    mlp_forward(&g_rate_net, rate_in, N_RATE_INPUT, omega_scaled, K_LATENT);

    /* Inverse arcsinh scale for this latent index */
    double raw = omega_scaled[latent_idx] * g_sc.omegaZ_std_scale[latent_idx]
                 + g_sc.omegaZ_std_mean[latent_idx];
    double omega_j = sinh(raw) * g_sc.omegaZ_arcsinh_scale[latent_idx]; /* [s-1] */

    /* Multiply by density to get [kg m-3 s-1] for the UDS source */
    double rho = C_R(c, t);
    return rho * omega_j;
}

/* TODO: Replace with K_LATENT DEFINE_SOURCE macros, one per latent scalar. */
DEFINE_SOURCE(scarfs_nc_uds_source_0, cell, thread, dS, eqn)
{
    dS[eqn] = 0.0;
    return (real)get_latent_source(cell, thread, 0);
}

/* =========================================================================
 * DEFINE_SOURCE for the energy equation
 * ========================================================================= */

DEFINE_SOURCE(scarfs_nc_energy_source, cell, thread, dS, eqn)
{
    double Z[K_LATENT];
    int j;
    for (j = 0; j < K_LATENT; j++) Z[j] = C_UDSI(cell, thread, j);

    double T = C_T(cell, thread);
    double P = C_P(cell, thread);
    double T_min = g_sc.thermo_mean[0] - 5.0 * g_sc.thermo_scale[0];
    double T_max = g_sc.thermo_mean[0] + 5.0 * g_sc.thermo_scale[0];
    if (T < T_min) T = T_min; if (T > T_max) T = T_max;
    double P_min = g_sc.thermo_mean[1] - 5.0 * g_sc.thermo_scale[1];
    double P_max = g_sc.thermo_mean[1] + 5.0 * g_sc.thermo_scale[1];
    if (P < P_min) P = P_min; if (P > P_max) P = P_max;

    double energy_in[N_RATE_INPUT];
    for (j = 0; j < K_LATENT; j++) energy_in[j] = Z[j];
    double thermo[N_THERMO] = {T, P, 1.0/T, log(T)};
    for (j = 0; j < N_THERMO; j++)
        energy_in[K_LATENT + j] = (thermo[j] - g_sc.thermo_mean[j]) / g_sc.thermo_scale[j];

    double S_E_scaled[1];
    mlp_forward(&g_energy_net, energy_in, N_RATE_INPUT, S_E_scaled, 1);

    double raw = S_E_scaled[0] * g_sc.energy_std_scale + g_sc.energy_std_mean;
    double S_E = sinh(raw) * g_sc.energy_arcsinh_scale; /* [J m-3 s-1] */

    dS[eqn] = 0.0;
    return (real)S_E;
}
