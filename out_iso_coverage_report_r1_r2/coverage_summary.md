# Isothermal enrichment coverage summary

Script version: `visualize_isothermal_enrichment_coverage_v3_multi_iso_rstar`

## Quantitative coverage

### TX_coverage

- Occupied bins before: 99 / 320

- Occupied bins after: 319 / 320

- Newly occupied by isothermal enrichment: 220 bins

- Empty bins before → after: 221 → 1

- Under-target bins before → after: 251 → 64 using target_count=200

- Isothermal rows in previously empty bins: 112890

- Isothermal rows in previously under-target bins: 149660

### Ttau_coverage

- Occupied bins before: 184 / 320

- Occupied bins after: 320 / 320

- Newly occupied by isothermal enrichment: 136 bins

- Empty bins before → after: 136 → 0

- Under-target bins before → after: 183 → 16 using target_count=200

- Isothermal rows in previously empty bins: 119928

- Isothermal rows in previously under-target bins: 241023

## Isothermal case diagnostics

- `n_cases_with_isothermal_rows`: 28423

- `case_level_outcome_counts`: {'PFR missed → state probe': 21444, 'Native PFR failed → state probe': 4423, 'PFR missed → no suitable anchor': 2183, 'PFR missed → no probe/unknown': 373}

- `case_level_probe_used_cases`: 25867

- `case_level_unfilled_cases`: 2556

- `case_sample_kind_patterns`: {"('state_probe', 'trajectory')": 21444, "('state_probe',)": 4423, "('trajectory',)": 2556}

- `case_final_design_kind_counts`: {'isothermal_pfr_missed_target': 23627, 'anchored_state_probe_after_pfr_miss': 21444, 'anchored_state_probe_after_native_pfr_failure': 4423, 'isothermal_pfr_hit_target': 373}

- `pfr_hit_cases`: 373

- `pfr_hit_fraction`: 0.013123174893572107

- `fallback_status_case_counts`: {'<NA>': 21817, 'used_after_pfr_miss': 21444, 'used_after_native_pfr_failure': 4423, 'no_suitable_anchor_available': 2183}

## Generated figures

See `figures/` for overlay scatter plots, occupancy heatmaps, coverage-gain maps, and case-level outcome plots.


Case-level data are written to `isothermal_case_level_summary.csv`; this is the preferred table for judging PFR hit/miss/probe outcomes because it counts each CaseID once.
