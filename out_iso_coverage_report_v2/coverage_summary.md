# Isothermal enrichment coverage summary

Script version: `visualize_isothermal_enrichment_coverage_v2_case_level`

## Quantitative coverage

### TX_coverage

- Occupied bins before: 99 / 320

- Occupied bins after: 318 / 320

- Newly occupied by isothermal enrichment: 219 bins

- Empty bins before → after: 221 → 2

- Under-target bins before → after: 251 → 223 using target_count=200

- Isothermal rows in previously empty bins: 18372

- Isothermal rows in previously under-target bins: 26749

### Ttau_coverage

- Occupied bins before: 184 / 320

- Occupied bins after: 320 / 320

- Newly occupied by isothermal enrichment: 136 bins

- Empty bins before → after: 136 → 0

- Under-target bins before → after: 183 → 127 using target_count=200

- Isothermal rows in previously empty bins: 30675

- Isothermal rows in previously under-target bins: 46524

## Isothermal case diagnostics

- `n_cases_with_isothermal_rows`: 4703

- `case_level_outcome_counts`: {'PFR missed → no suitable anchor': 2183, 'PFR missed → state probe': 2111, 'Native PFR failed → state probe': 274, 'PFR missed → no probe/unknown': 135}

- `case_level_probe_used_cases`: 2385

- `case_level_unfilled_cases`: 2318

- `case_sample_kind_patterns`: {"('trajectory',)": 2318, "('state_probe', 'trajectory')": 2111, "('state_probe',)": 274}

- `case_final_design_kind_counts`: {'isothermal_pfr_missed_target': 4294, 'anchored_state_probe_after_pfr_miss': 2111, 'anchored_state_probe_after_native_pfr_failure': 274, 'isothermal_pfr_hit_target': 135}

- `pfr_hit_cases`: 135

- `pfr_hit_fraction`: 0.02870508186264087

- `fallback_status_case_counts`: {'<NA>': 2246, 'no_suitable_anchor_available': 2183, 'used_after_pfr_miss': 2111, 'used_after_native_pfr_failure': 274}

## Generated figures

See `figures/` for overlay scatter plots, occupancy heatmaps, coverage-gain maps, and case-level outcome plots.


Case-level data are written to `isothermal_case_level_summary.csv`; this is the preferred table for judging PFR hit/miss/probe outcomes because it counts each CaseID once.
