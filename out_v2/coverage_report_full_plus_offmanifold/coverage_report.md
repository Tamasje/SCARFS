# SCARFS full + off-manifold coverage check
## Files
- full: `C:\Users\mbonheur\OneDrive - UGent\Documenten\GitHub\SCARFS\out_v2\full.parquet`
- offmanifold: `C:\Users\mbonheur\OneDrive - UGent\Documenten\GitHub\SCARFS\out_v2\offmanifold_1000000.parquet`

## Row counts
- full rows from parquet metadata: 749,001
- off-manifold rows from parquet metadata: 1,000,000
- full sample rows used: 250,000
- off-manifold sample rows used: 250,000

## State variables used
- T [K]
- P [bar]
- tau [ms]
- X_C2H6 [%]
- Y_C2H6
- Y_C2H4
- Y_C2H2
- Y_C3H6
- Y_C3H8
- Heat absorption [MW/m3]
- Wall source [MW/m3]

## PCA
- PC1 explained variance: 43.36%
- PC2 explained variance: 12.89%
- PC1 + PC2 explained variance: 56.26%

## Nearest-distance interpretation
| metric                         |     n |   n_columns | columns_used                                                             |        p05 |       p50 |      p90 |      p95 |      p99 |      max |   frac_farther_than_full_p95 |   frac_farther_than_full_p99 |
|:-------------------------------|------:|------------:|:-------------------------------------------------------------------------|-----------:|----------:|---------:|---------:|---------:|---------:|-----------------------------:|-----------------------------:|
| full_to_nearest_full_reference | 80000 |           8 | T [K];P [bar];Y_C2H6;Y_C2H4;Y_C2H2;Y_C3H6;Y_C3H8;Heat absorption [MW/m3] | 0.00151093 | 0.0150436 | 0.040675 | 0.057915 | 0.112079 | 0.478913 |                   nan        |                   nan        |
| off_to_nearest_full            | 80000 |           8 | T [K];P [bar];Y_C2H6;Y_C2H4;Y_C2H2;Y_C3H6;Y_C3H8;Heat absorption [MW/m3] | 0.0185385  | 0.0570684 | 0.221988 | 0.413997 | 0.778855 | 2.43519  |                     0.491013 |                     0.185988 |

Interpretation: `off_to_nearest_full` measures how far perturbed points sit from the original trajectory manifold after robust normalisation by the full trajectory envelope. If the off-manifold median is close to the full-to-full nearest-neighbour scale, the cloud is mainly local thickening. If the p95/p99 values are much larger, the cloud also explores wider drift regions.

## Off-manifold outside the full envelope
| variable                |      full_min |      full_p01 |      full_p99 |       full_max |   off_n |   off_below_full_min_frac |   off_above_full_max_frac |   off_outside_full_minmax_frac |   off_outside_full_p01p99_frac |
|:------------------------|--------------:|--------------:|--------------:|---------------:|--------:|--------------------------:|--------------------------:|-------------------------------:|-------------------------------:|
| tau [ms]                |   0           |   0           |  717.802      |  2003.9        |  250000 |                  1        |                  0        |                       1        |                       1        |
| Heat absorption [MW/m3] |  -4.87958     |   8.50305e-05 |  389.182      | 14077.5        |  250000 |                  0.040964 |                  0        |                       0.040964 |                       0.110452 |
| Y_C2H6                  |   0.00101914  |   0.134112    |    0.899999   |     0.9        |  250000 |                  3.2e-05  |                  0.046436 |                       0.046468 |                       0.057504 |
| P [bar]                 |   1.50017     |   1.52031     |    3.47751    |     3.49987    |  250000 |                  0.00982  |                  0.019712 |                       0.029532 |                       0.041276 |
| T [K]                   | 823.25        | 840.62        | 1209.55       |  1422.43       |  250000 |                  0.003564 |                  7.2e-05  |                       0.003636 |                       0.02188  |
| Y_C2H4                  |  -4.32752e-08 |   0           |    0.38984    |     0.568076   |  250000 |                  0        |                  0.000444 |                       0.000444 |                       0.013608 |
| Y_C3H8                  |  -9.60451e-16 |   0           |    0.00144309 |     0.00237963 |  250000 |                  0        |                  0.001412 |                       0.001412 |                       0.01194  |
| Y_C3H6                  |  -2.40453e-15 |   0           |    0.0101236  |     0.0181118  |  250000 |                  0        |                  0.000984 |                       0.000984 |                       0.01104  |
| Y_C2H2                  |  -1.46879e-18 |   0           |    0.00598626 |     0.188095   |  250000 |                  0        |                  4e-06    |                       4e-06    |                       0.010112 |
| Wall source [MW/m3]     |   0           |   0           |   24.5551     |    32.6495     |  250000 |                  0        |                  0        |                       0        |                       0        |

Interpretation: values outside the full min-max envelope are genuinely outside the original trajectory range for that variable. Values outside p01-p99 but inside min-max usually indicate useful enrichment of tails rather than unphysical extrapolation.

## Pairwise bin-coverage gain
| x      | y                       |   bins_per_axis |   full_occupied_bins |   off_occupied_bins |   combined_occupied_bins |   new_bins_added_by_off |   full_occupied_fraction |   off_occupied_fraction |   combined_occupied_fraction |   absolute_coverage_gain |   relative_gain_vs_full |   n_full |   n_off |
|:-------|:------------------------|----------------:|---------------------:|--------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-----------------------------:|-------------------------:|------------------------:|---------:|--------:|
| Y_C3H6 | Heat absorption [MW/m3] |              30 |                  103 |                 524 |                      524 |                     421 |                 0.114444 |                0.582222 |                     0.582222 |                 0.467778 |                4.08738  |   180000 |  180000 |
| Y_C2H2 | Heat absorption [MW/m3] |              30 |                  128 |                 481 |                      483 |                     355 |                 0.142222 |                0.534444 |                     0.536667 |                 0.394444 |                2.77344  |   180000 |  180000 |
| Y_C2H4 | Heat absorption [MW/m3] |              30 |                  379 |                 697 |                      707 |                     328 |                 0.421111 |                0.774444 |                     0.785556 |                 0.364444 |                0.865435 |   180000 |  180000 |
| Y_C2H6 | Y_C2H4                  |              30 |                  357 |                 674 |                      674 |                     317 |                 0.396667 |                0.748889 |                     0.748889 |                 0.352222 |                0.887955 |   180000 |  180000 |
| T [K]  | Y_C2H6                  |              30 |                  596 |                 835 |                      835 |                     239 |                 0.662222 |                0.927778 |                     0.927778 |                 0.265556 |                0.401007 |   180000 |  180000 |
| Y_C2H6 | Heat absorption [MW/m3] |              30 |                  600 |                 834 |                      837 |                     237 |                 0.666667 |                0.926667 |                     0.93     |                 0.263333 |                0.395    |   180000 |  180000 |
| Y_C2H4 | Y_C3H6                  |              30 |                  438 |                 655 |                      656 |                     218 |                 0.486667 |                0.727778 |                     0.728889 |                 0.242222 |                0.497717 |   180000 |  180000 |
| Y_C3H6 | Y_C3H8                  |              30 |                  575 |                 766 |                      769 |                     194 |                 0.638889 |                0.851111 |                     0.854444 |                 0.215556 |                0.337391 |   180000 |  180000 |
| Y_C3H8 | Heat absorption [MW/m3] |              30 |                  582 |                 715 |                      774 |                     192 |                 0.646667 |                0.794444 |                     0.86     |                 0.213333 |                0.329897 |   180000 |  180000 |
| Y_C2H4 | Y_C3H8                  |              30 |                  549 |                 733 |                      737 |                     188 |                 0.61     |                0.814444 |                     0.818889 |                 0.208889 |                0.342441 |   180000 |  180000 |
| Y_C2H4 | Y_C2H2                  |              30 |                  400 |                 582 |                      588 |                     188 |                 0.444444 |                0.646667 |                     0.653333 |                 0.208889 |                0.47     |   180000 |  180000 |
| Y_C2H2 | Y_C3H8                  |              30 |                  550 |                 677 |                      703 |                     153 |                 0.611111 |                0.752222 |                     0.781111 |                 0.17     |                0.278182 |   180000 |  180000 |
| Y_C2H6 | Y_C3H8                  |              30 |                  638 |                 770 |                      772 |                     134 |                 0.708889 |                0.855556 |                     0.857778 |                 0.148889 |                0.210031 |   180000 |  180000 |
| Y_C2H6 | Y_C2H2                  |              30 |                  253 |                 366 |                      366 |                     113 |                 0.281111 |                0.406667 |                     0.406667 |                 0.125556 |                0.44664  |   180000 |  180000 |
| T [K]  | Y_C2H2                  |              30 |                  223 |                 332 |                      333 |                     110 |                 0.247778 |                0.368889 |                     0.37     |                 0.122222 |                0.493274 |   180000 |  180000 |

Interpretation: high absolute gain means that the off-manifold cloud adds new occupied 2D regions that were absent in `full.parquet`. Focus especially on projections involving heat absorption, acetylene, conversion and temperature.

## Automatic conclusion
- The off-manifold file is not just a duplicate of the full trajectory file: its median nearest distance to the full manifold is 0.05707, with p95 0.414.
- The off-manifold cloud gives a mixed near- and wider-perturbation dataset; this is probably useful, but wide points should not dominate the training weights.
- The mean pairwise bin-coverage gain from off-manifold data is 10.6% of all 2D bins, with a maximum gain of 46.8%.
- The largest fraction outside the full min-max envelope for any checked variable is 100.0%.

## Output files to inspect first
- `coverage_report.md`
- `range_comparison_by_dataset.csv`
- `off_outside_full_envelope.csv`
- `pairwise_2d_bin_gain.csv`
- `nearest_off_to_full_summary.csv`
- `02_pca_full_vs_offmanifold.png`
- `03_off_to_full_nearest_distance.png`
- `04_off_outside_full_envelope.png`
- `05_pairwise_coverage_gain.png`
- `06_T_vs_heat_density.png`
- `07_X_or_YC2H6_vs_heat_density.png`
- `08_C2H2_vs_heat_density.png`
