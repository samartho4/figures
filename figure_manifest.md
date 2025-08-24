# Figure Manifest

**Generated**: 2025-08-24T17:46:38.993

## Input Files

- **Data Source**: per_scenario_summary.csv
- **SHA256**: ee419dcf4dba0b156923a15bffe5882a39aff70c5482f4eda62bebe5d36fb181
- **Scenarios**: 10

## Generated Figures

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig 1 | fig1_scatter_rmse_x2_ude_vs_physics | UDE vs Physics scatter plot |
| Fig 2 | fig2_hist_delta_rmse_x2_ude_minus_physics | Delta histogram |
| Fig 3 | fig3_bland_altman_rmse_x2 | Bland-Altman plot |
| Fig 4 | fig4_paired_lines_rmse_x2_by_model | Paired lines plot |
| Fig 5 | fig5_r2x2_delta_ude_minus_physics | R² delta bar plot |
| Fig 6 | fig6_calibration_bnode_pre_post | BNode calibration |
| Fig 7 | fig7_baselines_rmse_r2_summary | Baseline comparison |

## Statistical Results

### RMSE x2
- Mean delta: -0.004488
- 95% CI: [-0.039525, 0.031113]
- Wilcoxon p-value: 0.921875
- Cohen's dz: -0.074712

### R² x2
- Mean delta: -0.032195
- 95% CI: [-0.142912, 0.07223]
- Wilcoxon p-value: 0.845703
- Cohen's dz: -0.175481

## Library Versions

- Julia: 1.11.6
- Plots: Available
- DataFrames: Available
- HypothesisTests: Available
- StatsBase: Available

## Random Seed
- Seed: 42
- Bootstrap resamples: 10,000
