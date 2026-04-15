"""Generate raw and normed mean heatmap plots for noun_scan_v5_full by patching common paths."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent.parent
V5_OUTPUTS = BASE / "outputs" / "noun_scan_v5_full"
V5_PLOTS = BASE / "plots" / "noun_scan_v5_full" / "playground"
V5_HIGHLIGHTS = BASE / "plots" / "noun_scan_v5_full" / "highlights"

import common
common.OUTPUTS_DIR = V5_OUTPUTS
common.PLOTS_DIR = V5_PLOTS
common.HIGHLIGHTS_DIR = V5_HIGHLIGHTS

import gap_per_secret_grid_raw
gap_per_secret_grid_raw.plot()
print("Saved gap_per_secret_grid_raw.png (v5)")
gap_per_secret_grid_raw.plot_sample4()
print("Saved gap_per_secret_grid_raw_sample4.png (v5)")

import gap_mean_heatmap_raw
gap_mean_heatmap_raw.plot()
print("Saved gap_mean_heatmap_raw.png (v5)")

import gap_mean_heatmap_normed
gap_mean_heatmap_normed.plot()
print("Saved gap_mean_heatmap_normed.png (v5)")

import gap_mean_heatmap_raw_maxmean
gap_mean_heatmap_raw_maxmean.plot()
print("Saved gap_mean_heatmap_raw_maxmean.png (v5)")

import gap_mean_heatmap_normed_maxmean
gap_mean_heatmap_normed_maxmean.plot()
print("Saved gap_mean_heatmap_normed_maxmean.png (v5)")

import gap_mean_heatmap_baseline
gap_mean_heatmap_baseline.plot_raw()
print("Saved gap_mean_heatmap_raw_maxmean_baseline.png (v5)")
gap_mean_heatmap_baseline.plot_normed()
print("Saved gap_mean_heatmap_normed_maxmean_baseline.png (v5)")

import gap_mean_heatmap_nooi
gap_mean_heatmap_nooi.plot_raw()
print("Saved gap_mean_heatmap_raw_nooi.png (v5)")
gap_mean_heatmap_nooi.plot_normed()
print("Saved gap_mean_heatmap_normed_nooi.png (v5)")
