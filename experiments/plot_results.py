"""
plot_results.py
===============
Reads the CSV produced by run_pacman.py or run_cartpole and generates two learning-curve
plots modelled after the EF-OnTL:

  1. **Binned figure** (main, publication-quality):
     - Episodes are grouped into fixed-size bins (default 50).
     - For each bin, mean +/- 1 SE error bars are drawn as discrete markers
       (no continuous shaded band), matching the style of the original paper.
     - A vertical dashed line annotates where knowledge transfer begins.

  2. **Raw figure** (per-episode, for transparency / appendix):
     - Mean line + shaded +-1 SE band.
     - Saved as <out_stem>_raw.png.

Usage
-----
    python plot_results.py
    python plot_results.py --csv results/transfer_results_cartpole.csv \
                           --config EFO_TL_parameters_cartpole.json \
                           --out results/learning_curve.png \
                           --bin-size 50
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Optional seaborn for theming — imported only if available
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_selection_label(mode: str) -> str:
    """Convert raw source_selection_mode string to a readable abbreviation."""
    _map = {
        "avg_uncertainty": "Avg. Uncertainty (U)",
        "avg_confidence":  "Avg. Confidence (C)",
        "random":          "Random",
        "loss":            "Loss",
    }
    return _map.get(mode, mode)


def _build_subtitle(params: dict) -> str:
    """
    Build the second title line dynamically from JSON values.
    Example:  "SS: Avg. Uncertainty (U)  .  B=5,000  .  transfer from ep 600"
    """
    transfer = params.get("transfer", {})
    ss_mode  = _source_selection_label(transfer.get("source_selection_mode", "N/A"))
    batch_b  = transfer.get("batch_transfer_size", "N/A")
    ep_share = transfer.get("ep_start_sharing", "N/A")

    try:
        batch_str = f"{int(batch_b):,}"
    except (ValueError, TypeError):
        batch_str = str(batch_b)

    return f"SS: {ss_mode}  \u00b7  B={batch_str}  \u00b7  transfer from ep {ep_share}"


# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Plot EFo-TL learning curves (publication quality).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--csv",
    default=None,
    help="Path to results CSV. Falls back to value in the config JSON.",
)
parser.add_argument(
    "--config",
    default="EFO_TL_parameters_cartpole.json",
    help="Path to the EFo-TL JSON configuration file.",
)
parser.add_argument(
    "--out",
    default=None,
    help="Output PNG path for the binned figure. "
         "Defaults to <csv_dir>/learning_curve.png.",
)
parser.add_argument(
    "--bin-size",
    type=int,
    default=50,
    dest="bin_size",
    help="Number of consecutive episodes to group into each bin.",
)
parser.add_argument(
    "--env-label",
    default=None,
    dest="env_label",
    help="Human-readable environment label used in the plot title when "
         "env_name is not present in the config JSON.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
if not os.path.isfile(args.config):
    sys.exit(f"[ERROR] Config file not found: {args.config}")

with open(args.config, "r") as f:
    parameters = json.load(f)

# Resolve CSV path
csv_path = args.csv or parameters.get("experiment", {}).get("results_csv")
if not csv_path:
    sys.exit("[ERROR] No CSV path provided via --csv and none found in config JSON.")

# Resolve output path (binned figure)
out_dir   = os.path.dirname(csv_path) or "."
out_path  = args.out or os.path.join(out_dir, "learning_curve.png")

# Derive raw-figure path from the main output path
stem, ext      = os.path.splitext(out_path)
out_path_raw   = f"{stem}_raw{ext or '.png'}"

os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data  -- column names must be: Simulation, Agent, Episode, Reward
# ---------------------------------------------------------------------------
if not os.path.isfile(csv_path):
    sys.exit(f"[ERROR] CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows from '{csv_path}'")

required_cols = {"Simulation", "Agent", "Episode", "Reward"}
missing = required_cols - set(df.columns)
if missing:
    sys.exit(f"[ERROR] CSV is missing expected columns: {missing}")

# ---------------------------------------------------------------------------
# Metadata from config / CLI
# ---------------------------------------------------------------------------
training    = parameters.get("training", {})
transfer    = parameters.get("transfer", {})
experiment  = parameters.get("experiment", {})

env_name        = training.get("env_name") or args.env_label or "Unknown Env"
num_sims_data   = df["Simulation"].nunique()
num_agents_data = df["Agent"].nunique()
ep_start        = transfer.get("ep_start_sharing", None)

subtitle    = _build_subtitle(parameters)
title_line1 = (
    f"EFo-TL  \u00b7  {env_name}  \u00b7  "
    f"{num_sims_data} sim(s) \u00d7 {num_agents_data} agent(s)"
)

# ---------------------------------------------------------------------------
# Apply seaborn theme (if available)
# ---------------------------------------------------------------------------
if _HAS_SEABORN:
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
else:
    plt.rcParams.update({
        "axes.grid":      True,
        "grid.linestyle": "--",
        "grid.alpha":     0.4,
        "font.family":    "sans-serif",
    })

# ---------------------------------------------------------------------------
# (1) Per-episode statistics  (used by the raw figure)
# ---------------------------------------------------------------------------
grouped_raw  = df.groupby("Episode")["Reward"]
episodes_raw = np.array(sorted(df["Episode"].unique()))
means_raw    = grouped_raw.mean().reindex(episodes_raw).values
stds_raw     = grouped_raw.std().reindex(episodes_raw).values
counts_raw   = grouped_raw.count().reindex(episodes_raw).values
se_raw       = stds_raw / np.sqrt(counts_raw)

# ---------------------------------------------------------------------------
# (2) Binned statistics  (used by the main publication figure)
# ---------------------------------------------------------------------------
bin_size  = args.bin_size
max_ep    = int(df["Episode"].max())
bin_edges = range(0, max_ep + bin_size, bin_size)

bin_labels = []
bin_means  = []
bin_se     = []

for lo in bin_edges:
    hi    = lo + bin_size
    mask  = (df["Episode"] >= lo) & (df["Episode"] < hi)
    chunk = df.loc[mask, "Reward"]
    if len(chunk) == 0:
        continue
    n = len(chunk)
    bin_labels.append(lo + bin_size / 2.0)   # midpoint of bin range
    bin_means.append(chunk.mean())
    bin_se.append(chunk.std() / math.sqrt(n))

bin_labels = np.array(bin_labels)
bin_means  = np.array(bin_means)
bin_se     = np.array(bin_se)

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
COLOR_MAIN  = "#2563EB"   # vivid blue  -- main line / markers
COLOR_SHADE = "#93C5FD"   # light blue  -- raw shaded band
COLOR_VLINE = "#6B7280"   # muted grey  -- transfer line

# ===========================================================================
# FIGURE 1 -- Binned (publication-quality)
# ===========================================================================
fig1, ax1 = plt.subplots(figsize=(11, 5.5))

# Mean line + discrete error-bar markers
ax1.errorbar(
    bin_labels, bin_means, yerr=bin_se,
    fmt="o-",
    color=COLOR_MAIN,
    linewidth=2,
    markersize=5,
    capsize=4,
    capthick=1.4,
    elinewidth=1.2,
    label="Mean reward \u00b1 1 SE",
    zorder=3,
)

# Vertical dashed line at transfer start
if ep_start is not None:
    ax1.axvline(
        x=ep_start,
        color=COLOR_VLINE,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
    )
    # Annotate -- position text slightly to the right of the line
    ylims = ax1.get_ylim()
    y_text = ylims[0] + (ylims[1] - ylims[0]) * 0.92
    ax1.text(
        ep_start + max(bin_size * 0.6, 5),
        y_text,
        "Transfer\nstarts",
        color=COLOR_VLINE,
        fontsize=9,
        va="top",
        ha="left",
        linespacing=1.3,
    )

ax1.set_title(
    f"{title_line1}\n{subtitle}",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax1.set_xlabel("Episode", fontsize=12)
ax1.set_ylabel("Accumulated Reward per Episode", fontsize=12)
ax1.legend(fontsize=10, framealpha=0.85)
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))

plt.tight_layout()
fig1.savefig(out_path, dpi=300)
print(f"[OK] Binned figure saved to: {out_path}")
plt.close(fig1)

# ===========================================================================
# FIGURE 2 -- Raw per-episode (transparency / appendix)
# ===========================================================================
fig2, ax2 = plt.subplots(figsize=(11, 5.5))

ax2.plot(
    episodes_raw, means_raw,
    color=COLOR_MAIN,
    linewidth=1.8,
    label="Mean reward",
    zorder=3,
)
ax2.fill_between(
    episodes_raw,
    means_raw - se_raw,
    means_raw + se_raw,
    color=COLOR_SHADE,
    alpha=0.45,
    label="\u00b11 Std. Error",
    zorder=2,
)

# Vertical dashed line at transfer start
if ep_start is not None:
    ax2.axvline(
        x=ep_start,
        color=COLOR_VLINE,
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )
    ylims2  = ax2.get_ylim()
    y_text2 = ylims2[0] + (ylims2[1] - ylims2[0]) * 0.92
    ax2.text(
        ep_start + max(bin_size * 0.6, 5),
        y_text2,
        "Transfer\nstarts",
        color=COLOR_VLINE,
        fontsize=9,
        va="top",
        ha="left",
        linespacing=1.3,
    )

ax2.set_title(
    f"{title_line1} [per-episode / raw]\n{subtitle}",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax2.set_xlabel("Episode", fontsize=12)
ax2.set_ylabel("Accumulated Reward per Episode", fontsize=12)
ax2.legend(fontsize=10, framealpha=0.85)
ax2.grid(True, linestyle="--", alpha=0.35)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))

plt.tight_layout()
fig2.savefig(out_path_raw, dpi=300)
print(f"[OK] Raw (per-episode) figure saved to: {out_path_raw}")
plt.close(fig2)
