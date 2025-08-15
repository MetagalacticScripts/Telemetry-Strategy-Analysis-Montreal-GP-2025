# Scripts/03_plots_basic.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTDIR = ROOT / "Reports" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
features = pd.read_parquet(DATA / "laps_features_canada_2025.parquet")

# Safety: only keep rows with the essentials
need_cols = ["Driver","LapNumber","LapTimeSeconds","Compound","StintNo","DeltaToStintMedian"]
missing = [c for c in need_cols if c not in features.columns]
if missing:
    raise RuntimeError(f"Missing columns in features parquet: {missing}")

# ---------- helper: lap-status bands (SC/VSC) ----------
def compute_status_bands(df: pd.DataFrame):
    """
    Returns spans like [(start_lap, end_lap, 'SC'/'VSC'), ...] using the per-lap modal TrackStatusLabel.
    If TrackStatusLabel doesn't exist, returns [].
    """
    if "TrackStatusLabel" not in df.columns:
        return []
    tmp = df[["LapNumber","TrackStatusLabel"]].dropna()
    # Per lap, take the most common status across drivers
    mode_status = (
        tmp.groupby("LapNumber")["TrackStatusLabel"]
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "GREEN")
    )
    spans = []
    current = None  # (status, start_lap)
    for lap, status in mode_status.items():
        status = str(status)
        status_key = "SC" if status.startswith("SC") else ("VSC" if status.startswith("VSC") else "GREEN")
        if status_key in ("SC","VSC"):
            if current is None:
                current = (status_key, lap)
            elif current[0] != status_key:
                # close previous and start new
                spans.append((current[1], lap-1, current[0]))
                current = (status_key, lap)
        else:
            if current is not None:
                spans.append((current[1], lap-1, current[0]))
                current = None
    if current is not None:
        last_lap = int(mode_status.index.max())
        spans.append((current[1], last_lap, current[0]))
    return spans

status_spans = compute_status_bands(features)

# ---------- Plot 1: Lap evolution (per driver rolling median) ----------
def plot_lap_evolution(df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Rolling median per driver to smooth noise
    for drv, g in df.groupby("Driver"):
        g = g.dropna(subset=["LapNumber","LapTimeSeconds"]).sort_values("LapNumber")
        if g.empty:
            continue
        y = (
            g["LapTimeSeconds"]
            .rolling(window=5, min_periods=3, center=True)
            .median()
        )
        ax.plot(g["LapNumber"], y, label=drv, linewidth=1.6)

    # SC/VSC shading
    for start, end, kind in status_spans:
        ax.axvspan(start-0.5, end+0.5, alpha=0.12, hatch=None, label=None)

    ax.set_title("Montreal 2025 — Lap Evolution (Rolling Median by Driver)")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (s)")
    # Keep legend readable
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

plot_lap_evolution(features, OUTDIR / "lap_evolution_rolling.png")

# ---------- Plot 2: Stint pace vs compound ----------
def plot_stint_pace_vs_compound(df: pd.DataFrame, outpath: Path):
    # Use normalized delta vs stint median: values < 0 = faster than own stint median
    sub = df.dropna(subset=["Compound","DeltaToStintMedian"]).copy()
    order = ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"]
    sub["Compound"] = sub["Compound"].astype(str).str.upper().str.strip()
    comp_groups = [g for c, g in sub.groupby("Compound")]

    # Build data in the desired order but only for compounds present
    data = []
    labels = []
    for c in order:
        arr = sub.loc[sub["Compound"] == c, "DeltaToStintMedian"].to_numpy()
        if arr.size > 0:
            data.append(arr)
            labels.append(c)

    if not data:
        raise RuntimeError("No compound data available to plot.")

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        whis=(10, 90),
        patch_artist=False  # keep default styling
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Montreal 2025 — Stint-Normalized Pace by Compound\n(Δ to own stint median, s)")
    ax.set_ylabel("Δ to Stint Median (s) — lower is better")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

plot_stint_pace_vs_compound(features, OUTDIR / "stint_pace_by_compound.png")

print("Saved plots to:", OUTDIR)
