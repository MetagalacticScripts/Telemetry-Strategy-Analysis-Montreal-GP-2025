# Scripts/04_strategy_plots.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTDIR = ROOT / "Reports" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load processed + raw laps
# -----------------------------
features = pd.read_parquet(DATA / "laps_features_canada_2025.parquet")
stints_summary = pd.read_parquet(DATA / "stints_summary_canada_2025.parquet")

# We also want raw laps (to read in/out laps for pit-loss calc)
raw = pd.read_parquet(DATA / "laps_canada_2025.parquet").copy()

# Normalize a few columns on raw to make logic robust
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "LapTimeSeconds" not in df.columns and "LapTime" in df.columns:
        try:
            df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
        except Exception:
            pass
    for c in ("Driver","Compound","TrackStatus"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    # mark in/out laps if columns exist
    df["IsInLap"]  = df.get("PitInTime").notna()  if "PitInTime"  in df.columns else False
    df["IsOutLap"] = df.get("PitOutTime").notna() if "PitOutTime" in df.columns else False
    # map track status to labels (minimal)
    ts = df.get("TrackStatus")
    if ts is not None:
        m = {"1":"GREEN","2":"YELLOW","3":"DOUBLE_YELLOW","4":"SC","5":"VSC","6":"VSC_ENDING","7":"SC_ENDING"}
        df["TrackStatusLabel"] = ts.astype(str).str.strip().map(m).fillna("UNKNOWN")
    return df

raw = _normalize(raw)

# Ensure driver order is stable (by finishing order if available; else alphabetical)
driver_order = (
    features.groupby("Driver")["LapNumber"].max().sort_values(ascending=False).index.tolist()
)

# -----------------------------
#  Plot A: Stint timeline
# -----------------------------
def plot_stint_timeline(stints: pd.DataFrame, outpath: Path):
    """
    Horizontal timeline per driver:
      - y: driver
      - x: lap range of each stint
      - color: compound
    """
    if stints.empty:
        raise RuntimeError("stints_summary is empty — run 02_build_features.py first.")

    # Order drivers top-to-bottom
    drivers = sorted(stints["Driver"].unique(), key=lambda d: driver_order.index(d) if d in driver_order else 999)
    y_pos = {d: i for i, d in enumerate(drivers)}

    # Simple color map per compound (matplotlib defaults; keep names consistent)
    comp_colors = {
        "SOFT": None, "MEDIUM": None, "HARD": None,
        "INTERMEDIATE": None, "WET": None
    }

    fig, ax = plt.subplots(figsize=(12, 8))
    for _, r in stints.sort_values(["Driver","StintNo"]).iterrows():
        y = y_pos.get(r["Driver"], 0)
        x0, x1 = int(r["first_lap"]), int(r["last_lap"])
        label = str(r.get("Compound", "UNK"))
        ax.hlines(y, x0, x1, linewidth=8, label=label if (y==0 and r["StintNo"]==stints["StintNo"].min()) else None)

    # y-axis labels
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(drivers)
    ax.set_xlabel("Lap")
    ax.set_title("Montreal 2025 — Stint Timeline (color = compound)")
    ax.grid(True, axis="x", alpha=0.3)
    # Build a legend with unique compounds present
    present = [c for c in ["SOFT","MEDIUM","HARD","INTERMEDIATE","WET"] if c in stints["Compound"].astype(str).unique()]
    if present:
        # Draw invisible lines for legend entries
        for c in present:
            ax.plot([], [], label=c)
        ax.legend(ncol=len(present), frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

# -----------------------------
#  Pit-loss estimation
# -----------------------------
def estimate_pit_losses(raw_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate pit stop loss per driver using in-lap + out-lap vs surrounding pace.
    pit_loss ≈ (in_lap + out_lap) - 2 * median(pre/post clean laps)
    Also tag if the stop occurred under SC/VSC (either lap).
    """
    df = raw_laps.copy()
    # guard rails
    need = {"Driver","LapNumber","LapTimeSeconds","IsInLap","IsOutLap"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Raw laps missing: {missing}")

    results = []
    for drv, g in df.sort_values(["Driver","LapNumber"]).groupby("Driver"):
        g = g.reset_index(drop=True)
        # any lap with PitInTime is the in-lap; out-lap is next lap index if exists
        in_idx = g.index[g["IsInLap"].fillna(False)].tolist()
        for i in in_idx:
            in_row = g.iloc[i]
            # find next out-lap (usually immediate next row)
            out_row = None
            if i+1 < len(g):
                out_row = g.iloc[i+1]
            if out_row is None or not bool(out_row["IsOutLap"]):
                # try to locate the first next IsOutLap within +2 laps (failsafe)
                nxt = g.iloc[i+1:i+3]
                cand = nxt.index[nxt["IsOutLap"].fillna(False)].tolist()
                if cand:
                    out_row = g.loc[cand[0]]
            if out_row is None:
                continue  # skip weird cases

            in_time = in_row.get("LapTimeSeconds", np.nan)
            out_time = out_row.get("LapTimeSeconds", np.nan)
            if not np.isfinite(in_time) or not np.isfinite(out_time):
                continue

            in_lapnum = int(in_row["LapNumber"])
            out_lapnum = int(out_row["LapNumber"])

            # Build reference from nearby clean laps (exclude in/out, +/- 3 laps before and after)
            window_before = g[(g["LapNumber"] >= in_lapnum-4) & (g["LapNumber"] <= in_lapnum-1) & ~g["IsInLap"] & ~g["IsOutLap"]]
            window_after  = g[(g["LapNumber"] >= out_lapnum+1) & (g["LapNumber"] <= out_lapnum+4) & ~g["IsInLap"] & ~g["IsOutLap"]]
            ref = pd.concat([window_before["LapTimeSeconds"], window_after["LapTimeSeconds"]], ignore_index=True)
            if ref.empty:
                continue
            ref_med = float(ref.median())

            loss = (in_time + out_time) - 2.0 * ref_med

            # Tag SC/VSC if either lap had SC/VSC status
            sc_vsc = False
            for row in (in_row, out_row):
                status = str(row.get("TrackStatusLabel", ""))
                if status.startswith("SC") or status.startswith("VSC"):
                    sc_vsc = True
                    break

            results.append({
                "Driver": drv,
                "in_lap": in_lapnum,
                "out_lap": out_lapnum,
                "pit_loss_seconds": loss,
                "under_sc_or_vsc": sc_vsc
            })

    return pd.DataFrame(results)

def plot_pit_losses(pits_df: pd.DataFrame, outpath: Path):
    if pits_df.empty:
        raise RuntimeError("No pit stops detected to plot.")

    # Sort by driver; each driver can have multiple stops
    pits_df = pits_df.sort_values(["Driver", "in_lap"]).reset_index(drop=True)

    # Baselines: medians split by SC/VSC vs green
    base_green = pits_df.loc[~pits_df["under_sc_or_vsc"], "pit_loss_seconds"].median()
    base_sc    = pits_df.loc[pits_df["under_sc_or_vsc"],  "pit_loss_seconds"].median()

    # Build x labels like "RUS#1", "RUS#2"
    label_counts = pits_df.groupby("Driver").cumcount() + 1
    xlabels = [f"{d}#{n}" for d, n in zip(pits_df["Driver"], label_counts)]
    x = np.arange(len(xlabels))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, pits_df["pit_loss_seconds"].to_numpy())
    ax.axhline(base_green if np.isfinite(base_green) else 0.0, linewidth=1, linestyle="--", label="Median (Green)")
    if np.isfinite(base_sc):
        ax.axhline(base_sc, linewidth=1, linestyle=":", label="Median (SC/VSC)")

    # Mark SC/VSC stops
    for xi, is_sc in zip(x, pits_df["under_sc_or_vsc"]):
        if is_sc:
            ax.plot([xi, xi], [0, pits_df.loc[xi, "pit_loss_seconds"]], linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_ylabel("Estimated Pit Loss (s)")
    ax.set_title("Montreal 2025 — Pit Stop Loss vs Nearby Pace\n(Bar = in-lap + out-lap − 2×median of surrounding laps)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

# -----------------------------
# Run both plots
# -----------------------------
plot_stint_timeline(stints_summary, OUTDIR / "stint_timeline.png")
pits_est = estimate_pit_losses(raw)
pits_est.to_parquet(DATA / "pit_losses_estimated.parquet", index=False)
plot_pit_losses(pits_est, OUTDIR / "pit_loss.png")

print("Saved:", OUTDIR)
