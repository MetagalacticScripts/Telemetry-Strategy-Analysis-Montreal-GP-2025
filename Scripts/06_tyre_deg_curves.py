# Scripts/06_tyre_deg_curves.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTDIR = ROOT / "Reports" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA / "laps_features_canada_2025.parquet")

# --- Clean & types ---
# keep laps with tyre age + valid laptime
df = df[~df["TyreAgeLaps"].isna() & df["LapTimeSeconds"].notna()].copy()
# remove extreme outliers relative to stint median
df = df[df["DeltaToStintMedian"].abs() < 8.0].copy()

# Ensure numeric dtypes for fitting
df["TyreAgeLaps"] = pd.to_numeric(df["TyreAgeLaps"], errors="coerce").astype("float64")
df["LapTimeSeconds"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce").astype("float64")
df["Compound"] = df["Compound"].astype(str).str.upper().str.strip()

# Median lap time per compound & tyre age
med = (
    df.groupby(["Compound", "TyreAgeLaps"], dropna=False)["LapTimeSeconds"]
      .median()
      .reset_index()
      .dropna(subset=["TyreAgeLaps", "LapTimeSeconds"])
)

# Degradation slope per compound (linear fit over first 15 laps of tyre age)
deg_rows = []
for comp, g in med.groupby("Compound"):
    g15 = g[g["TyreAgeLaps"] <= 15].sort_values("TyreAgeLaps")
    x = g15["TyreAgeLaps"].to_numpy(dtype="float64")
    y = g15["LapTimeSeconds"].to_numpy(dtype="float64")
    if x.size < 2 or np.allclose(x, x[0]):
        # not enough or zero-variance data to fit
        continue
    slope, intercept = np.polyfit(x, y, 1)
    deg_rows.append({"Compound": comp, "deg_s_per_lap": float(slope), "intercept": float(intercept), "n_points": int(x.size)})

deg_df = pd.DataFrame(deg_rows)
deg_df.to_parquet(DATA / "tyre_deg_summary.parquet", index=False)

# -------- Plot curves --------
fig, ax = plt.subplots(figsize=(8, 6))
order = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
for comp in [c for c in order if c in med["Compound"].unique()] + \
            [c for c in med["Compound"].unique() if c not in order]:
    g = med[med["Compound"] == comp].sort_values("TyreAgeLaps")
    ax.plot(g["TyreAgeLaps"], g["LapTimeSeconds"], marker="o", label=comp)

ax.set_xlabel("Tyre Age (laps)")
ax.set_ylabel("Median Lap Time (s)")
ax.set_title("Montreal 2025 — Tyre Degradation Curves")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUTDIR / "tyre_deg_curves.png", dpi=160)
plt.close(fig)

print("Saved:")
print(f"  {OUTDIR/'tyre_deg_curves.png'}")
print(f"  {DATA/'tyre_deg_summary.parquet'}")
