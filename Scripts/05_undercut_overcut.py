# Scripts/05_undercut_overcut.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTDIR = ROOT / "Reports" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load data ----------------
features = pd.read_parquet(DATA / "laps_features_canada_2025.parquet")
raw = pd.read_parquet(DATA / "laps_canada_2025.parquet")

# Normalize a few fields on raw
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "LapTimeSeconds" not in df.columns and "LapTime" in df.columns:
        df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
    for c in ("Driver", "Compound"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.upper().str.strip()
    df["IsInLap"]  = df.get("PitInTime").notna()  if "PitInTime"  in df.columns else False
    df["IsOutLap"] = df.get("PitOutTime").notna() if "PitOutTime" in df.columns else False
    # keep only laps with a valid number & time
    df = df[df["LapNumber"].notna() & df["LapTimeSeconds"].notna()].copy()
    df["LapNumber"] = df["LapNumber"].astype(int)
    # cumulative elapsed time from race start per driver
    df = df.sort_values(["Driver","LapNumber"])
    df["Elapsed"] = df.groupby("Driver")["LapTimeSeconds"].cumsum()
    return df

raw = normalize(raw)

# Build a quick lookup: per driver, lap -> elapsed time
def make_elapsed_lookup(df: pd.DataFrame):
    d = {}
    for drv, g in df.groupby("Driver"):
        d[drv] = g.set_index("LapNumber")["Elapsed"]
    return d

elapsed = make_elapsed_lookup(raw)

# Extract pit events (in-lap and next out-lap)
def detect_pits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drv, g in df.sort_values(["Driver","LapNumber"]).groupby("Driver"):
        g = g.reset_index(drop=True)
        in_idxs = g.index[g["IsInLap"].fillna(False)].tolist()
        for i in in_idxs:
            in_lap = int(g.loc[i, "LapNumber"])
            out_lap = None
            # usually out-lap is next row; search up to +2 laps as fallback
            for j in range(i+1, min(i+4, len(g))):
                if bool(g.loc[j, "IsOutLap"]):
                    out_lap = int(g.loc[j, "LapNumber"])
                    break
            if out_lap is None:
                continue
            rows.append({"Driver": drv, "in_lap": in_lap, "out_lap": out_lap})
    return pd.DataFrame(rows)

pits = detect_pits(raw)

# Helper to get elapsed time at a driver's lap; returns np.nan if missing
def get_elapsed(drv: str, lap: int) -> float:
    s = elapsed.get(drv)
    if s is None:
        return np.nan
    val = s.get(lap, np.nan)
    try:
        return float(val)
    except Exception:
        return np.nan

# Compute undercut/overcut pairwise
pairs = []
drivers = sorted(raw["Driver"].unique())

# Rival proximity threshold (seconds) and pit window
NEAR_GAP = 5.0
PIT_WINDOW = 3  # laps

# Precompute per-lap field fastest to sanity-check after laps exist
max_lap = int(raw["LapNumber"].max()) if not raw.empty else 0

for _, stopA in pits.iterrows():
    A = stopA["Driver"]; L_in = int(stopA["in_lap"]); L_out = int(stopA["out_lap"])
    # gap BEFORE at lap L_in - 1
    lap_before = L_in - 1
    if lap_before < 1:
        continue
    A_before = get_elapsed(A, lap_before)
    if not np.isfinite(A_before):
        continue

    # find candidate rivals B who were near A on lap_before (within NEAR_GAP)
    for B in drivers:
        if B == A:
            continue
        B_before = get_elapsed(B, lap_before)
        if not np.isfinite(B_before):
            continue
        gap_before = A_before - B_before  # +ve => A ahead (smaller elapsed)
        if abs(gap_before) > NEAR_GAP:
            continue

        # does B pit within ±PIT_WINDOW laps of A's in-lap?
        pits_B = pits[pits["Driver"] == B]
        if pits_B.empty:
            continue
        # take the first B stop within window around A's stop
        cand = pits_B[(pits_B["in_lap"] >= L_in - PIT_WINDOW) & (pits_B["in_lap"] <= L_in + PIT_WINDOW)]
        if cand.empty:
            continue
        in_B = int(cand.iloc[0]["in_lap"])
        out_B = int(cand.iloc[0]["out_lap"])

        # AFTER: compare at lap L_ref = max(L_out, out_B) + 1 (both have rejoined, give 1 lap to stabilize)
        L_ref = max(L_out, out_B) + 1
        if L_ref > max_lap:
            continue

        A_after = get_elapsed(A, L_ref)
        B_after = get_elapsed(B, L_ref)
        if not (np.isfinite(A_after) and np.isfinite(B_after)):
            continue
        gap_after = A_after - B_after

        gain = gap_before - gap_after  # +ve: A gained vs B

        pairs.append({
            "A_driver": A,
            "B_rival": B,
            "A_in_lap": L_in,
            "B_in_lap": in_B,
            "compare_lap": L_ref,
            "gap_before_s": gap_before,
            "gap_after_s": gap_after,
            "gain_s": gain,
            "who_stopped_first": "A" if L_in <= in_B else "B"
        })

pairs_df = pd.DataFrame(pairs).drop_duplicates(subset=["A_driver","B_rival","A_in_lap","B_in_lap","compare_lap"])

# Save the table
out_table = DATA / "undercut_overcut_pairs.parquet"
pairs_df.to_parquet(out_table, index=False)

# -------- Plot: Top undercut gains (by first stopper) --------
def plot_undercut_gains(df: pd.DataFrame, outpath: Path, n=20):
    if df.empty:
        raise RuntimeError("No rival pairs found for undercut/overcut.")
    # Keep only cases where A stopped first so 'gain_s' is "undercut gain" for A
    sub = df[df["who_stopped_first"] == "A"].copy()
    if sub.empty:
        raise RuntimeError("No cases where the measured driver stopped first.")
    sub = sub.sort_values("gain_s", ascending=False).head(n)
    labels = [f"{a} vs {b} (L{int(la)}→L{int(lb)})" for a,b,la,lb in zip(sub["A_driver"], sub["B_rival"], sub["A_in_lap"], sub["B_in_lap"])]
    x = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x, sub["gain_s"].to_numpy())
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Undercut Gain (s)  — positive = first stopper gained")
    ax.set_title("Montreal 2025 — Biggest Undercut Gains (rival-matched)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

plot_undercut_gains(pairs_df, OUTDIR / "undercut_gains_top.png")

print("Saved:")
print(f"  {OUTDIR/'undercut_gains_top.png'}")
print(f"  {out_table}")
