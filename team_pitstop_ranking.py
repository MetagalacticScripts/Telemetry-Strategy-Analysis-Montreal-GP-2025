"""
Team Pit Stop Ranking – Montreal GP 2025
- Computes average stationary pit time per team.
- Primary source: FastF1 session.get_pit_stops() (uses your local cache).
- Fallback: rough estimate using laps.csv if FastF1 unavailable.
- Produces a broadcast-style horizontal bar chart.

Run:
  python team_pitstop_ranking.py
"""

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
YEAR = 2025
EVENT = "Canadian Grand Prix"   # Montreal
SESSION = "R"                   # Race

CACHE_DIR = Path("fastf1_cache")
DATA_DIR = Path("data_montreal_2025")
LAPS_CSV = DATA_DIR / "laps.csv"
RESULTS_CSV = DATA_DIR / "results.csv"

OUT_PNG = "fig_team_pitstop_ranking.png"

TEAM_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "Red Bull": "#1E41FF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#00A0DE",
    "RB": "#2231D5",              # VCARB / RB
    "Sauber": "#52E252",          # Stake F1 Team Kick Sauber (adjust to your year naming)
    "Haas": "#B6BABD",
}

# --------------------------------

def fmt_time(t):
    # seconds -> e.g. 2.36s
    if pd.isna(t):
        return ""
    return f"{t:.2f}s"

def load_pitstops_fastf1():
    """Try to load pit stops via FastF1 (preferred). Returns DataFrame with
    columns: Driver, TeamName, Duration(s)."""
    try:
        import fastf1
        # enable cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(CACHE_DIR.as_posix())

        session = fastf1.get_session(YEAR, EVENT, SESSION)
        session.load(telemetry=False, weather=False, laps=False)

        pstops = session.get_pit_stops()  # columns: Driver, Lap, Time, Duration, etc.
        if pstops is None or pstops.empty:
            return None

        # map drivers -> teams using session.results
        res = session.results
        team_map = {}
        if res is not None and not res.empty:
            for _, r in res.iterrows():
                abbr = r.get("Abbreviation")
                team = r.get("TeamName") or r.get("Team") or ""
                if pd.notna(abbr):
                    team_map[str(abbr)] = team

        # Duration is a pandas Timedelta; convert to seconds
        out = pstops.copy()
        if "Duration" in out.columns:
            out["Duration_s"] = pd.to_timedelta(out["Duration"]).dt.total_seconds()
        elif "Stop" in out.columns and "Time" in out.columns:
            # fallback heuristic if Duration missing
            out["Duration_s"] = pd.to_timedelta(out["Time"]).diff().dt.total_seconds()
        else:
            return None

        out["Driver"] = out["Driver"].astype(str)
        out["TeamName"] = out["Driver"].map(team_map).fillna("")
        # filter obviously-bad or missing durations
        out = out[(out["Duration_s"].notna()) & (out["Duration_s"] > 1.5) & (out["Duration_s"] < 20)]
        return out[["Driver", "TeamName", "Duration_s"]]
    except Exception as e:
        print(f"[info] FastF1 pit-stop table unavailable ({e}). Will try fallback from laps.csv.")
        return None

def fallback_from_laps():
    """Very rough pit loss estimate from laps.csv if FastF1 pit table not available.
    Uses in-lap + out-lap penalty vs. each driver’s median 'green' lap time."""
    if not LAPS_CSV.exists():
        print("[warn] laps.csv not found for fallback.")
        return None
    laps = pd.read_csv(LAPS_CSV)

    # Ensure time columns are usable
    if "LapTime" not in laps.columns:
        print("[warn] laps.csv has no LapTime column; cannot estimate pit stops.")
        return None

    if not pd.api.types.is_numeric_dtype(laps["LapTime"]):
        laps["LapTime"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

    # Try to infer team names from results.csv if available
    team_map = {}
    if RESULTS_CSV.exists():
        results = pd.read_csv(RESULTS_CSV)
        if "Abbreviation" in results.columns:
            for _, r in results.iterrows():
                abbr = r.get("Abbreviation")
                team = r.get("TeamName") or r.get("Team") or ""
                if pd.notna(abbr):
                    team_map[str(abbr)] = team

    # Heuristic: flag likely pit events using presence of PitInTime/PitOutTime, else big lap deltas
    has_pit_cols = ("PitInTime" in laps.columns) or ("PitOutTime" in laps.columns)
    est_rows = []

    if has_pit_cols:
        # If either column exists, mark in/out laps and compute penalty vs median green laps
        # Build median "clean" lap time per driver (exclude laps with pit markers if possible)
        laps["pit_marker"] = laps[["PitInTime", "PitOutTime"]].notna().any(axis=1) if "PitInTime" in laps.columns or "PitOutTime" in laps.columns else False
        med = laps[~laps["pit_marker"]].groupby("Driver")["LapTime"].median()

        for drv, grp in laps.groupby("Driver"):
            grp = grp.sort_values("LapNumber")
            # Roughly take any lap with a pit marker as "in" and the next lap as "out"
            pit_indices = grp.index[grp["pit_marker"]].tolist()
            for idx in pit_indices:
                ln = grp.loc[idx, "LapNumber"]
                # penalty = (in-lap + out-lap) - 2*median
                t_in = grp.loc[idx, "LapTime"]
                # out-lap is next real lap for driver
                nxt = grp[grp["LapNumber"] == ln + 1]
                if not nxt.empty:
                    t_out = nxt["LapTime"].iloc[0]
                else:
                    t_out = None
                baseline = med.get(drv, pd.NA)
                if pd.notna(baseline):
                    penalty = (t_in if pd.notna(t_in) else 0) + (t_out if pd.notna(t_out) else 0) - 2 * baseline
                    if pd.notna(penalty) and penalty > 3 and penalty < 30:
                        est_rows.append({
                            "Driver": drv,
                            "TeamName": team_map.get(drv, ""),
                            "Duration_s": float(penalty)
                        })
    else:
        print("[warn] No pit markers; using big lap deltas heuristic.")
        # Fallback-fallback: mark a "stop" whenever LapTime - rolling median > threshold
        est_rows = []
        for drv, grp in laps.groupby("Driver"):
            grp = grp.sort_values("LapNumber")
            base = grp["LapTime"].rolling(7, center=True, min_periods=3).median()
            excess = grp["LapTime"] - base
            # pick candidates with >8s excess (heuristic)
            candidates = grp[excess > 8]
            for _, row in candidates.iterrows():
                est_rows.append({
                    "Driver": drv,
                    "TeamName": team_map.get(drv, ""),
                    "Duration_s": float(excess.loc[row.name])
                })

    if not est_rows:
        return None
    return pd.DataFrame(est_rows)

def main():
    # Try FastF1 first
    pstops = load_pitstops_fastf1()

    if pstops is None or pstops.empty:
        pstops = fallback_from_laps()

    if pstops is None or pstops.empty:
        raise SystemExit("No pit stop data could be derived. Ensure FastF1 cache is present or laps.csv includes pit markers.")

    # Clean team names; fill unknowns as 'Unknown'
    pstops["TeamName"] = pstops["TeamName"].replace({None: "", pd.NA: ""}).fillna("")
    pstops.loc[pstops["TeamName"] == "", "TeamName"] = "Unknown"

    # Compute team averages
    team_avg = pstops.groupby("TeamName", as_index=False)["Duration_s"].mean()
    # Filter unrealistic values
    team_avg = team_avg[(team_avg["Duration_s"] > 1.5) & (team_avg["Duration_s"] < 20)]
    team_avg = team_avg.sort_values("Duration_s", ascending=True).reset_index(drop=True)

    # --- Plot ---
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 7))

    labels = team_avg["TeamName"].tolist()
    values = team_avg["Duration_s"].tolist()

    # Colors: team color or default steel blue; fastest highlighted gold
    colors = []
    for i, team in enumerate(labels):
        if i == 0:
            colors.append("#FFD700")  # highlight fastest team
        else:
            colors.append(TEAM_COLORS.get(team, "#4C78A8"))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=1.0)

    # Value labels
    xmax = max(values) if values else 0
    for b, v in zip(bars, values):
        ax.text(b.get_width() + 0.12, b.get_y() + b.get_height()/2,
                fmt_time(v), va="center", ha="left", fontsize=10, color="#EAEAEA")

    # Title & subtitle
    ax.set_title("Fastest Pit Crews – Average Stationary Time by Team\nMontreal GP 2025",
                 fontsize=16, weight="bold", pad=10)
    ax.set_xlabel("Average Pit Stop (s)")

    # Styling
    ax.invert_yaxis()  # fastest at top
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved -> {OUT_PNG}")

if __name__ == "__main__":
    main()
