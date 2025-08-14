import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

LAPS_CSV = "data_montreal_2025/laps.csv"
RESULTS_CSV = "data_montreal_2025/results.csv"
OUT = "fig_fastest_lap_delta.png"

# --- load ---
laps = pd.read_csv(LAPS_CSV)
results = pd.read_csv(RESULTS_CSV)

# make sure LapTime is numeric (seconds)
if not pd.api.types.is_numeric_dtype(laps["LapTime"]):
    laps["LapTime"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

# fastest per driver (use 3-letter code in laps["Driver"])
fastest = laps.dropna(subset=["LapTime"]).groupby("Driver", as_index=False)["LapTime"].min()

# attach names/teams if available in results
name_map = {}
team_map = {}
if "Abbreviation" in results.columns:
    for _, r in results.iterrows():
        if pd.notna(r.get("Abbreviation")):
            name_map[r["Abbreviation"]] = r.get("FullName", r.get("BroadcastName", r["Abbreviation"]))
            team_map[r["Abbreviation"]] = r.get("TeamName", r.get("Team", ""))
fastest["Name"] = fastest["Driver"].map(lambda d: name_map.get(d, d))
fastest["Team"] = fastest["Driver"].map(lambda d: team_map.get(d, ""))

# sort by absolute time and compute delta to P1
fastest = fastest.sort_values("LapTime", ascending=True).reset_index(drop=True)
p1_time = fastest.loc[0, "LapTime"]
fastest["DeltaToP1"] = fastest["LapTime"] - p1_time

# simple F1-ish team colors (fallback gray)
TEAM_COLORS = defaultdict(lambda: "#5F6A6A", {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "Red Bull": "#1E41FF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#00A0DE",
    "RB": "#2231D5",
    "Sauber": "#52E252",
    "Haas": "#B6BABD",
    "AlphaTauri": "#2B4562",
    "Aston Martin Aramco": "#006F62",
    "Scuderia Ferrari": "#DC0000",
    "Oracle Red Bull Racing": "#1E41FF",
    "MoneyGram Haas F1 Team": "#B6BABD",
})

def fmt_time(sec):
    import math
    m = int(sec // 60)
    s = sec - 60*m
    return f"{m:02d}:{s:06.3f}"

fastest["FastestLapStr"] = fastest["LapTime"].apply(fmt_time)
fastest["DeltaStr"] = fastest["DeltaToP1"].apply(lambda x: f"+{x:.3f}s" if x>1e-6 else "P1")

# highlight George Russell (RUS) in gold
def bar_color(row):
    if row["Driver"] in ("RUS",) or "Russell" in str(row["Name"]):
        return "#FFD700"
    return TEAM_COLORS[row["Team"]] if row["Team"] else "#4682B4"

colors = fastest.apply(bar_color, axis=1)

# --- plot: horizontal bars sorted fastest (top) to slowest (bottom) ---
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(fastest["Name"], fastest["LapTime"], color=colors, edgecolor="white", linewidth=1.1)

# annotate inside bars with absolute time; outside right with delta
for bar, t_str, d_str in zip(bars, fastest["FastestLapStr"], fastest["DeltaStr"]):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height()/2
    ax.text(x - 0.15, y, t_str, va="center", ha="right", fontsize=10, fontweight="bold", color="black")
    ax.text(x + 0.25, y, d_str, va="center", ha="left", fontsize=10, color="#E5E7E9")

# titles & styling
ax.set_title("Fastest Lap – Delta to P1 (Montreal GP 2025)", fontsize=16, weight="bold", pad=12)
ax.set_xlabel("Lap Time (s)")
ax.invert_yaxis()                      # fastest at top
ax.grid(axis="x", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved -> {OUT}")
