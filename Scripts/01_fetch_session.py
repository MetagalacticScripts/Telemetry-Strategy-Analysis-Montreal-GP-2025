# scripts/01_fetch_session.py
import os
from pathlib import Path
import pandas as pd
import fastf1

YEAR = 2025
# Montreal is the Canadian GP
EVENT_NAME_CANDIDATES = ["Canadian Grand Prix", "Canada", "CanadianGP"]
SESSION = "R"  # Race

# data dirs
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / ".fastf1_cache"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---- enable cache ----
fastf1.Cache.enable_cache(CACHE_DIR.as_posix())

# ---- locate and load the session ----
session = None
for name in EVENT_NAME_CANDIDATES:
    try:
        session = fastf1.get_session(YEAR, name, SESSION)
        break
    except Exception:
        continue

if session is None:
    raise RuntimeError("Could not find the Canadian GP 2025 session. Try updating EVENT_NAME_CANDIDATES.")

print("Loading session… (this may take a few minutes the first time)")
session.load()  # downloads and parses timing + telemetry

# ---- extract core tables ----
laps = session.laps.reset_index(drop=True)
# helpful convenience columns
if "Driver" in laps.columns and "LapTime" in laps.columns:
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

# stints (per-driver tyre usage segments)
try:
    stints = session.laps[['Driver', 'Stint', 'Compound', 'LapNumber']].dropna()
    # convert to per-stint summary (first/last lap)
    stints = (
        stints.groupby(['Driver', 'Stint', 'Compound'])
        .agg(first_lap=('LapNumber', 'min'), last_lap=('LapNumber', 'max'))
        .reset_index()
        .sort_values(['Driver', 'Stint'])
    )
except KeyError:
    stints = pd.DataFrame()

# pit events
pits = session.laps[session.laps['PitOutTime'].notna() | session.laps['PitInTime'].notna()][
    ['Driver', 'LapNumber', 'PitInTime', 'PitOutTime']
].reset_index(drop=True)

# ---- save to disk (parquet is compact + fast) ----
laps_path = DATA_PROCESSED / "laps_canada_2025.parquet"
stints_path = DATA_PROCESSED / "stints_canada_2025.parquet"
pits_path = DATA_PROCESSED / "pits_canada_2025.parquet"

laps.to_parquet(laps_path, index=False)
if not stints.empty:
    stints.to_parquet(stints_path, index=False)
pits.to_parquet(pits_path, index=False)

print("Saved:")
print(f"  {laps_path}")
if not stints.empty:
    print(f"  {stints_path}")
print(f"  {pits_path}")

# ---- quick sanity print ----
print("\nSanity check:")
print(laps[['Driver', 'LapNumber', 'LapTime', 'Compound']].head(10))
