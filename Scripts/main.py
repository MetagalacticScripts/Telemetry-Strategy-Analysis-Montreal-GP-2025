# save as fetch_montreal_2025.py
# Usage: python fetch_montreal_2025.py

import pathlib
import pandas as pd
import fastf1

# --- Settings ---
YEAR = 2025
EVENT = "Canadian Grand Prix"   # Montreal
SESSION = "R"                   # Race
OUTDIR = pathlib.Path("data_montreal_2025")
CACHE_DIR = pathlib.Path("fastf1_cache")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR.as_posix())

    print(f"Loading {YEAR} {EVENT} ({SESSION}) …")
    session = fastf1.get_session(YEAR, EVENT, SESSION)
    # laps=True and weather=True pull the majority of what you’ll need
    session.load(laps=True, telemetry=False, weather=True)
    print("Session loaded.")

    # --- Laps ---
    laps = session.laps.copy()
    laps.reset_index(drop=True, inplace=True)
    # Convert timedeltas to seconds for easier plotting later
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[col + "_s"] = laps[col].dt.total_seconds()
    laps.to_csv(OUTDIR / "laps.csv", index=False)
    print(f"Saved laps -> {OUTDIR/'laps.csv'}")

    # --- Results ---
    results = session.results.copy()
    results.reset_index(drop=True, inplace=True)
    results.to_csv(OUTDIR / "results.csv", index=False)
    print(f"Saved results -> {OUTDIR/'results.csv'}")

    # --- Weather ---
    if session.weather_data is not None:
        weather = session.weather_data.copy()
        weather.reset_index(drop=True, inplace=True)
        weather.to_csv(OUTDIR / "weather.csv", index=False)
        print(f"Saved weather -> {OUTDIR/'weather.csv'}")
    else:
        print("No weather data available for this session.")

    # --- Fastest-lap telemetry for top 5 finishers (lightweight example) ---
    # Full per-lap telemetry for every driver is very large; start small.
    try:
        top5 = results.sort_values("Position").head(5)["Abbreviation"].tolist()
    except Exception:
        # Fallback if Position not present
        top5 = laps["Driver"].dropna().unique().tolist()[:5]

    tel_dir = OUTDIR / "telemetry_fastlaps_top5"
    tel_dir.mkdir(exist_ok=True)

    for drv in top5:
        try:
            fastest_lap = laps.pick_driver(drv).pick_fastest()
            if fastest_lap is None or fastest_lap.empty:
                print(f"[{drv}] No fastest lap found.")
                continue
            tel = fastest_lap.get_telemetry()  # time series along lap distance
            # Keep commonly used channels only
            keep_cols = [c for c in tel.columns if c in {
                "Time", "Distance", "Speed", "Throttle", "Brake",
                "nGear", "DRS", "RPM", "Source", "X", "Y", "Z"
            }]
            tel_out = tel[keep_cols].copy()
            tel_out.to_csv(tel_dir / f"{drv}_fastlap.csv", index=False)
            print(f"Saved fastest-lap telemetry -> {tel_dir / f'{drv}_fastlap.csv'}")
        except Exception as e:
            print(f"[{drv}] Telemetry export skipped ({e})")

    print("Done.")

if __name__ == "__main__":
    main()
