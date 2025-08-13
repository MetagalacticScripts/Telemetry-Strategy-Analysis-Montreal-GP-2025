from pathlib import Path
import pandas as pd
import fastf1

# ---- Config ----
YEAR   = 2025
EVENT  = "Canadian Grand Prix"   # Montreal
SESS   = "R"                     # Race
CACHE  = Path("fastf1_cache")
OUTDIR = Path("data_montreal_2025/telemetry_fastlaps_top5")

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE.as_posix())

    print(f"Loading {YEAR} {EVENT} ({SESS}) …")
    session = fastf1.get_session(YEAR, EVENT, SESS)
    # telemetry=True pulls car + position data so get_telemetry() has X/Y/Distance
    session.load(telemetry=True, laps=True, weather=False)
    print("Session loaded.")

    results = session.results.copy()
    if results is None or results.empty:
        raise RuntimeError("No session results available.")

    # Top 5 by finishing position (fallback: by laps)
    try:
        top5 = results.sort_values("Position").head(5)["Abbreviation"].tolist()
    except Exception:
        top5 = session.laps["Driver"].dropna().unique().tolist()[:5]

    exported = []
    for drv in top5:
        try:
            fl = session.laps.pick_driver(drv).pick_fastest()
            if fl is None or fl.empty:
                print(f"[{drv}] No fastest lap found, skipping.")
                continue

            # This returns merged car+pos telemetry with X,Y,Z and Distance (if available)
            tel = fl.get_telemetry()
            # Ensure 'Distance' exists; if not, approximate from speed*time
            if "Distance" not in tel.columns:
                if "Speed" in tel.columns and "Time" in tel.columns:
                    # compute cumulative distance in meters (Speed [km/h] -> m/s)
                    t = pd.to_timedelta(tel["Time"]).dt.total_seconds()
                    dt = t.diff().fillna(0.0)
                    ms = tel["Speed"].fillna(0.0) * (1000/3600.0)
                    tel["Distance"] = (ms * dt).cumsum()
                else:
                    print(f"[{drv}] No Distance and cannot derive (missing Speed/Time).")

            keep = [c for c in tel.columns if c in {
                "Time","Distance","Speed","Throttle","Brake","nGear","DRS","RPM","X","Y","Z","Source"
            }]
            tel_out = tel[keep].copy()

            out_path = OUTDIR / f"{drv}_fastlap.csv"
            tel_out.to_csv(out_path, index=False)
            exported.append(out_path.name)
            print(f"[{drv}] Saved -> {out_path}")
        except Exception as e:
            print(f"[{drv}] Export failed: {e}")

    if not exported:
        print("No telemetry exported. Check session/event naming or data availability.")
    else:
        print("Exported files:", ", ".join(exported))

if __name__ == "__main__":
    main()
