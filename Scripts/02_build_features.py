# Scripts/02_build_features.py
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"

IN_LAPS = {"INLAP", "IN LAP", "IN"}
OUT_LAPS = {"OUTLAP", "OUT LAP", "OUT"}

TRACK_STATUS_MAP = {
    "1": "GREEN",
    "2": "YELLOW",
    "3": "DOUBLE_YELLOW",
    "4": "SC",
    "5": "VSC",
    "6": "VSC_ENDING",
    "7": "SC_ENDING",
}

def load_sources():
    laps = pd.read_parquet(DATA_PROCESSED / "laps_canada_2025.parquet")
    pits = None
    stints = None
    p_p = DATA_PROCESSED / "pits_canada_2025.parquet"
    s_p = DATA_PROCESSED / "stints_canada_2025.parquet"
    if p_p.exists():
        pits = pd.read_parquet(p_p)
    if s_p.exists():
        stints = pd.read_parquet(s_p)
    return laps, pits, stints

def normalize_columns(df):
    df = df.copy()
    if "LapTimeSeconds" not in df.columns and "LapTime" in df.columns:
        try:
            df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
        except Exception:
            pass
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype(str).str.upper().str.strip()
    if "Driver" in df.columns:
        df["Driver"] = df["Driver"].astype(str).str.upper().str.strip()
    if "TrackStatus" in df.columns:
        df["TrackStatus"] = df["TrackStatus"].astype(str).str.strip()
        df["TrackStatusLabel"] = df["TrackStatus"].map(TRACK_STATUS_MAP).fillna("UNKNOWN")
    if "IsAccurate" in df.columns:
        df["IsAccurate"] = df["IsAccurate"].astype("boolean")
    if "LapNumber" in df.columns:
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").astype("Int64")
    return df

def mark_in_out_laps(df):
    df = df.copy()
    # robust defaults if cols are missing
    pit_in = df["PitInTime"] if "PitInTime" in df.columns else pd.Series(pd.NA, index=df.index)
    pit_out = df["PitOutTime"] if "PitOutTime" in df.columns else pd.Series(pd.NA, index=df.index)
    df["IsInLap"] = pd.notna(pit_in)
    df["IsOutLap"] = pd.notna(pit_out)
    if "LapType" in df.columns:
        laptype = df["LapType"].astype(str).str.upper().str.strip()
        df["IsInLap"] = df["IsInLap"] | laptype.isin(IN_LAPS)
        df["IsOutLap"] = df["IsOutLap"] | laptype.isin(OUT_LAPS)
    return df

def add_stint_and_tyre_age(df):
    df = df.copy()
    if "Stint" in df.columns and df["Stint"].notna().any():
        df["StintNo"] = pd.to_numeric(df["Stint"], errors="coerce").fillna(0).astype(int)
    else:
        df = df.sort_values(["Driver", "LapNumber"])
        df["StintNo"] = (df["IsOutLap"] & ~df["IsInLap"]).groupby(df["Driver"]).cumsum() + 1

    df = df.sort_values(["Driver", "LapNumber"])

    def tyre_age(group):
        out_flags = group["IsOutLap"].fillna(False).astype(bool).to_numpy()
        ages_list = [pd.NA] * len(group)
        current = -1
        for i, is_out in enumerate(out_flags):
            if is_out:
                current = 0
            else:
                if current >= 0:
                    current += 1
            ages_list[i] = current if current >= 0 else pd.NA
        return pd.Series(pd.array(ages_list, dtype="Int64"), index=group.index, name="TyreAgeLaps")

    df["TyreAgeLaps"] = df.groupby("Driver", group_keys=False).apply(tyre_age)
    return df

def filter_clean_race_laps(df):
    mask_valid = df["LapTimeSeconds"].notna()
    mask_io = ~(df["IsInLap"] | df["IsOutLap"])
    mask_green = df["TrackStatusLabel"].eq("GREEN") if "TrackStatusLabel" in df.columns else True
    mask_acc = df["IsAccurate"].fillna(True) if "IsAccurate" in df.columns else True
    return df[mask_valid & mask_io & mask_green & mask_acc].copy()

def add_pace_helpers(clean):
    df = clean.copy()
    df = df[df["LapNumber"].notna()].copy()

    # ✅ make this a Series, not a DataFrame
    lap_fastest = df.groupby("LapNumber")["LapTimeSeconds"].transform("min")
    df["DeltaToLapFastest"] = df["LapTimeSeconds"] - lap_fastest

    df = df.sort_values(["LapNumber", "LapTimeSeconds"])
    df["GapToNextOnSameLap"] = df.groupby("LapNumber")["LapTimeSeconds"].diff().fillna(pd.NA)

    df = df.sort_values(["Driver", "LapNumber"])
    df["DriverRollingMed_5"] = df.groupby("Driver")["LapTimeSeconds"].transform(
        lambda s: s.rolling(window=5, min_periods=3, center=True).median()
    )
    df["StintMedianSeconds"] = df.groupby(["Driver", "StintNo"])["LapTimeSeconds"].transform("median")
    df["DeltaToStintMedian"] = df["LapTimeSeconds"] - df["StintMedianSeconds"]
    return df

def _robust_slope(series):
    y = series.dropna().to_numpy()
    if len(y) < 3:
        return np.nan
    slopes = [(y[j] - y[i]) / (j - i) for i in range(len(y)) for j in range(i + 1, len(y)) if (j - i) != 0]
    return float(np.median(slopes)) if slopes else np.nan

def build_stint_summary(clean):
    return (
        clean.groupby(["Driver", "StintNo", "Compound"], dropna=False)
        .agg(
            laps_in_stint=("LapNumber", "count"),
            first_lap=("LapNumber", "min"),
            last_lap=("LapNumber", "max"),
            stint_median=("LapTimeSeconds", "median"),
            stint_best=("LapTimeSeconds", "min"),
            deg_per_lap=("LapTimeSeconds", _robust_slope),
        )
        .reset_index()
        .sort_values(["Driver", "StintNo"])
    )

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    laps, pits, stints = load_sources()
    laps = normalize_columns(laps)
    laps = mark_in_out_laps(laps)
    laps = add_stint_and_tyre_age(laps)

    clean = filter_clean_race_laps(laps)
    features = add_pace_helpers(clean)
    stint_summary = build_stint_summary(clean)

    features_path = DATA_PROCESSED / "laps_features_canada_2025.parquet"
    stintsum_path = DATA_PROCESSED / "stints_summary_canada_2025.parquet"
    features.to_parquet(features_path, index=False)
    stint_summary.to_parquet(stintsum_path, index=False)

    print("Saved:")
    print(f"  {features_path}")
    print(f"  {stintsum_path}")
    cols = [
        "Driver","LapNumber","Compound","StintNo","TyreAgeLaps",
        "LapTimeSeconds","DeltaToLapFastest","DriverRollingMed_5","DeltaToStintMedian"
    ]
    print(features[cols].head(12).to_string(index=False))

if __name__ == "__main__":
    main()
