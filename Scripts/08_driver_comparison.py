# Scripts/08_driver_comparison.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
import glob

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_montreal_2025" / "telemetry_fastlaps_top5"
OUTDIR = ROOT / "Reports" / "media"
OUTDIR.mkdir(parents=True, exist_ok=True)

REF = "RUS"
DRIVERS = ["RUS", "VER", "ANT"]

COLOR_MAP = {
    "RUS": "#FFD700",  # gold
    "VER": "#1E41FF",  # blue
    "ANT": "#00D2BE"   # teal
}

FPS = 50
TRAIL_RATIO = 0.08
DPI = 200

# ------------- HELPERS -----------------
def find_file_for_driver(code: str) -> Path:
    files = glob.glob(str(DATA_DIR / f"{code}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV for driver {code} in {DATA_DIR}")
    return Path(sorted(files)[0])

def _parse_time_like(col: pd.Series) -> pd.Series | None:
    """Try to coerce a column to seconds as float."""
    # Already numeric?
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce").astype(float)
    # Try to parse time-like strings (e.g., '0:01:12.345')
    try:
        td = pd.to_timedelta(col, errors="coerce")
        if td.notna().any():
            return td.dt.total_seconds()
    except Exception:
        pass
    return None

def _cumtime_from_speed(distance_m: np.ndarray, speed: np.ndarray | None) -> np.ndarray:
    """Build cumulative time (s) from distance and speed.
       - Detect km/h vs m/s
       - Robust to small gaps/NaNs (forward fill, clamp minimum speed)
    """
    n = len(distance_m)
    if speed is None or np.all(~np.isfinite(speed)):
        # Fallback: assume constant average speed -> proportional to distance
        avg_v = (distance_m[-1] - distance_m[0]) / n if n > 1 else 60.0
        v = np.full(n, max(avg_v, 1.0))
    else:
        v = speed.astype(float)
        # Detect units: if typical > 100, likely km/h
        scale = 1/3.6 if np.nanmedian(v) > 60 else 1.0  # km/h -> m/s else assume m/s
        v = v * scale
        # Fill bad values
        if np.any(~np.isfinite(v)):
            # forward fill then back fill
            s = pd.Series(v)
            v = s.ffill().bfill().to_numpy()
        v = np.clip(v, 1.0, None)  # avoid divide by zero

    # integrate dt = ds / v  (use mid-point speeds)
    s = distance_m.astype(float)
    s = np.nan_to_num(s, nan=np.nanmin(s))
    ds = np.diff(s, prepend=s[0])
    # smooth speeds a little
    v_smooth = pd.Series(v).rolling(5, center=True, min_periods=1).mean().to_numpy()
    dt = ds / v_smooth
    t = np.cumsum(dt)
    t -= t[0]
    return t

def load_driver_data(code: str) -> pd.DataFrame:
    path = find_file_for_driver(code)
    df = pd.read_csv(path)
    if "Distance" not in df.columns:
        raise ValueError(f"Missing 'Distance' in {path.name}")
    df = df[np.isfinite(df["Distance"])].copy()
    df = df.sort_values("Distance")

    # Build/locate a cumulative time column "CumTime"
    time_cols = ["LapTime", "Time", "Elapsed", "CumTime", "SessionTime"]
    cum = None
    for c in time_cols:
        if c in df.columns:
            cand = _parse_time_like(df[c])
            if cand is not None and cand.notna().sum() >= max(10, len(df)//10):
                cum = cand.to_numpy()
                # If SessionTime is absolute, normalize to start at 0
                if c == "SessionTime":
                    cum = cum - np.nanmin(cum)
                break
    if cum is None:
        # Build from speed
        speed = None
        for c in ["Speed", "SpeedKMH", "Speed_mps", "speed"]:
            if c in df.columns:
                speed = pd.to_numeric(df[c], errors="coerce").to_numpy()
                break
        cum = _cumtime_from_speed(df["Distance"].to_numpy(), speed)
    df["CumTime"] = cum
    return df

def resample_to_grid(df: pd.DataFrame, npts=800) -> pd.DataFrame:
    d = df["Distance"].to_numpy(dtype=float)
    grid = np.linspace(d.min(), d.max(), npts)
    out = {"Distance": grid}

    # Interpolate known numeric columns of interest
    for col in df.columns:
        if col in ["Distance"]:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        out[col] = np.interp(grid, d, s.fillna(method="ffill").fillna(method="bfill"))

    # If X/Y exist but were non-numeric dtype, try to coerce separately
    for col in ["X", "Y"]:
        if col in df.columns and col not in out:
            try:
                s = pd.to_numeric(df[col], errors="coerce")
                out[col] = np.interp(grid, d, s.fillna(method="ffill").fillna(method="bfill"))
            except Exception:
                pass
    return pd.DataFrame(out)

def build_track_outline(df: pd.DataFrame):
    if "X" in df.columns and "Y" in df.columns and np.isfinite(df["X"]).any() and np.isfinite(df["Y"]).any():
        x = pd.to_numeric(df["X"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(float)
        return x, y, True
    x = df["Distance"].to_numpy(float)
    y = np.zeros_like(x)
    return x, y, False

# ------------- MAIN -----------------
def main():
    print(f"[info] Data dir: {DATA_DIR}")
    data_raw = {drv: load_driver_data(drv) for drv in DRIVERS}
    print(f"[info] Loaded drivers: {', '.join(data_raw.keys())}")

    # Resample all to a common distance grid
    npts = 900
    resampled = {drv: resample_to_grid(df, npts=npts) for drv, df in data_raw.items()}

    # ----- Delta time plot (vs REF) -----
    ref_df = resampled[REF]
    fig, ax = plt.subplots(figsize=(10, 5))
    for drv in DRIVERS:
        if drv == REF:
            continue
        delta = resampled[drv]["CumTime"].to_numpy() - ref_df["CumTime"].to_numpy()
        ax.plot(ref_df["Distance"], delta, color=COLOR_MAP.get(drv, "white"), label=f"{drv} vs {REF}")
    ax.axhline(0, color="white", lw=1)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Δ Time (s) vs " + REF)
    ax.set_title(f"Delta Time vs {REF} — Montreal GP 2025 (Fast Laps)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    delta_png = OUTDIR / "delta_time_plot.png"
    fig.savefig(delta_png, dpi=DPI)
    plt.close(fig)
    print(f"[info] Saved delta time plot -> {delta_png}")

    # ----- Animation: side-by-side-by-side -----
    x0, y0, use_xy = build_track_outline(data_raw[REF])
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8 if use_xy else 5))

    if use_xy:
        points = np.array([x0, y0]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, linewidths=2, alpha=0.35, colors="#AAAAAA")
        ax.add_collection(lc)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(x0.min() - 20, x0.max() + 20)
        ax.set_ylim(y0.min() - 20, y0.max() + 20)
        ax.axis("off")
    else:
        ax.plot(x0, y0, color="#AAAAAA", alpha=0.35)
        ax.set_xlabel("Distance (m)")
        ax.set_yticks([])
        ax.grid(alpha=0.15)

    artists, trails = {}, {}
    frames = len(next(iter(resampled.values())))
    trail_len = max(4, int(frames * TRAIL_RATIO))

    # Prepare positions (interpolate XY if available)
    for drv in DRIVERS:
        df_r = resampled[drv]
        if use_xy and "X" in data_raw[drv].columns and "Y" in data_raw[drv].columns:
            Xi = np.interp(df_r["Distance"], data_raw[drv]["Distance"], pd.to_numeric(data_raw[drv]["X"], errors="coerce"))
            Yi = np.interp(df_r["Distance"], data_raw[drv]["Distance"], pd.to_numeric(data_raw[drv]["Y"], errors="coerce"))
            df_r["_X"], df_r["_Y"] = Xi, Yi
        else:
            df_r["_X"], df_r["_Y"] = df_r["Distance"], np.zeros_like(df_r["Distance"])
        resampled[drv] = df_r

        p, = ax.plot([], [], "o", markersize=10, color=COLOR_MAP.get(drv, "white"),
                     markeredgecolor="black", markeredgewidth=0.8, zorder=5, label=drv)
        t, = ax.plot([], [], linewidth=3, color=COLOR_MAP.get(drv, "white"), alpha=0.7, zorder=4)
        artists[drv], trails[drv] = p, t

    title = ax.text(0.02, 0.98, f"Russell vs Verstappen vs Antonelli — Montreal GP 2025 (Fast Laps)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=13, weight="bold")
    subtitle = ax.text(0.02, 0.94, "Synced by distance | live Δ vs Russell", transform=ax.transAxes,
                       ha="left", va="top", fontsize=10, alpha=0.85)
    delta_text = ax.text(0.98, 0.02, "", transform=ax.transAxes,
                         ha="right", va="bottom", fontsize=11)

    def update(i):
        for drv in DRIVERS:
            df_r = resampled[drv]
            x = df_r["_X"].to_numpy()
            y = df_r["_Y"].to_numpy()
            artists[drv].set_data([x[i]], [y[i]])
            j0 = max(0, i - trail_len)
            trails[drv].set_data(x[j0:i+1], y[j0:i+1])

        # Live delta clocks vs REF using CumTime
        deltas = []
        for drv in DRIVERS:
            if drv == REF:
                continue
            dtime = resampled[drv]["CumTime"].iloc[i] - resampled[REF]["CumTime"].iloc[i]
            deltas.append(f"{drv}: {dtime:+.2f}s")
        delta_text.set_text("\n".join(deltas))
        return list(artists.values()) + list(trails.values()) + [title, subtitle, delta_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=True)

    out_mp4 = OUTDIR / "driver_comparison.mp4"
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=4000)
        anim.save(out_mp4, writer=writer, dpi=DPI)
        print(f"[info] Saved animation -> {out_mp4}")
    except Exception as e:
        out_gif = OUTDIR / "driver_comparison.gif"
        writer = PillowWriter(fps=30)
        anim.save(out_gif, writer=writer, dpi=160)
        print(f"[warn] ffmpeg failed: {e}\n[info] Saved GIF -> {out_gif}")

    plt.close(fig)

if __name__ == "__main__":
    main()
