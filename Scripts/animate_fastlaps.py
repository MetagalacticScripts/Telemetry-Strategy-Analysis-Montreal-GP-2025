"""
Animate Fast Laps on Track – Montreal GP 2025
- Loads all CSVs in data_montreal_2025/telemetry_fastlaps_top5/
- Synchronizes drivers by distance and animates markers around the track
- Saves MP4 (requires ffmpeg) and GIF (fallback via Pillow)
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection

DATA_DIR = Path("data_montreal_2025/telemetry_fastlaps_top5")
OUT_MP4 = "anim_fastlaps.mp4"
OUT_GIF = "anim_fastlaps.gif"

# Choose who to animate (leave empty to auto-pick up to 5 from folder)
DRIVERS_TO_SHOW = []  # e.g., ["RUS","NOR","VER","LEC","PIA"]

# Colors (feel free to tweak)
COLOR_MAP = {
    "RUS": "#FFD700",  # Russell gold
    "NOR": "#32CD32",  # Norris green
    "VER": "#1E41FF",  # Red Bull blue-ish
    "LEC": "#DC0000",  # Ferrari red
    "PIA": "#FF8700",  # McLaren papaya
}

def load_fastlap_csvs():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {DATA_DIR}")
    data = {}
    for f in files:
        key = Path(f).stem.split("_")[0]  # expects like RUS_fastlap.csv
        df = pd.read_csv(f)
        # Minimal sanity
        if "Distance" not in df.columns:
            raise ValueError(f"Missing 'Distance' in {f}")
        data[key] = df
    return data

def resample_by_distance(df, npts=500):
    """Resample telemetry to common distance grid (0..max) with linear interpolation."""
    d = df["Distance"].values
    # Ensure strictly increasing for interpolation
    good = np.isfinite(d)
    d = d[good]
    interp_cols = {}
    cols = []
    for col in df.columns:
        if col == "Distance":
            continue
        arr = df[col].values[good]
        if np.issubdtype(arr.dtype, np.number):
            cols.append(col)
            interp_cols[col] = np.interp
        else:
            # skip non-numerics for interpolation
            pass

    dmin, dmax = float(np.nanmin(d)), float(np.nanmax(d))
    grid = np.linspace(dmin, dmax, npts)

    out = {"Distance": grid}
    for col in cols:
        y = df.loc[good, col].values
        out[col] = np.interp(grid, d, y)
    return pd.DataFrame(out)

def build_track_outline(df):
    """Return (x, y) outline for background. Prefer X/Y; else use normalized 'Distance' as a line."""
    if "X" in df.columns and "Y" in df.columns:
        return df["X"].values, df["Y"].values, True
    # fallback: linear track
    x = df["Distance"].values
    y = np.zeros_like(x)
    return x, y, False

def main():
    data = load_fastlap_csvs()

    # Select drivers
    if DRIVERS_TO_SHOW:
        drivers = [d for d in DRIVERS_TO_SHOW if d in data][:5]
    else:
        # auto-pick up to 5 (keep RUS and NOR first if present)
        order = []
        for pref in ("RUS","NOR"):
            if pref in data and pref not in order:
                order.append(pref)
        for d in data.keys():
            if d not in order:
                order.append(d)
        drivers = order[:5]

    if not drivers:
        raise RuntimeError("No drivers selected.")

    # Resample all to same grid and collect XY
    resampled = {}
    xy_for_outline = None
    uses_xy = False

    for d in drivers:
        df = resample_by_distance(data[d], npts=600)
        resampled[d] = df
        if xy_for_outline is None:
            x, y, flag_xy = build_track_outline(data[d])
            # Smooth outline a bit for nice look
            if len(x) > 3 and flag_xy:
                # optional light smoothing using rolling mean
                s = pd.DataFrame({"x": x, "y": y}).rolling(5, center=True, min_periods=1).mean()
                x, y = s["x"].values, s["y"].values
            xy_for_outline = (x, y)
            uses_xy = flag_xy

    # Prepare figure
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8 if uses_xy else 5))

    # Draw track outline
    x0, y0 = xy_for_outline
    if uses_xy:
        # Create a faint track with slight gradient segments
        points = np.array([x0, y0]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, linewidths=2, alpha=0.35, colors="#AAAAAA")
        ax.add_collection(lc)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(x0.min()-20, x0.max()+20)
        ax.set_ylim(y0.min()-20, y0.max()+20)
        ax.axis("off")
    else:
        ax.plot(x0, y0, color="#AAAAAA", alpha=0.35)
        ax.set_xlabel("Distance (m)")
        ax.set_yticks([])
        ax.grid(alpha=0.15)

    # Create moving dots and trails
    artists = {}
    trails = {}
    for d in drivers:
        c = COLOR_MAP.get(d, "#1f77b4")
        df = resampled[d]
        # Precompute XY path for this driver
        if uses_xy and "X" in data[d].columns and "Y" in data[d].columns:
            # interpolate X,Y onto the resampled distance grid
            Xi = np.interp(df["Distance"], data[d]["Distance"], data[d]["X"])
            Yi = np.interp(df["Distance"], data[d]["Distance"], data[d]["Y"])
            df["_X"], df["_Y"] = Xi, Yi
        else:
            df["_X"], df["_Y"] = df["Distance"], np.zeros_like(df["Distance"])
        resampled[d] = df

        # scatter point
        p, = ax.plot([], [], marker="o", markersize=10, color=c, markeredgecolor="black", markeredgewidth=0.8, zorder=5, label=d)
        # trail (short recent path)
        t, = ax.plot([], [], linewidth=3, color=c, alpha=0.7, zorder=4)
        artists[d] = p
        trails[d] = t

    # Title/legend
    title = ax.text(0.02, 0.98, "Fast Lap Animation – Montreal GP 2025", transform=ax.transAxes,
                    ha="left", va="top", fontsize=14, weight="bold")
    legend = ax.legend(loc="lower right", frameon=False, title="Drivers")

    frames = len(next(iter(resampled.values())))  # num points on grid
    trail_len = int(frames * 0.08)  # ~8% of lap as visible trail

    def update(i):
        for d in drivers:
            df = resampled[d]
            x = df["_X"].values
            y = df["_Y"].values
            artists[d].set_data([x[i]], [y[i]])
            # trail window
            j0 = max(0, i - trail_len)
            trails[d].set_data(x[j0:i+1], y[j0:i+1])
        return list(artists.values()) + list(trails.values()) + [title, legend]

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

    # Try MP4 first
    try:
        writer = FFMpegWriter(fps=50, bitrate=4000)
        anim.save(OUT_MP4, writer=writer, dpi=200)
        print(f"Saved -> {OUT_MP4}")
    except Exception as e:
        print(f"ffmpeg unavailable or failed ({e}); saving GIF instead…")
        writer = PillowWriter(fps=30)
        anim.save(OUT_GIF, writer=writer, dpi=160)
        print(f"Saved -> {OUT_GIF}")

    plt.close(fig)

if __name__ == "__main__":
    main()
