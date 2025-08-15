# Scripts/07_fastlap_animation.py
from pathlib import Path
import re
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib import colormaps as cm

# -------- paths & config --------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_montreal_2025" / "telemetry_fastlaps_top5"
OUTDIR = ROOT / "Reports" / "media"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MP4 = OUTDIR / "anim_fastlaps.mp4"
OUT_GIF = OUTDIR / "anim_fastlaps.gif"

# leave [] to auto-pick up to 5 from folder; otherwise list driver codes like ["RUS","VER","LEC","PIA","ANT"]
DRIVERS_TO_SHOW = []

# handy colors for common codes; fallback uses Matplotlib cycle
COLOR_MAP = {
    "RUS": "#FFD700", "HAM": "#00D2BE",   # Merc
    "VER": "#1E41FF", "PER": "#3671C6",   # RBR
    "LEC": "#DC0000", "SAI": "#DC0000",   # Ferrari
    "NOR": "#FF8700", "PIA": "#FF8700",   # McLaren
    "ALO": "#2D826D", "STR": "#2D826D",   # AMR
    "OCO": "#0090FF","GAS": "#0090FF",    # Alpine
    "HUL": "#52E252","MAG": "#52E252",    # Haas
    "BOT": "#009AFA","ZHO": "#009AFA",    # Sauber/Kick
    "ALB": "#1E5BC6","SAR": "#1E5BC6",    # Williams
    "TSU": "#4E7FFF","RIC": "#4E7FFF",    # RB
    "ANT": "#FFD700"
}

# animation look
FPS_MP4 = 50
DPI_MP4 = 200
TRAIL_RATIO = 0.10   # visible trail ~10% of lap
POINT_SIZE = 9
LINE_WIDTH = 2.6

TITLE = "Fast Lap Animation — Montreal GP 2025"
SUBTITLE = "Synchronized by distance; trails show recent path"

# -------- helpers --------
def log(msg):
    print(f"[info] {msg}")

def find_files():
    log(f"Looking for CSVs in: {DATA_DIR}")
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {DATA_DIR}")
    return files

def infer_driver_code(path: Path) -> str:
    name = path.stem
    m = re.match(r"([A-Z]{3})[_-]", name.upper())
    return m.group(1) if m else name[:3].upper()

def load_fastlaps() -> dict[str, pd.DataFrame]:
    files = find_files()
    log(f"Found {len(files)} files")
    for f in files:
        log(f"       - {Path(f).name}")
    data = {}
    for fp in files:
        p = Path(fp)
        code = infer_driver_code(p)
        df = pd.read_csv(p)
        if "Distance" not in df.columns:
            raise ValueError(f"Missing 'Distance' in {p.name}")
        df = df[np.isfinite(df["Distance"])].copy()
        df = df.sort_values("Distance")
        data[code] = df
    log(f"Loaded drivers: {', '.join(data.keys())}")
    return data

def choose_drivers(data: dict, want: list[str]) -> list[str]:
    if want:
        return [d for d in want if d in data][:5]
    pref = ["RUS","VER","NOR","LEC","PIA","HAM","ALO","ANT"]
    order = [d for d in pref if d in data]
    order += [d for d in data.keys() if d not in order]
    return order[:5]

def resample_by_distance(df: pd.DataFrame, npts=700) -> pd.DataFrame:
    d = df["Distance"].to_numpy(dtype=float)
    good = np.isfinite(d)
    d = d[good]
    if d.size < 5:
        raise ValueError("Too few distance points to resample")
    grid = np.linspace(d.min(), d.max(), npts)

    out = {"Distance": grid}
    for col in df.columns:
        if col == "Distance":
            continue
        arr = df[col].to_numpy()[good]
        if np.issubdtype(np.array(arr).dtype, np.number):
            out[col] = np.interp(grid, d, arr.astype(float, copy=False))
    return pd.DataFrame(out)

def build_track_outline(sample_df: pd.DataFrame):
    if "X" in sample_df.columns and "Y" in sample_df.columns:
        x = sample_df["X"].to_numpy(dtype=float)
        y = sample_df["Y"].to_numpy(dtype=float)
        return x, y, True
    x = sample_df["Distance"].to_numpy(dtype=float)
    y = np.zeros_like(x)
    return x, y, False

# -------- main --------
def main():
    log(f"ROOT: {ROOT}")
    log(f"OUTDIR: {OUTDIR}")

    data = load_fastlaps()
    drivers = choose_drivers(data, DRIVERS_TO_SHOW)
    log(f"Drivers to animate: {', '.join(drivers)}")

    resampled = {}
    xy_outline = None
    uses_xy = False
    for d in drivers:
        df = resample_by_distance(data[d], npts=700)
        resampled[d] = df
        if xy_outline is None:
            x, y, flag_xy = build_track_outline(data[d])
            xy_outline = (x, y)
            uses_xy = flag_xy

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 8 if uses_xy else 5))

    # Draw track outline
    x0, y0 = xy_outline
    if uses_xy:
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

    # Set up artists
    artists_points = {}
    artists_trails = {}
    color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    for d in drivers:
        col = COLOR_MAP.get(d, next(color_cycle))
        df = resampled[d]
        if uses_xy and "X" in data[d].columns and "Y" in data[d].columns:
            Xi = np.interp(df["Distance"], data[d]["Distance"], data[d]["X"])
            Yi = np.interp(df["Distance"], data[d]["Distance"], data[d]["Y"])
            df["_X"], df["_Y"] = Xi, Yi
        else:
            df["_X"], df["_Y"] = df["Distance"], np.zeros_like(df["Distance"])
        resampled[d] = df

        p, = ax.plot([], [], marker="o", markersize=POINT_SIZE,
                     color=col, markeredgecolor="black", markeredgewidth=0.8,
                     zorder=6, label=d)
        t, = ax.plot([], [], linewidth=LINE_WIDTH, color=col, alpha=0.8, zorder=5)
        artists_points[d] = p
        artists_trails[d] = t

    title = ax.text(0.02, 0.98, TITLE, transform=ax.transAxes,
                    ha="left", va="top", fontsize=14, weight="bold")
    subtitle = ax.text(0.02, 0.94, SUBTITLE, transform=ax.transAxes,
                       ha="left", va="top", fontsize=10, alpha=0.7)
    legend = ax.legend(loc="lower right", frameon=False, title="Drivers")

    frames = len(next(iter(resampled.values())))
    trail_len = int(frames * TRAIL_RATIO)

    def update(i):
        for d in drivers:
            df = resampled[d]
            x = df["_X"].values
            y = df["_Y"].values
            artists_points[d].set_data([x[i]], [y[i]])
            j0 = max(0, i - trail_len)
            artists_trails[d].set_data(x[j0:i+1], y[j0:i+1])
        return list(artists_points.values()) + list(artists_trails.values()) + [title, subtitle, legend]

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

    # Try MP4 first
    try:
        writer = FFMpegWriter(fps=FPS_MP4, bitrate=4000)
        anim.save(OUT_MP4, writer=writer, dpi=DPI_MP4)
        log(f"Saved MP4 -> {OUT_MP4}")
    except Exception as e:
        log(f"ffmpeg failed ({e}); saving GIF instead…")
        writer = PillowWriter(fps=30)
        anim.save(OUT_GIF, writer=writer, dpi=160)
        log(f"Saved GIF -> {OUT_GIF}")

    plt.close(fig)

if __name__ == "__main__":
    main()
