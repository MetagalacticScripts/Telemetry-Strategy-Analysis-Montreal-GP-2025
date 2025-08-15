# Scripts/10_race_replay_animation.py
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection

# ------------------ CONFIG ------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_montreal_2025" / "telemetry_fastlaps_top5"  # change if needed
OUTDIR = ROOT / "Reports" / "media"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MP4 = OUTDIR / "race_replay.mp4"
OUT_GIF = OUTDIR / "race_replay.gif"

FPS = 40                 # animation frames per second
DURATION_S = None        # None = auto from longest driver trace
DPI = 200
FIGSIZE = (11, 7)

TRAIL_RATIO = 0.12       # fraction of lap/time window to show as trail
MARKER_SIZE = 10
LINE_WIDTH = 2.8
SHOW_GAP_PANEL = True    # toggle side panel showing live gap to leader
MAX_DRIVERS = 12         # safety cap

# Fallback colors (team-ish); anything missing gets auto cycle color
COLOR_MAP = {
    "RUS": "#00D2BE",  # Merc teal (you used gold earlier; feel free to swap)
    "ANT": "#FFD700",  # Antonelli gold
    "VER": "#1E41FF",  # RBR blue
    "LEC": "#DC0000",  # Ferrari red
    "SAI": "#DC0000",
    "NOR": "#FF8700",  # McLaren papaya
    "PIA": "#FF8700",
    "ALO": "#2D826D",
    "RIC": "#4E7FFF",
    "TSU": "#4E7FFF",
    "ALB": "#1E5BC6",
    "SAR": "#1E5BC6",
    "HUL": "#52E252",
    "MAG": "#52E252",
    "BOT": "#009AFA",
    "ZHO": "#009AFA",
    "PER": "#3671C6",
    "HAM": "#00D2BE",
}

TITLE = "Race Replay — Montreal 2025"
SUBTITLE = "Real X/Y positions, synchronized by time • right panel: live gap to leader"

# ------------------ HELPERS ------------------
def list_csvs():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {DATA_DIR}")
    return files

def infer_code(filename: str) -> str:
    name = Path(filename).stem
    m = re.match(r"([A-Za-z]{3})[_-]", name)
    return (m.group(1) if m else name[:3]).upper()

def parse_time_series(s: pd.Series) -> np.ndarray:
    """
    Return time in seconds starting at 0.
    Accepts either already-numeric seconds, or pandas Timedelta-like strings.
    """
    if np.issubdtype(s.dtype, np.number):
        t = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    else:
        # handle '0 days 00:00:00.088000' etc.
        tdelta = pd.to_timedelta(s, errors="coerce")
        t = tdelta.dt.total_seconds().to_numpy(dtype=float)
    # normalize start to 0 and drop non-finite
    if np.isfinite(t).any():
        t0 = np.nanmin(t[np.isfinite(t)])
        t = t - t0
    return t

def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

def load_driver_trace(fp: str):
    """Load a single driver CSV -> dict with time(s), X, Y, Distance; filters bad rows."""
    df = pd.read_csv(fp)
    needed = []
    if "Time" in df.columns:
        needed.append("Time")
    if "X" in df.columns and "Y" in df.columns:
        needed += ["X", "Y"]
    if "Distance" in df.columns:
        needed.append("Distance")
    if not needed:
        raise ValueError(f"{Path(fp).name}: missing required columns (need Time + X/Y or Distance).")

    # build arrays
    t = parse_time_series(df["Time"]) if "Time" in df.columns else None
    x = safe_numeric(df["X"]) if "X" in df.columns else None
    y = safe_numeric(df["Y"]) if "Y" in df.columns else None
    d = safe_numeric(df["Distance"]) if "Distance" in df.columns else None

    # coherence mask
    mask = np.ones(len(df), dtype=bool)
    if t is not None:
        mask &= np.isfinite(t)
    if x is not None:
        mask &= np.isfinite(x)
    if y is not None:
        mask &= np.isfinite(y)
    if d is not None:
        mask &= np.isfinite(d)

    # if we have X/Y missing but have Distance, we still allow (we'll fake XY later)
    if x is None or y is None:
        # keep mask only for available arrays
        mask = np.isfinite(t) if t is not None else mask

    # apply mask
    if t is not None: t = t[mask]
    if x is not None: x = x[mask]
    if y is not None: y = y[mask]
    if d is not None: d = d[mask]

    return {
        "time": t,
        "X": x,
        "Y": y,
        "Distance": d
    }

def resample_to_time(trace: dict, t_grid: np.ndarray):
    """Interpolate X/Y/Distance onto a common time grid."""
    t = trace["time"]
    if t is None or len(t) < 2:
        return None
    # ensure strictly increasing for interp
    good = np.isfinite(t)
    t = t[good]
    if len(t) < 2:
        return None

    out = {"time": t_grid}
    for key in ["X", "Y", "Distance"]:
        arr = trace.get(key)
        if arr is not None:
            arr = arr[np.isfinite(arr) & good[:len(arr)]] if len(arr) == len(good) else arr[good[:len(arr)]]
            # handle degenerate
            if np.isfinite(arr).sum() < 2:
                out[key] = np.full_like(t_grid, np.nan, dtype=float)
            else:
                # fill gaps at ends
                s = pd.Series(arr).ffill().bfill().to_numpy()
                out[key] = np.interp(t_grid, t, s)
        else:
            out[key] = np.full_like(t_grid, np.nan, dtype=float)
    return out

def build_track_outline(sample_xy: dict):
    """Return outline segments and bounds. If no XY, fallback to straight line by Distance."""
    x = sample_xy.get("X")
    y = sample_xy.get("Y")
    if x is not None and y is not None and np.isfinite(x).any() and np.isfinite(y).any():
        # slight smoothing for aesthetics
        df = pd.DataFrame({"x": x, "y": y})
        sm = df.rolling(7, center=True, min_periods=1).mean()
        xs, ys = sm["x"].to_numpy(), sm["y"].to_numpy()
        pts = np.column_stack((xs, ys))
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        return segs, (xs.min(), xs.max(), ys.min(), ys.max()), True
    else:
        # fallback to linear "track" using distance (not as pretty, but works)
        d = sample_xy.get("Distance")
        if d is None or not np.isfinite(d).any():
            # final fallback
            t = sample_xy["time"]
            d = np.linspace(0, 1, len(t))
        xs, ys = d, np.zeros_like(d)
        pts = np.column_stack((xs, ys))
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        return segs, (xs.min(), xs.max(), ys.min(), ys.max()), False

# ------------------ MAIN ------------------
def main():
    print(f"[info] Data dir: {DATA_DIR}")
    files = list_csvs()
    if len(files) > MAX_DRIVERS:
        print(f"[warn] Found {len(files)} files; capping to {MAX_DRIVERS}.")
        files = files[:MAX_DRIVERS]

    # Load all drivers
    raw = {}
    codes = []
    for fp in files:
        code = infer_code(fp)
        try:
            tr = load_driver_trace(fp)
        except Exception as e:
            print(f"[warn] Skipping {Path(fp).name} ({code}): {e}")
            continue
        if tr["time"] is None or len(tr["time"]) < 2:
            print(f"[warn] Skipping {code}: insufficient time data.")
            continue
        raw[code] = tr
        codes.append(code)

    if not raw:
        print("[error] No usable drivers found.")
        return

    # Common time grid
    max_t = max(np.nanmax(tr["time"]) for tr in raw.values())
    if DURATION_S is not None:
        max_t = min(max_t, float(DURATION_S))
    frames = max(100, int(max_t * FPS))
    t_grid = np.linspace(0.0, max_t, frames)

    # Resample everyone onto t_grid
    traces = {}
    for c in codes:
        traces[c] = resample_to_time(raw[c], t_grid)

    # Choose a sample for the outline (first with XY; else fallback)
    outline_source = None
    for c in codes:
        if np.isfinite(traces[c]["X"]).any() and np.isfinite(traces[c]["Y"]).any():
            outline_source = c
            break
    if outline_source is None:
        outline_source = codes[0]

    segs_outline, (xmin, xmax, ymin, ymax), has_xy = build_track_outline(traces[outline_source])
    print(f"[info] Outline from: {outline_source} (XY available: {has_xy})")

    # Prepare figure
    plt.style.use("dark_background")
    if SHOW_GAP_PANEL:
        fig = plt.figure(figsize=FIGSIZE)
        # grid spec: big track area + narrow gap panel
        gs = fig.add_gridspec(1, 5)
        ax_track = fig.add_subplot(gs[0, :4])
        ax_gap = fig.add_subplot(gs[0, 4])
    else:
        fig, ax_track = plt.subplots(figsize=FIGSIZE)
        ax_gap = None

    # Draw track outline
    lc = LineCollection(segs_outline, colors="#A9A9A9", linewidths=2.0, alpha=0.35)
    ax_track.add_collection(lc)
    if has_xy:
        ax_track.set_aspect("equal", adjustable="datalim")
        ax_track.set_xlim(xmin - 20, xmax + 20)
        ax_track.set_ylim(ymin - 20, ymax + 20)
        ax_track.axis("off")
    else:
        ax_track.autoscale()
        ax_track.set_yticks([])
        ax_track.set_xlabel("Distance (m)")
        ax_track.grid(alpha=0.15)

    # Artists (markers + trails)
    artists = {}
    trails = {}
    palette_iter = iter(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
    for c in codes:
        color = COLOR_MAP.get(c, next(palette_iter, "#1f77b4"))
        artists[c], = ax_track.plot([], [], marker="o", markersize=MARKER_SIZE,
                                    color=color, markeredgecolor="black", markeredgewidth=0.8, zorder=5, label=c)
        trails[c], = ax_track.plot([], [], linewidth=LINE_WIDTH, color=color, alpha=0.8, zorder=4)

    # Title + legend
    title = ax_track.text(0.02, 0.98, TITLE, transform=ax_track.transAxes,
                          ha="left", va="top", fontsize=14, weight="bold")
    subtitle = ax_track.text(0.02, 0.935, SUBTITLE, transform=ax_track.transAxes,
                             ha="left", va="top", fontsize=10, alpha=0.85)
    legend = ax_track.legend(loc="lower right", frameon=False, ncol=2)

    # Gap panel setup
    if SHOW_GAP_PANEL and ax_gap is not None:
        ax_gap.set_title("Gap to Leader (m)", fontsize=10)
        ax_gap.invert_yaxis()  # leader on top
        ax_gap.set_xlim(0, 50)  # will autoscale later
        ax_gap.set_xticks([0, 10, 20, 30, 40, 50])
        ax_gap.grid(alpha=0.15)
        gap_texts = {}

    # Trail length in frames
    trail_len = int(TRAIL_RATIO * frames)

    # Prepack arrays for quick access
    X = {c: traces[c]["X"] for c in codes}
    Y = {c: traces[c]["Y"] for c in codes}
    D = {c: traces[c]["Distance"] for c in codes}

    def update(frame_idx):
        # Prepare leader by "Distance" if available; else by arc-length along XY (approx)
        lead_metric = {}
        for c in codes:
            di = D[c][frame_idx]
            if np.isfinite(di):
                lead_metric[c] = di
            else:
                # approximate: cumulative XY distance from start
                xi = X[c][:frame_idx+1]
                yi = Y[c][:frame_idx+1]
                if np.isfinite(xi).sum() > 1 and np.isfinite(yi).sum() > 1:
                    dx = np.diff(xi[np.isfinite(xi)])
                    dy = np.diff(yi[np.isfinite(yi)])
                    lead_metric[c] = np.sum(np.hypot(dx, dy))
                else:
                    lead_metric[c] = np.nan

        # leader is max distance progressed
        leader = None
        if any(np.isfinite(v) for v in lead_metric.values()):
            leader = max((v, c) for c, v in lead_metric.items() if np.isfinite(v))[1]

        # Update markers and trails
        for c in codes:
            xi = X[c][frame_idx]
            yi = Y[c][frame_idx]
            # fallback to distance line if XY missing
            if not np.isfinite(xi) or not np.isfinite(yi):
                xi = D[c][frame_idx] if np.isfinite(D[c][frame_idx]) else np.nan
                yi = 0.0
            artists[c].set_data([xi], [yi])

            j0 = max(0, frame_idx - trail_len)
            xt = X[c][j0:frame_idx+1]
            yt = Y[c][j0:frame_idx+1]
            if not np.isfinite(xt).any() or not np.isfinite(yt).any():
                xt = D[c][j0:frame_idx+1]
                yt = np.zeros_like(xt)
            trails[c].set_data(xt, yt)

        # Update gap panel
        if SHOW_GAP_PANEL and ax_gap is not None:
            ax_gap.cla()
            ax_gap.set_title("Gap to Leader (m)", fontsize=10)
            if leader is None:
                # just list drivers
                y = np.arange(len(codes)) + 1
                ax_gap.barh(y, [0]*len(codes))
                ax_gap.set_yticks(y, labels=codes)
                ax_gap.set_xlim(0, 1)
            else:
                leader_d = lead_metric[leader]
                gaps = []
                labels = []
                colors = []
                for c in codes:
                    labels.append(c)
                    colors.append(COLOR_MAP.get(c, "#bbbbbb"))
                    di = lead_metric[c]
                    gap = float(leader_d - di) if np.isfinite(di) else np.nan
                    gaps.append(max(0.0, gap) if np.isfinite(gap) else 0.0)
                order = np.argsort(gaps)  # smallest gap (leader) first
                y = np.arange(len(codes)) + 1
                ax_gap.barh(y, np.array(gaps)[order], color=np.array(colors)[order])
                ax_gap.set_yticks(y, labels=np.array(labels)[order])
                xmax = max(10.0, np.nanmax(gaps) if np.isfinite(gaps).any() else 10.0)
                ax_gap.set_xlim(0, xmax * 1.1)
                ax_gap.grid(alpha=0.15)
                # add text labels
                for yi, gi in zip(y, np.array(gaps)[order]):
                    ax_gap.text(gi + xmax*0.02, yi, f"+{gi:0.1f}", va="center", fontsize=9)

        # time label
        seconds = t_grid[frame_idx]
        subtitle.set_text(f"{SUBTITLE}\nT+{seconds:0.2f}s")
        return list(artists.values()) + list(trails.values()) + [title, subtitle, legend]

    anim = FuncAnimation(
        fig, update, frames=frames, interval=1000/FPS, blit=False
    )

    # Save animation
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=5000)
        anim.save(OUT_MP4, writer=writer, dpi=DPI)
        print(f"[ok] Saved -> {OUT_MP4}")
    except Exception as e:
        print(f"[warn] ffmpeg unavailable or failed ({e}); saving GIF instead…")
        writer = PillowWriter(fps=FPS)
        anim.save(OUT_GIF, writer=writer, dpi=DPI)
        print(f"[ok] Saved -> {OUT_GIF}")

    plt.close(fig)

if __name__ == "__main__":
    main()
