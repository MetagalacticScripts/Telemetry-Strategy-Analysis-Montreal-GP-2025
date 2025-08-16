# Scripts/10_race_replay_animation.py
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# ------------------ CONFIG ------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_montreal_2025" / "telemetry_fastlaps_top5"  # change if needed
OUTDIR = ROOT / "Reports" / "media"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MP4 = OUTDIR / "race_replay.mp4"
OUT_GIF = OUTDIR / "race_replay.gif"

# Playback tuning
FPS = 40                   # animation frames per second
ANIM_SPEEDUP = 8.0         # >1 = faster than real-time (e.g., 8x compresses runtime)
DURATION_S = None          # None = use full race duration from data; else crop to this many seconds

DPI = 200
FIGSIZE = (12.5, 7.5)

TRAIL_RATIO = 0.10         # fraction of frames to show as trail (window length relative to total frames)
MARKER_SIZE = 9
LEADER_SIZE = 13
TRAIL_ALPHA = 0.35
TRAIL_WIDTH = 2.0
SHOW_GAP_PANEL = True      # toggle side panel showing live gap to leader
MAX_DRIVERS = 12           # safety cap
SHOW_LEADER_LABEL = True

# Fallback colors (team-ish); anything missing gets auto cycle color
COLOR_MAP = {
    "RUS": "#00D2BE",  # Merc teal
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
SUBTITLE = "Real X/Y positions synchronized by time • right panel: live gap to leader"

# Column name aliases for speed (we integrate to get whole-race progress & lap number)
SPEED_COLS = ["Speed", "SpeedKMH", "speed", "Velocity"]

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
    """Return time in seconds starting at 0."""
    if np.issubdtype(s.dtype, np.number):
        t = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    else:
        tdelta = pd.to_timedelta(s, errors="coerce")
        t = tdelta.dt.total_seconds().to_numpy(dtype=float)
    if np.isfinite(t).any():
        t0 = np.nanmin(t[np.isfinite(t)])
        t = t - t0
    return t

def safe_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

def _find_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _speed_mps(df: pd.DataFrame) -> np.ndarray:
    c = _find_first(df, SPEED_COLS)
    if c is None:
        return np.full(len(df), np.nan)
    v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(v).any():
        return np.full(len(df), np.nan)
    # heuristic: if typical speed > 60 it's probably km/h
    return v * ((1/3.6) if np.nanmedian(v) > 60 else 1.0)

def load_driver_trace(fp: str):
    """Load a single driver CSV -> dict with time(s), X, Y, Distance, Speed(m/s); filters bad rows."""
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

    t = parse_time_series(df["Time"]) if "Time" in df.columns else None
    x = safe_numeric(df["X"]) if "X" in df.columns else None
    y = safe_numeric(df["Y"]) if "Y" in df.columns else None
    d = safe_numeric(df["Distance"]) if "Distance" in df.columns else None
    v = _speed_mps(df)  # m/s (may contain NaN)

    mask = np.ones(len(df), dtype=bool)
    if t is not None: mask &= np.isfinite(t)
    if x is not None: mask &= np.isfinite(x)
    if y is not None: mask &= np.isfinite(y)
    if d is not None: mask &= np.isfinite(d)
    # We allow missing XY provided Time is valid
    if x is None or y is None:
        mask = np.isfinite(t) if t is not None else mask

    if t is not None: t = t[mask]
    if x is not None: x = x[mask]
    if y is not None: y = y[mask]
    if d is not None: d = d[mask]
    if v is not None: v = v[mask]

    return {"time": t, "X": x, "Y": y, "Distance": d, "Speed": v}

def resample_to_time(trace: dict, t_grid: np.ndarray):
    """Interpolate X/Y/Distance/Speed onto a common time grid."""
    t = trace["time"]
    if t is None or len(t) < 2:
        return None
    good = np.isfinite(t)
    t = t[good]
    if len(t) < 2:
        return None

    out = {"time": t_grid}
    for key in ["X", "Y", "Distance", "Speed"]:
        arr = trace.get(key)
        if arr is not None:
            # align length if needed
            if len(arr) != np.sum(good):
                arr = arr[:np.sum(good)]
            arr = arr[good[:len(arr)]]
            if np.isfinite(arr).sum() < 2:
                out[key] = np.full_like(t_grid, np.nan, dtype=float)
            else:
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
        df = pd.DataFrame({"x": x, "y": y}).rolling(7, center=True, min_periods=1).mean()
        xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
        pts = np.column_stack((xs, ys))
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        return segs, (xs.min(), xs.max(), ys.min(), ys.max()), True
    else:
        d = sample_xy.get("Distance")
        if d is None or not np.isfinite(d).any():
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

    # Real race duration from data
    max_t = max(np.nanmax(tr["time"]) for tr in raw.values())
    if DURATION_S is not None:
        max_t = min(max_t, float(DURATION_S))

    # Animation compression: fewer frames for the same real duration
    frames = max(120, int(max_t * FPS / ANIM_SPEEDUP))
    t_grid = np.linspace(0.0, max_t, frames)

    # Resample onto common grid
    traces = {c: resample_to_time(raw[c], t_grid) for c in codes}

    # Choose outline source
    outline_source = next((c for c in codes if
                           np.isfinite(traces[c]["X"]).any() and np.isfinite(traces[c]["Y"]).any()),
                          codes[0])
    segs_outline, (xmin, xmax, ymin, ymax), has_xy = build_track_outline(traces[outline_source])
    print(f"[info] Outline from: {outline_source} (XY available: {has_xy})")

    # Estimate track length (meters) for lap counting:
    # Prefer Distance column 99th percentile across drivers; fallback to XY scale.
    track_len_candidates = []
    for c in codes:
        d = traces[c].get("Distance")
        if d is not None and np.isfinite(d).any():
            track_len_candidates.append(np.nanpercentile(d, 99))
    if len(track_len_candidates) >= 1 and np.isfinite(track_len_candidates).any():
        TRACK_LEN = float(np.nanmedian(track_len_candidates))
    else:
        TRACK_LEN = float(np.hypot(xmax - xmin, ymax - ymin)) * 6.0  # heuristic fallback
    print(f"[info] Estimated track length ≈ {TRACK_LEN:0.1f} m")

    # Prepare figure (extra margins so texts don’t overlap track)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=FIGSIZE)

    if SHOW_GAP_PANEL:
        gs = fig.add_gridspec(1, 6, left=0.05, right=0.97, top=0.87, bottom=0.07, wspace=0.15)
        ax_track = fig.add_subplot(gs[0, :4])
        ax_gap   = fig.add_subplot(gs[0, 4:])
    else:
        gs = fig.add_gridspec(1, 1, left=0.06, right=0.97, top=0.87, bottom=0.08)
        ax_track = fig.add_subplot(gs[0, 0])
        ax_gap = None

    # Global title/subtitle placed in figure (not on the track axes)
    fig.text(0.05, 0.94, TITLE, fontsize=16, weight="bold", ha="left", va="center")
    subtitle_text = fig.text(0.05, 0.90, SUBTITLE, fontsize=10.5, ha="left", va="center", alpha=0.9)
    time_text     = fig.text(0.05, 0.865, "T+0.00s", fontsize=10.5, ha="left", va="center", alpha=0.9)
    lap_text      = fig.text(0.95, 0.94, "Lap 1", fontsize=14, weight="bold", ha="right", va="center")

    # Track outline
    lc = LineCollection(segs_outline, colors="#A9A9A9", linewidths=2.0, alpha=0.35, zorder=1)
    ax_track.add_collection(lc)
    if has_xy:
        ax_track.set_aspect("equal", adjustable="datalim")
        ax_track.set_xlim(xmin - 25, xmax + 25)
        ax_track.set_ylim(ymin - 25, ymax + 25)
        ax_track.axis("off")
    else:
        ax_track.autoscale()
        ax_track.set_yticks([])
        ax_track.set_xlabel("Distance (m)")
        ax_track.grid(alpha=0.15)

    # Artists (markers + trails)
    artists = {}
    halos = {}
    labels = {}
    trails = {}
    palette_iter = iter(plt.rcParams['axes.prop_cycle'].by_key().get('color', []))
    for c in codes:
        color = COLOR_MAP.get(c, next(palette_iter, "#1f77b4"))
        # trail underneath everything
        trails[c], = ax_track.plot([], [], linewidth=TRAIL_WIDTH, color=color,
                                   alpha=TRAIL_ALPHA, zorder=2, solid_capstyle="round")
        # marker
        artists[c], = ax_track.plot([], [], marker="o", markersize=MARKER_SIZE,
                                    color=color, markeredgecolor="black",
                                    markeredgewidth=0.8, zorder=5, label=c)
        # halo for leader (we’ll update which one is leader every frame)
        halos[c], = ax_track.plot([], [], marker="o", markersize=LEADER_SIZE+6,
                                  color="none", markeredgecolor=color,
                                  markeredgewidth=2.5, alpha=0.85, zorder=4)
        halos[c].set_visible(False)
        # leader label (hidden unless leader)
        labels[c] = ax_track.text(0, 0, "", fontsize=10, weight="bold",
                                  color=color, zorder=6,
                                  path_effects=[pe.withStroke(linewidth=2.5, foreground="black", alpha=0.7)])
        labels[c].set_visible(False)

    # Legend (outside lower center)
    fig.legend(handles=[artists[c] for c in codes], labels=codes,
               loc="lower center", ncol=min(6, len(codes)), frameon=False, bbox_to_anchor=(0.5, 0.01))

    # Gap panel setup
    if SHOW_GAP_PANEL and ax_gap is not None:
        ax_gap.set_title("Gap to Leader (m)", fontsize=11)
        ax_gap.invert_yaxis()  # leader on top
        ax_gap.set_xlim(0, 50)  # will autoscale later
        ax_gap.set_xticks([0, 10, 20, 30, 40, 50])
        ax_gap.grid(alpha=0.15)

    # Precompute cumulative distance for leader/gaps & lap counter
    # Use integrated speed (m/s) over time; fallback to Distance if speed missing.
    dt = np.diff(t_grid, prepend=t_grid[0])
    dt[0] = 0.0
    cum_dist = {}
    for c in codes:
        v = traces[c].get("Speed")
        if v is not None and np.isfinite(v).any():
            # Fill NaNs with series median to avoid stalls; then integrate
            v_series = pd.Series(v).fillna(pd.Series(v).median())
            cum = np.cumsum(v_series.to_numpy() * dt)
            cum_dist[c] = cum
        else:
            # Fallback: integrate Distance deltas (handling lap-wrap resets)
            d = traces[c].get("Distance")
            if d is not None and np.isfinite(d).any():
                dd = pd.Series(d).interpolate(limit_direction="both").to_numpy()
                diff = np.diff(dd, prepend=dd[0])
                # Ignore big negative jumps (lap counter reset)
                diff[diff < -TRACK_LEN * 0.5] = 0.0
                cum = np.cumsum(np.clip(diff, 0, None))
                cum_dist[c] = cum
            else:
                cum_dist[c] = np.zeros_like(t_grid)

    # Trail length in frames
    trail_len = int(TRAIL_RATIO * frames)

    X = {c: traces[c]["X"] for c in codes}
    Y = {c: traces[c]["Y"] for c in codes}
    D = {c: traces[c]["Distance"] for c in codes}

    def leader_by_progress(idx):
        """Pick leader by greatest cumulative progress (meters)."""
        best_c = None
        best_v = -np.inf
        for c in codes:
            v = float(cum_dist[c][idx]) if np.isfinite(cum_dist[c][idx]) else -np.inf
            if v > best_v:
                best_v, best_c = v, c
        return best_c, best_v

    def update(i):
        # who’s leading?
        leader, leader_val = leader_by_progress(i)

        # update markers and trails
        for c in codes:
            xi = X[c][i]
            yi = Y[c][i]
            if not np.isfinite(xi) or not np.isfinite(yi):
                # fallback: draw on distance line if XY missing
                di = D[c][i] if D[c] is not None else np.nan
                xi = di if np.isfinite(di) else np.nan
                yi = 0.0
            # marker
            artists[c].set_data([xi], [yi])
            artists[c].set_zorder(6 if c == leader else 5)
            artists[c].set_markersize(LEADER_SIZE if c == leader else MARKER_SIZE)

            # halo + leader label
            if c == leader:
                halos[c].set_data([xi], [yi])
                halos[c].set_visible(True)
                if SHOW_LEADER_LABEL:
                    labels[c].set_position((xi + (xmax-xmin)*0.01, yi + (ymax-ymin)*0.01))
                    labels[c].set_text(f"{c} (P1)")
                    labels[c].set_visible(True)
            else:
                halos[c].set_visible(False)
                labels[c].set_visible(False)

            # trail
            j0 = max(0, i - trail_len)
            xt = X[c][j0:i+1]; yt = Y[c][j0:i+1]
            if not np.isfinite(xt).any() or not np.isfinite(yt).any():
                # fallback to Distance line if XY bad
                if D[c] is not None:
                    xt = D[c][j0:i+1]; yt = np.zeros_like(xt)
                else:
                    xt = np.array([]); yt = np.array([])
            trails[c].set_data(xt, yt)
            trails[c].set_zorder(3)

        # gap panel
        if SHOW_GAP_PANEL and ax_gap is not None:
            ax_gap.cla()
            ax_gap.set_title("Gap to Leader (m)", fontsize=11)
            ax_gap.grid(alpha=0.15)
            if leader is None:
                y = np.arange(len(codes)) + 1
                ax_gap.barh(y, [0]*len(codes))
                ax_gap.set_yticks(y, labels=codes)
                ax_gap.set_xlim(0, 1)
            else:
                gaps = []
                labels_gap = []
                colors = []
                for c in codes:
                    labels_gap.append(c)
                    colors.append(COLOR_MAP.get(c, "#bbbbbb"))
                    di = float(cum_dist[c][i]) if np.isfinite(cum_dist[c][i]) else np.nan
                    gap = (leader_val - di) if np.isfinite(di) else np.nan
                    gaps.append(max(0.0, float(gap)) if np.isfinite(gap) else 0.0)

                order = np.argsort(gaps)  # leader first
                y = np.arange(len(codes)) + 1
                gaps_sorted = np.array(gaps)[order]
                colors_sorted = np.array(colors)[order]
                labels_sorted = np.array(labels_gap)[order]

                ax_gap.barh(y, gaps_sorted, color=colors_sorted, edgecolor="none")
                ax_gap.set_yticks(y, labels=labels_sorted)
                xmax_local = max(10.0, float(np.nanmax(gaps_sorted)) if np.isfinite(gaps_sorted).any() else 10.0)
                ax_gap.set_xlim(0, xmax_local * 1.15)

                # value labels
                for yy, gi in zip(y, gaps_sorted):
                    ax_gap.text(gi + xmax_local*0.02, yy, f"+{gi:0.1f}", va="center", fontsize=9)

                # highlight leader row with an outline
                ax_gap.barh(y[0], gaps_sorted[0], color=colors_sorted[0], edgecolor="white", linewidth=1.2)

        # time label
        time_text.set_text(f"T+{t_grid[i]:0.2f}s")

        # Lap number from leader’s cumulative meters
        lap_num = int(np.floor(leader_val / TRACK_LEN) + 1) if np.isfinite(leader_val) and TRACK_LEN > 0 else 1
        lap_text.set_text(f"Lap {lap_num}")

        # artists to re-draw
        draw_list = list(trails.values()) + list(halos.values()) + list(artists.values())
        draw_list += [time_text, subtitle_text, lap_text]
        return draw_list

    anim = FuncAnimation(fig, update, frames=frames, interval=1000/FPS, blit=False)

    # Save
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
