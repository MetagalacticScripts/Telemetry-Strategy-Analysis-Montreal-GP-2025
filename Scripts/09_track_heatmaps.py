# Scripts/09_track_heatmaps.py
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib import colormaps as cm

# ---------------- Config ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_montreal_2025" / "telemetry_fastlaps_top5"
OUTDIR = ROOT / "Reports" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Which metric(s) to color by: any of ["Speed", "Throttle", "Gear"]
METRICS = ["Speed", "Throttle", "Gear"]

# Which drivers to render (empty = auto from folder, up to 6)
DRIVERS = []  # e.g., ["RUS","VER","ANT","LEC","PIA"]

# Smoothing window (odd int) for the XY outline (for nicer curves)
SMOOTH_WIN = 7

# Downsample segments to keep file sizes & render time sane
MAX_SEGMENTS = 1600

# Colormaps per metric (Matplotlib names)
CMAP = {
    "Speed": "viridis",
    "Throttle": "plasma",
    "Gear": "turbo",
}

# Value ranges (None = auto from 5th..95th percentiles)
RANGE = {
    "Speed": None,      # e.g., (200, 340) if Speed in km/h
    "Throttle": (0, 100),
    "Gear": None        # e.g., (1, 8)
}

FIGSIZE = (8, 8)
DPI = 200

# -------------- Helpers ---------------
def _list_csvs():
    files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    return files

def _infer_driver(path: Path) -> str:
    name = path.stem.upper()
    if "_" in name:
        return name.split("_", 1)[0][:3]
    return name[:3]

def _load_driver_csv(code: str) -> pd.DataFrame:
    matches = sorted(glob.glob(str(DATA_DIR / f"{code}_*.csv")))
    if not matches:
        raise FileNotFoundError(f"No CSV for driver {code} in {DATA_DIR}")
    df = pd.read_csv(matches[0])
    if "Distance" not in df.columns:
        df["Distance"] = np.arange(len(df))
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def _smooth_xy(x, y, win=7):
    if win < 3 or win % 2 == 0:
        return x, y
    s = (
        pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")})
        .rolling(win, center=True, min_periods=1).mean()
    )
    return s["x"].to_numpy(), s["y"].to_numpy()

def _prepare_xy(df: pd.DataFrame):
    # Prefer X/Y, fallback to "linear track" using distance
    if {"X", "Y"}.issubset(df.columns):
        x = pd.to_numeric(df["X"], errors="coerce").to_numpy()
        y = pd.to_numeric(df["Y"], errors="coerce").to_numpy()
        use_xy = True
    else:
        x = pd.to_numeric(df["Distance"], errors="coerce").to_numpy()
        y = np.zeros_like(x)
        use_xy = False
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], use_xy

def _downsample_pairs(x, y, v, max_segments=1600):
    n = len(x)
    if n < 2:
        return None, None
    pts = np.column_stack((x, y))
    segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    cvals = v[:-1]
    if segs.shape[0] > max_segments:
        idx = np.linspace(0, segs.shape[0]-1, max_segments, dtype=int)
        segs = segs[idx]
        cvals = cvals[idx]
    return segs, cvals

def _norm_for_metric(values: np.ndarray, metric: str):
    rng = RANGE.get(metric)
    if rng and all(np.isfinite(rng)):
        vmin, vmax = rng
    else:
        vmin = np.nanpercentile(values, 5)
        vmax = np.nanpercentile(values, 95)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
    return Normalize(vmin=vmin, vmax=vmax, clip=True)

def _metric_series(df: pd.DataFrame, metric: str) -> np.ndarray:
    aliases = {
        "Speed": ["Speed", "SpeedKMH", "speed"],
        "Throttle": ["Throttle", "ThrottlePedal", "throttle"],
        "Gear": ["nGear", "Gear", "gear"],
    }
    for col in aliases.get(metric, []):
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy()
            if np.isfinite(arr).any():
                return arr
    return np.zeros(len(df))  # fallback renders grey

def render_heatmap_for_driver(code: str, metrics: list[str]):
    df = _load_driver_csv(code)
    x, y, use_xy = _prepare_xy(df)
    x, y = _smooth_xy(x, y, win=SMOOTH_WIN)

    for metric in metrics:
        vals_full = _metric_series(df, metric)
        m = min(len(x), len(vals_full))
        x_use, y_use, vals = x[:m], y[:m], vals_full[:m]

        segs, cvals = _downsample_pairs(x_use, y_use, vals, MAX_SEGMENTS)
        if segs is None:
            print(f"[warn] Not enough points for {code} {metric}")
            continue

        cmap = cm.get_cmap(CMAP.get(metric, "viridis"))
        norm = _norm_for_metric(cvals, metric)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=3.0, alpha=0.95)
        lc.set_array(cvals)
        ax.add_collection(lc)

        if use_xy:
            ax.set_aspect("equal", adjustable="datalim")
            ax.autoscale()
            ax.axis("off")
        else:
            ax.autoscale()
            ax.set_yticks([])
            ax.set_xlabel("Distance (m)")
            ax.grid(alpha=0.15)

        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(metric)
        ax.set_title(f"Montreal 2025 — {code} Fast Lap\nTrack Heatmap: {metric}")
        fig.savefig(OUTDIR / f"heatmap_{metric}_{code}.png", dpi=DPI)
        plt.close(fig)
        print(f"[ok] Saved {OUTDIR / f'heatmap_{metric}_{code}.png'}")

def render_multi_driver_grid(codes: list[str], metric: str):
    """Small multiples: same metric, multiple drivers, uniform color scale (no layout warnings)."""
    dfs = {c: _load_driver_csv(c) for c in codes}
    all_vals, xy = [], {}
    for c, df in dfs.items():
        v = _metric_series(df, metric)
        x, y, _ = _prepare_xy(df)
        x, y = _smooth_xy(x, y, win=SMOOTH_WIN)
        m = min(len(x), len(v))
        xy[c] = (x[:m], y[:m], v[:m])
        all_vals.append(v[:m])
    all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0])

    cmap = cm.get_cmap(CMAP.get(metric, "viridis"))
    norm = _norm_for_metric(all_vals, metric)

    n = len(codes)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols*4.5, rows*4.5), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, c in zip(axes, codes):
        x, y, v = xy[c]
        segs, cvals = _downsample_pairs(x, y, v, MAX_SEGMENTS)
        if segs is None:
            ax.axis("off"); continue
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=3.0, alpha=0.95)
        lc.set_array(cvals)
        ax.add_collection(lc)
        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale()
        ax.axis("off")
        ax.set_title(c)

    # clear unused axes
    for ax in axes[n:]:
        ax.axis("off")

    # Shared colorbar attached to all active axes (works with constrained layout)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[:n], location="right", shrink=0.9, pad=0.02)
    cb.set_label(metric)

    fig.suptitle(f"Montreal 2025 — Track Heatmaps ({metric})")
    out = OUTDIR / f"heatmap_{metric}_multidriver.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[ok] Saved {out}")

# -------------- Run ----------------
def main():
    # Resolve drivers
    if DRIVERS:
        codes = DRIVERS
    else:
        codes = []
        for f in _list_csvs():
            codes.append(_infer_driver(Path(f)))
        codes = list(dict.fromkeys(codes))[:6]  # stable order, cap at 6

    print("[info] Drivers:", ", ".join(codes))
    print("[info] Metrics:", ", ".join(METRICS))

    # Per-driver heatmaps
    for c in codes:
        render_heatmap_for_driver(c, METRICS)

    # Multi-driver grid for the first metric (nice overview)
    if codes and METRICS:
        render_multi_driver_grid(codes, METRICS[0])

if __name__ == "__main__":
    main()
