"""
Team Pit Stop Ranking – Montreal GP 2025
- Bars start at 0 (no negative axis).
- Team colors match mapping below.
- White square logo tiles (uniform size) sit just to the left edge of each bar (touching x=0).
- Logos are drawn above bars (no overlap underneath).

Run:
  python team_pitstop_ranking_logos_tiles_v2.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ---------------- CONFIG ----------------
LOGO_DIR = Path("logos")             # PNGs named exactly as TeamName
OUT_PNG  = "fig_team_pitstop_ranking_logos_tiles_v2.png"

# Official-ish team colors (tweak if your cache uses different naming)
TEAM_COLORS = {
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "Red Bull": "#1E41FF",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#00A0DE",
    "RB (VCARB)": "#2231D5",
    "Sauber": "#52E252",
    "Haas": "#B6BABD",
    "Unknown": "#4C78A8",
}

# White tile + layout
TILE_PX       = 120     # square tile size (pixels)
CONTENT_SCALE = 0.80    # logo fits within this fraction of tile
TILE_ZOOM     = 0.60    # visual size of the tile in the figure
TILE_X_OFFSET = 0.015   # as a fraction of max value: tile center is at +TILE_X_OFFSET*max on x-axis

TITLE = "Fastest Pit Crews – Average Stationary Time by Team\nMontreal GP 2025"

# ---------------- SAMPLE DATA ----------------
# Replace this block with your computed team averages (do NOT re-sort later if you want a fixed order).
team_avg = (
    pd.DataFrame({
        "TeamName": ["Ferrari", "Red Bull", "Mercedes", "McLaren", "RB (VCARB)", "Alpine"],
        "Duration_s": [2.45, 2.52, 2.70, 2.73, 2.88, 3.01]
    })
    .sort_values("Duration_s", ascending=True)
    .reset_index(drop=True)
)
# --------------------------------------------


def fmt_time(t: float) -> str:
    return f"{t:.2f}s"


def make_white_tile_with_logo(team: str, tile_px: int, content_scale: float) -> Image.Image | None:
    """Create a white square tile and paste the team logo centered on it."""
    p = LOGO_DIR / f"{team}.png"
    if not p.exists():
        return None
    try:
        logo = Image.open(p).convert("RGBA")
        inner = int(tile_px * content_scale)
        scale = min(inner / max(1, logo.width), inner / max(1, logo.height))
        new_w = max(1, int(round(logo.width * scale)))
        new_h = max(1, int(round(logo.height * scale)))
        logo_resized = logo.resize((new_w, new_h), Image.LANCZOS)

        tile = Image.new("RGBA", (tile_px, tile_px), (255, 255, 255, 255))
        offx = (tile_px - new_w) // 2
        offy = (tile_px - new_h) // 2
        tile.paste(logo_resized, (offx, offy), logo_resized)
        return tile
    except Exception:
        return None


def main():
    # Safety check: no negatives in data
    team_avg["Duration_s"] = team_avg["Duration_s"].clip(lower=0)

    labels = team_avg["TeamName"].tolist()
    values = team_avg["Duration_s"].tolist()
    vmax = max(values) if values else 0.0

    # Colors: use mapping exactly by team; optionally highlight fastest by outlining
    colors = [TEAM_COLORS.get(t, TEAM_COLORS["Unknown"]) for t in labels]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11.5, 7.2))

    # X-axis: start at 0 to avoid negative ticks
    ax.set_xlim(0, vmax * 1.10 if vmax > 0 else 1)

    # Bars (one per team)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="white", linewidth=1.0, zorder=3)

    # Value labels at bar ends
    for yi, v, b in zip(y, values, bars):
        ax.text(b.get_width() + vmax * 0.015, yi, fmt_time(v),
                va="center", ha="left", fontsize=10, color="#EAEAEA", zorder=5)

    # Put the white tiles so they "kiss" the bar start (slightly > 0 on the x-axis)
    x_tile_center = vmax * TILE_X_OFFSET  # a small positive x so it touches the bar edge at 0
    rendered_all_tiles = True
    for yi, team in zip(y, labels):
        tile = make_white_tile_with_logo(team, TILE_PX, CONTENT_SCALE)
        if tile is None:
            rendered_all_tiles = False
            # Fallback text label at bar origin if tile missing
            ax.text(x_tile_center, yi, team, va="center", ha="left",
                    color="#EAEAEA", fontsize=11, zorder=6)
            continue

        # Convert PIL -> array for OffsetImage
        tile_arr = np.asarray(tile)
        oi = OffsetImage(tile_arr, zoom=TILE_ZOOM)

        # Anchor tile so its right edge sits at x≈0 (touching bar). We approximate by shifting center slightly left.
        # Since we’re in data coords, a small backshift keeps the right edge near x=0 without going negative.
        # You can fine-tune with TILE_X_OFFSET above.
        ab = AnnotationBbox(
            oi,
            (x_tile_center, yi),                # xy in data coords
            frameon=False,
            box_alignment=(0.5, 0.5),
            xycoords=("data", "data"),
            pad=0,
            zorder=6,                           # above bars
            clip_on=False                       # never clipped under bars
        )
        ax.add_artist(ab)

    # Cosmetics
    ax.set_title(TITLE, fontsize=16, weight="bold", pad=10)
    ax.set_xlabel("Average Pit Stop (s)")
    ax.set_yticks(y)
    # Hide y tick labels if all tiles rendered (logos act as labels)
    if rendered_all_tiles:
        ax.set_yticklabels([""] * len(labels))
    else:
        ax.set_yticklabels(labels)

    ax.invert_yaxis()  # fastest at top
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
