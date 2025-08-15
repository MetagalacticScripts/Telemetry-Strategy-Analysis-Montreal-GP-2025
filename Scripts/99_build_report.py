# Scripts/99_build_report.py
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import html

ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data" / "processed"
PLOTS    = ROOT / "Reports" / "plots"
MEDIA    = ROOT / "Reports" / "media"
REPORTS  = ROOT / "Reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# -------------------- helpers --------------------
def read_parquet_safe(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"[warn] Failed to read {path.name}: {e}")
    return pd.DataFrame()

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def df_to_markdown(df: pd.DataFrame, floatfmt=".3f") -> str:
    """Robust markdown table: use pandas->tabulate if available, else monospaced fallback."""
    if df.empty:
        return "_(no data)_"
    try:
        # requires 'tabulate'
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        # plain text fallback
        return "```\n" + df.to_string(index=False) + "\n```"

def img_md(rel_from_root: Path, alt: str = "") -> str:
    return f"![{alt}]({rel_from_root.as_posix()})"

def video_html(src_rel: Path, poster_rel: Path | None = None) -> str:
    poster_attr = f' poster="{poster_rel.as_posix()}"' if poster_rel else ""
    return f'<video controls preload="metadata"{poster_attr} style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.06)"><source src="{src_rel.as_posix()}" type="video/mp4">Your browser does not support the video tag. <a href="{src_rel.as_posix()}">Download video</a>.</video>'

def media_exists(*names: str) -> Path | None:
    """Return first existing Path in MEDIA with any of the names."""
    for n in names:
        p = MEDIA / n
        if p.exists():
            return p
    return None

# -------------------- load data --------------------
features   = read_parquet_safe(DATA / "laps_features_canada_2025.parquet")
stints     = read_parquet_safe(DATA / "stints_summary_canada_2025.parquet")
pit_losses = read_parquet_safe(DATA / "pit_losses_estimated.parquet")
undercut   = read_parquet_safe(DATA / "undercut_overcut_pairs.parquet")
tyre_deg   = read_parquet_safe(DATA / "tyre_deg_summary.parquet")

# -------------------- key numbers --------------------
# Pace evolution (field median by lap)
if not features.empty and {"LapNumber", "LapTimeSeconds"}.issubset(features.columns):
    pace_med = (
        features.dropna(subset=["LapNumber", "LapTimeSeconds"])
                .groupby("LapNumber")["LapTimeSeconds"].median()
    )
    if not pace_med.empty:
        first10 = pace_med.loc[pace_med.index.min(): pace_med.index.min() + 9].mean() if len(pace_med) >= 10 else pace_med.head(5).mean()
        last10  = pace_med.loc[pace_med.index.max() - 9: pace_med.index.max()].mean() if len(pace_med) >= 10 else pace_med.tail(5).mean()
        field_improvement = first10 - last10
    else:
        field_improvement = np.nan
else:
    field_improvement = np.nan

# Best / worst degradation by stint
if not stints.empty and "deg_per_lap" in stints.columns:
    deg_tbl  = stints.sort_values("deg_per_lap")
    best_deg = deg_tbl.head(5).copy()
    worst_deg = deg_tbl.tail(5).copy()
else:
    best_deg = pd.DataFrame()
    worst_deg = pd.DataFrame()

# Pit losses
if not pit_losses.empty and {"under_sc_or_vsc", "pit_loss_seconds"}.issubset(pit_losses.columns):
    med_green = pit_losses.loc[~pit_losses["under_sc_or_vsc"], "pit_loss_seconds"].median()
    med_sc    = pit_losses.loc[pit_losses["under_sc_or_vsc"], "pit_loss_seconds"].median()
    stops_n   = int(len(pit_losses))
else:
    med_green = med_sc = np.nan
    stops_n = 0

# Top undercut gains
if not undercut.empty and {"who_stopped_first","gain_s"}.issubset(undercut.columns):
    top_under = (undercut[undercut["who_stopped_first"] == "A"]
                 .sort_values("gain_s", ascending=False)
                 .head(10)
                 .copy())
else:
    top_under = pd.DataFrame()

# Tyre degradation (summary lines)
tyre_lines = []
if not tyre_deg.empty and {"Compound","deg_s_per_lap","n_points"}.issubset(tyre_deg.columns):
    for _, r in tyre_deg.sort_values("deg_s_per_lap").iterrows():
        try:
            tyre_lines.append(f"- {r['Compound']}: {r['deg_s_per_lap']:+.3f} s/lap over first 15 laps (n={int(r['n_points'])})")
        except Exception:
            pass

# -------------------- visuals to include --------------------
# Plots (images)
plot_catalog = [
    ("Lap Evolution",                    PLOTS / "lap_evolution_rolling.png"),
    ("Stint Timeline",                   PLOTS / "stint_timeline.png"),
    ("Stint-Normalized Pace by Compound",PLOTS / "stint_pace_by_compound.png"),
    ("Pit Stop Loss vs Nearby Pace",     PLOTS / "pit_loss.png"),
    ("Undercut Gains (Top 20)",          PLOTS / "undercut_gains_top.png"),
    ("Tyre Degradation Curves",          PLOTS / "tyre_deg_curves.png"),
    # Heatmap small-multiples (first metric), if present
    ("Track Heatmaps — Speed (multidriver)", PLOTS / "heatmap_Speed_multidriver.png"),
]

# Media (videos / gifs / key images)
race_replay_mp4   = media_exists("race_replay.mp4")
race_replay_gif   = media_exists("race_replay.gif")
fastlap_mp4       = media_exists("anim_fastlaps.mp4", "anim_fastlaps.copy.mp4")
fastlap_gif       = media_exists("anim_fastlaps.gif")
delta_png         = (MEDIA / "delta_time_plot.png") if (MEDIA / "delta_time_plot.png").exists() else None

# -------------------- compose markdown --------------------
md = []
md.append("# Telemetry Strategy Analysis — Montreal GP 2025")
md.append("")
md.append(f"*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
md.append("")
md.append("## Headlines")
md.append(f"- Field median pace improved by **{fmt(field_improvement, 2)} s** from opening to closing 10 laps (fuel burn / rubbering).")
if stops_n:
    md.append(f"- Median **pit loss** under green: **{fmt(med_green,1)} s** vs **{fmt(med_sc,1)} s** under **SC/VSC** (n={stops_n}).")
if tyre_lines:
    md.append("- Tyre **degradation (sec/lap)** over first 15 laps:")
    md += tyre_lines
md.append("")

# Media previews (as links in MD)
md.append("## Highlights (Media)")
if delta_png:
    md.append("### Delta to Reference (fast-lap comparison)")
    md.append(img_md(delta_png.relative_to(ROOT), "Delta to reference"))
    md.append("")
if fastlap_mp4 or fastlap_gif:
    md.append("### Fast Lap Animation")
    if fastlap_mp4:
        md.append(f"[Watch MP4]({fastlap_mp4.relative_to(ROOT).as_posix()})")
    if fastlap_gif:
        md.append(f"[View GIF]({fastlap_gif.relative_to(ROOT).as_posix()})")
    md.append("")
if race_replay_mp4 or race_replay_gif:
    md.append("### Race Replay")
    if race_replay_mp4:
        md.append(f"[Watch MP4]({race_replay_mp4.relative_to(ROOT).as_posix()})")
    if race_replay_gif:
        md.append(f"[View GIF]({race_replay_gif.relative_to(ROOT).as_posix()})")
    md.append("")

md.append("## Visuals")
for title, p in plot_catalog:
    if p.exists():
        md.append(f"### {title}")
        md.append(img_md(p.relative_to(ROOT), title))
        md.append("")

md.append("## Stint Degradation — Best & Worst (sec/lap)")
if not best_deg.empty and not worst_deg.empty:
    best_tbl = best_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]].copy()
    worst_tbl = worst_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]].copy()
    md.append("**Best 5 (lowest degradation):**")
    md.append(df_to_markdown(best_tbl, floatfmt=".3f"))
    md.append("")
    md.append("**Worst 5 (highest degradation):**")
    md.append(df_to_markdown(worst_tbl, floatfmt=".3f"))
else:
    md.append("_Stint table missing or incomplete._")

md.append("")
md.append("## Top Undercut Pairs")
if not top_under.empty:
    show = top_under[["A_driver","B_rival","A_in_lap","B_in_lap","compare_lap","gain_s","gap_before_s","gap_after_s"]].copy()
    md.append(df_to_markdown(show, floatfmt=".3f"))
else:
    md.append("_No rival-matched undercut pairs found or file missing._")

md.append("")
md.append("## Method Notes")
md.append("- **Data:** FastF1 telemetry/timing; cached locally.")
md.append("- **Clean laps:** removed in/out laps, non-green (where available), and implausible outliers.")
md.append("- **Pit loss:** `(in-lap + out-lap) − 2 × median(neighbor laps)`; SC/VSC flagged if either lap affected.")
md.append("- **Undercut:** compared elapsed-time gap from lap before first stop to lap after both stops.")
md.append("- **Degradation:** median lap vs tyre age; linear fit over first 15 laps for slope.")
md.append("")

md_path = REPORTS / "MontrealGP2025_report.md"
md_path.write_text("\n".join(md), encoding="utf-8")

# -------------------- compose standalone HTML --------------------
def table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p class='muted'>—</p>"
    return df.to_html(index=False, escape=True, border=0, classes="table", float_format=lambda x: f"{x:.3f}")

# Build media HTML blocks
media_blocks = []
if delta_png:
    media_blocks.append(f"<h3>Delta to Reference (fast-lap comparison)</h3><img src='{delta_png.relative_to(ROOT).as_posix()}' alt='Delta plot'/>")
if fastlap_mp4 or fastlap_gif:
    media_blocks.append("<h3>Fast Lap Animation</h3>")
    if fastlap_mp4:
        media_blocks.append(video_html(fastlap_mp4.relative_to(ROOT), poster_rel=delta_png.relative_to(ROOT) if delta_png else None))
    if fastlap_gif and not fastlap_mp4:
        media_blocks.append(f"<img src='{fastlap_gif.relative_to(ROOT).as_posix()}' alt='Fast lap animation'/>")
if race_replay_mp4 or race_replay_gif:
    media_blocks.append("<h3>Race Replay</h3>")
    if race_replay_mp4:
        media_blocks.append(video_html(race_replay_mp4.relative_to(ROOT)))
    if race_replay_gif and not race_replay_mp4:
        media_blocks.append(f"<img src='{race_replay_gif.relative_to(ROOT).as_posix()}' alt='Race replay'/>")

plots_html = "".join(
    f"<h3>{html.escape(title)}</h3><img src='{p.relative_to(ROOT).as_posix()}' alt='{html.escape(title)}'/>"
    for title, p in plot_catalog if p.exists()
)

best_html  = table_html(best_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]]) if not best_deg.empty else "<p class='muted'>—</p>"
worst_html = table_html(worst_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]]) if not worst_deg.empty else "<p class='muted'>—</p>"
under_html = table_html(top_under[["A_driver","B_rival","A_in_lap","B_in_lap","compare_lap","gain_s","gap_before_s","gap_after_s"]]) if not top_under.empty else "<p class='muted'>—</p>"
tyre_html  = "<br>".join(tyre_lines) if tyre_lines else "—"

html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Telemetry Strategy Analysis — Montreal GP 2025</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; line-height: 1.55; }}
 h1, h2, h3 {{ margin-top: 1.6rem; }}
 img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
 table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.95rem; }}
 th, td {{ border: 1px solid #eaeaea; padding: 8px; text-align: left; }}
 th {{ background: #f7f7f7; }}
 code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
 .muted {{ color: #666; }}
 .table {{ }}
</style>
</head>
<body>
<article>
<h1>Telemetry Strategy Analysis — Montreal GP 2025</h1>
<p class="muted">Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<section>
<h2>Headlines</h2>
<ul>
  <li>Field median pace improved by <b>{fmt(field_improvement,2)}</b> s from opening to closing 10 laps.</li>
  <li>Median pit loss — Green: <b>{fmt(med_green,1)}</b> s; SC/VSC: <b>{fmt(med_sc,1)}</b> s.{" (n="+str(stops_n)+")" if stops_n else ""}</li>
  <li>Tyre degradation (first 15 laps):<br>{tyre_html}</li>
</ul>
</section>

<section>
<h2>Highlights (Media)</h2>
{''.join(media_blocks)}
</section>

<section>
<h2>Visuals</h2>
{plots_html if plots_html else "<p class='muted'>No static plots found.</p>"}
</section>

<section>
<h2>Stint Degradation — Best & Worst</h2>
<h3>Best 5 (lowest)</h3>
{best_html}
<h3>Worst 5 (highest)</h3>
{worst_html}
</section>

<section>
<h2>Top Undercut Pairs</h2>
{under_html}
</section>

<section>
<h2>Method Notes</h2>
<ul>
<li><b>Data:</b> FastF1 telemetry/timing; cached locally.</li>
<li><b>Clean laps:</b> removed in/out laps, non-green (where available), and implausible outliers.</li>
<li><b>Pit loss:</b> <code>(in-lap + out-lap) − 2 × median(neighbor laps)</code>; SC/VSC flagged if either lap affected.</li>
<li><b>Undercut:</b> compared elapsed-time gap from lap before first stop to lap after both stops.</li>
<li><b>Degradation:</b> median lap vs tyre age; linear fit over first 15 laps for slope.</li>
</ul>
</section>
</article>
</body>
</html>
"""

md_path   = REPORTS / "MontrealGP2025_report.md"
html_path = REPORTS / "MontrealGP2025_report.html"
md_path.write_text("\n".join(md), encoding="utf-8")
html_path.write_text(html_doc, encoding="utf-8")

print("Wrote:")
print(" -", md_path)
print(" -", html_path)
