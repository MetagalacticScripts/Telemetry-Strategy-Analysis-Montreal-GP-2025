# Scripts/99_build_report.py
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
PLOTS = ROOT / "Reports" / "plots"
REPORTS = ROOT / "Reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# --------- Load data we produced earlier ---------
features = pd.read_parquet(DATA / "laps_features_canada_2025.parquet")
stints = pd.read_parquet(DATA / "stints_summary_canada_2025.parquet")
pit_losses = pd.read_parquet(DATA / "pit_losses_estimated.parquet") if (DATA / "pit_losses_estimated.parquet").exists() else pd.DataFrame()
undercut = pd.read_parquet(DATA / "undercut_overcut_pairs.parquet") if (DATA / "undercut_overcut_pairs.parquet").exists() else pd.DataFrame()
tyre_deg = pd.read_parquet(DATA / "tyre_deg_summary.parquet") if (DATA / "tyre_deg_summary.parquet").exists() else pd.DataFrame()

# --------- Key numbers / insights ---------
# Pace evolution (field median by lap)
pace_med = (
    features.dropna(subset=["LapNumber","LapTimeSeconds"])
            .groupby("LapNumber")["LapTimeSeconds"].median()
)
if not pace_med.empty:
    first10 = pace_med.loc[pace_med.index.min():pace_med.index.min()+9].mean()
    last10  = pace_med.loc[pace_med.index.max()-9:pace_med.index.max()].mean()
    field_improvement = first10 - last10
else:
    field_improvement = np.nan

# Best/worst degradation by stint
deg_tbl = stints.sort_values("deg_per_lap")
best_deg = deg_tbl.head(5)
worst_deg = deg_tbl.tail(5)

# Pit loss medians
if not pit_losses.empty:
    med_green = pit_losses.loc[~pit_losses["under_sc_or_vsc"], "pit_loss_seconds"].median()
    med_sc    = pit_losses.loc[pit_losses["under_sc_or_vsc"], "pit_loss_seconds"].median()
    stops_n   = len(pit_losses)
else:
    med_green = med_sc = np.nan
    stops_n = 0

# Top undercut gains
top_under = pd.DataFrame()
if not undercut.empty:
    top_under = (undercut[undercut["who_stopped_first"]=="A"]
                 .sort_values("gain_s", ascending=False)
                 .head(10)
                 .copy())

# Tyre deg slopes
tyre_lines = []
if not tyre_deg.empty:
    for _, r in tyre_deg.sort_values("deg_s_per_lap").iterrows():
        tyre_lines.append(f"- {r['Compound']}: {r['deg_s_per_lap']:+.3f} s/lap over first 15 laps (n={int(r['n_points'])})")

# --------- Helper: safe format ---------
def fmt(x, nd=3):
    return "—" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"

# --------- Build Markdown ---------
md = []
md.append(f"# Telemetry Strategy Analysis — Montreal GP 2025")
md.append("")
md.append(f"*Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
md.append("")
md.append("## Headlines")
md.append("- Field median pace improved by **" + (fmt(field_improvement,2)) + " s** from opening to closing 10 laps (fuel burn / rubbering).")
if stops_n:
    md.append(f"- Median **pit loss** under green: **{fmt(med_green,1)} s** vs **{fmt(med_sc,1)} s** under **SC/VSC**.")
if tyre_lines:
    md.append("- Tyre **degradation (sec/lap)** over first 15 laps:")
    md += tyre_lines
md.append("")

md.append("## Visuals")
def img(rel):
    return f"![{rel}]({(PLOTS/rel).relative_to(ROOT).as_posix()})"
plots_to_include = [
    ("Lap Evolution", "lap_evolution_rolling.png"),
    ("Stint Timeline", "stint_timeline.png"),
    ("Stint-Normalized Pace by Compound", "stint_pace_by_compound.png"),
    ("Pit Stop Loss vs Nearby Pace", "pit_loss.png"),
    ("Undercut Gains (Top 20)", "undercut_gains_top.png"),
    ("Tyre Degradation Curves", "tyre_deg_curves.png"),
]
for title, fname in plots_to_include:
    if (PLOTS / fname).exists():
        md.append(f"### {title}")
        md.append(img(fname))
        md.append("")

md.append("## Stint Degradation — Best & Worst (sec/lap)")
if not stints.empty:
    best_tbl = best_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]].copy()
    worst_tbl = worst_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]].copy()
    md.append("**Best 5 (lowest degradation):**")
    md.append(best_tbl.to_markdown(index=False, floatfmt=".3f"))
    md.append("")
    md.append("**Worst 5 (highest degradation):**")
    md.append(worst_tbl.to_markdown(index=False, floatfmt=".3f"))
else:
    md.append("_Stint table missing._")

md.append("")
md.append("## Top Undercut Pairs")
if not top_under.empty:
    show = top_under[["A_driver","B_rival","A_in_lap","B_in_lap","compare_lap","gain_s","gap_before_s","gap_after_s"]].copy()
    md.append(show.to_markdown(index=False, floatfmt=".3f"))
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

# Write Markdown
md_path = REPORTS / "MontrealGP2025_report.md"
md_path.write_text("\n".join(md), encoding="utf-8")

# Also write a simple standalone HTML (no external deps)
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Telemetry Strategy Analysis — Montreal GP 2025</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; line-height: 1.5; }}
 h1, h2, h3 {{ margin-top: 1.6em; }}
 img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
 table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
 th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
 th {{ background: #f7f7f7; }}
 code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
 .muted {{ color: #666; }}
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
<li>Median pit loss — Green: <b>{fmt(med_green,1)}</b> s; SC/VSC: <b>{fmt(med_sc,1)}</b> s.</li>
<li>Tyre degradation (first 15 laps):<br>{"<br>".join(tyre_lines) if tyre_lines else "—"}</li>
</ul>
</section>
<section>
<h2>Visuals</h2>
{"".join(
    f'<h3>{title}</h3><img src="{(PLOTS/fname).relative_to(ROOT).as_posix()}" alt="{title}"/>'
    for title, fname in plots_to_include if (PLOTS/fname).exists()
)}
</section>
<section>
<h2>Stint Degradation — Best & Worst</h2>
<h3>Best 5 (lowest)</h3>
{best_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]].to_html(index=False, float_format=lambda x: f"{x:.3f}")}
<h3>Worst 5 (highest)</h3>
{worst_deg[["Driver","StintNo","Compound","laps_in_stint","deg_per_lap","stint_median","stint_best"]].to_html(index=False, float_format=lambda x: f"{x:.3f}")}
</section>
<section>
<h2>Top Undercut Pairs</h2>
{(top_under[["A_driver","B_rival","A_in_lap","B_in_lap","compare_lap","gain_s","gap_before_s","gap_after_s"]].to_html(index=False, float_format=lambda x: f"{x:.3f}")) if not top_under.empty else "<p>—</p>"}
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
</html>"""
html_path = REPORTS / "MontrealGP2025_report.html"
html_path.write_text(html, encoding="utf-8")

print("Wrote:")
print(" -", md_path)
print(" -", html_path)
