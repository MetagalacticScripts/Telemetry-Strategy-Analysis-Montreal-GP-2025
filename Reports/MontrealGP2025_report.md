# Telemetry Strategy Analysis — Montreal GP 2025

*Auto-generated on 2025-08-16 01:21*

## Headlines
- Field median pace improved by **2.43 s** from opening to closing 10 laps (fuel burn / rubbering).
- Median **pit loss** under green: **19.5 s** vs **76.5 s** under **SC/VSC** (n=82).
- Tyre **degradation (sec/lap)** over first 15 laps:
- HARD: -0.031 s/lap over first 15 laps (n=15)
- MEDIUM: +0.084 s/lap over first 15 laps (n=15)

## Highlights (Media)
### Delta to Reference (fast-lap comparison)
![Delta to reference](Reports/media/delta_time_plot.png)

### Fast Lap Animation
[Watch MP4](Reports/media/anim_fastlaps.mp4)

### Race Replay
[Watch MP4](Reports/media/race_replay.mp4)

## Visuals
### Lap Evolution
![Lap Evolution](Reports/plots/lap_evolution_rolling.png)

### Stint Timeline
![Stint Timeline](Reports/plots/stint_timeline.png)

### Stint-Normalized Pace by Compound
![Stint-Normalized Pace by Compound](Reports/plots/stint_pace_by_compound.png)

### Pit Stop Loss vs Nearby Pace
![Pit Stop Loss vs Nearby Pace](Reports/plots/pit_loss.png)

### Undercut Gains (Top 20)
![Undercut Gains (Top 20)](Reports/plots/undercut_gains_top.png)

### Tyre Degradation Curves
![Tyre Degradation Curves](Reports/plots/tyre_deg_curves.png)

### Track Heatmaps — Speed (multidriver)
![Track Heatmaps — Speed (multidriver)](Reports/plots/heatmap_Speed_multidriver.png)

## Stint Degradation — Best & Worst (sec/lap)
**Best 5 (lowest degradation):**
| Driver   |   StintNo | Compound   |   laps_in_stint |   deg_per_lap |   stint_median |   stint_best |
|:---------|----------:|:-----------|----------------:|--------------:|---------------:|-------------:|
| LEC      |         1 | HARD       |              26 |        -0.064 |         76.564 |       76.019 |
| RUS      |         3 | HARD       |              21 |        -0.063 |         74.869 |       74.119 |
| LAW      |         2 | MEDIUM     |              12 |        -0.059 |         77.278 |       76.320 |
| HUL      |         2 | HARD       |              44 |        -0.048 |         76.555 |       75.372 |
| ANT      |         3 | HARD       |              25 |        -0.043 |         74.860 |       74.455 |

**Worst 5 (highest degradation):**
| Driver   |   StintNo | Compound   |   laps_in_stint |   deg_per_lap |   stint_median |   stint_best |
|:---------|----------:|:-----------|----------------:|--------------:|---------------:|-------------:|
| ALB      |         1 | MEDIUM     |              21 |         0.080 |         78.040 |       77.275 |
| LEC      |         3 | MEDIUM     |              11 |         0.084 |         74.798 |       74.261 |
| VER      |         1 | MEDIUM     |              10 |         0.092 |         76.566 |       75.901 |
| TSU      |         2 | MEDIUM     |               7 |         0.110 |         75.910 |       75.358 |
| ALO      |         1 | MEDIUM     |              13 |         0.181 |         77.208 |       76.369 |

## Top Undercut Pairs
| A_driver   | B_rival   |   A_in_lap |   B_in_lap |   compare_lap |   gain_s |   gap_before_s |   gap_after_s |
|:-----------|:----------|-----------:|-----------:|--------------:|---------:|---------------:|--------------:|
| BOR        | STR       |         49 |         51 |            53 |    9.207 |         -4.821 |       -14.028 |
| VER        | HAM       |         12 |         15 |            17 |    8.384 |         -4.810 |       -13.194 |
| ANT        | HAM       |         14 |         15 |            17 |    4.440 |         -4.186 |        -8.626 |
| BOR        | STR       |         66 |         66 |            68 |    3.775 |         -2.771 |        -6.546 |
| SAI        | OCO       |         57 |         57 |            59 |    3.642 |          4.154 |         0.512 |
| COL        | STR       |         66 |         66 |            68 |    3.637 |         -4.668 |        -8.305 |
| TSU        | HAD       |         66 |         66 |            68 |    3.235 |         -4.838 |        -8.073 |
| VER        | ANT       |         12 |         14 |            16 |    3.118 |         -0.451 |        -3.569 |
| RUS        | ANT       |         13 |         14 |            16 |    2.796 |         -2.996 |        -5.792 |
| ANT        | PIA       |         67 |         67 |            69 |    2.733 |         -1.243 |        -3.976 |

## Method Notes
- **Data:** FastF1 telemetry/timing; cached locally.
- **Clean laps:** removed in/out laps, non-green (where available), and implausible outliers.
- **Pit loss:** `(in-lap + out-lap) − 2 × median(neighbor laps)`; SC/VSC flagged if either lap affected.
- **Undercut:** compared elapsed-time gap from lap before first stop to lap after both stops.
- **Degradation:** median lap vs tyre age; linear fit over first 15 laps for slope.
