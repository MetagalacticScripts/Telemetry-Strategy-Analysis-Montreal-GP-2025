# Montreal GP 2025 – Race Engineering Data Analysis
An in-depth review of the 2025 Montreal Grand Prix using official telemetry and lap timing data to uncover driver performance patterns, pit stop strategy decisions, and pace evolution.  

 

https://github.com/user-attachments/assets/1c0ebec2-04cf-4123-b285-3b6870317a1e



---

## Race Results – Top 10  
1. **George Russell** – Mercedes – 1:31:52.688  
2. **Max Verstappen** – Red Bull – +0.228s  
3. **Andrea Kimi Antonelli** – Mercedes – +1.014s  
4. **Oscar Piastri** – McLaren – +2.109s  
5. **Charles Leclerc** – Ferrari – +3.442s  
6. **Lewis Hamilton** – Ferrari – +10.713s  
7. **Fernando Alonso** – Aston Martin – +10.972s  
8. **Nico Hülkenberg** – Kick Sauber – +15.364s  
9. **Esteban Ocon** – Haas – +1 Lap  
10. **Carlos Sainz Jr.** – Williams – +1 Lap  

---

## Technical Project Highlights  
- **Data Acquisition** – Collected telemetry & timing data via `FastF1`.  
- **Data Structuring** – Integrated lap times, stint data, and pit stops into cohesive datasets.  
- **Performance Analysis** – Evaluated pace trends, tyre degradation, and sector time differentials.  
- **Strategy Modelling** – Compared stint lengths, compound choices, and pit stop timings.  
- **Visualization** – Produced charts for lap evolution, pit windows, tyre degradation, and position changes.  

---

## Key Visuals  

<table>
<tr>
<td>

**Lap Evolution (Rolling Median Pace)**  
<img src="Reports/plots/lap_evolution_rolling.png" width="100%">

</td>
<td>

**Stint Timeline**  
<img src="Reports/plots/stint_timeline.png" width="100%">

</td>
</tr>

<tr>
<td>

**Tyre Degradation Curves**  
<img src="Reports/plots/tyre_deg_curves.png" width="100%">

</td>
<td>

**Pit Stop Loss – Green vs SC/VSC**  
<img src="Reports/plots/pit_loss.png" width="100%">

</td>
</tr>

<tr>
<td>

**Undercut Gains (Top 20)**  
<img src="Reports/plots/undercut_gains_top.png" width="100%">

</td>
<td>

**Track Heatmap – Speed (Antonelli Example)**  
<img src="Reports/plots/heatmap_Speed_ANT.png" width="100%">

</td>
</tr>

<tr>
<td colspan="2">

**Delta Time Plot – Russell vs Verstappen vs Antonelli**  
<img src="Reports/media/delta_time_plot.png" width="100%">

</td>
</tr>
</table>

---

## Conclusions  

- **Race Pace Evolution** – The median pace across the field improved by **~2.43s** from the opening to the final stint, reflecting track rubbering and fuel burn-off.  
- **Pit Strategy Impact** – Average pit loss under green was **19.5s**, but ballooned to **76.5s** when stops coincided with SC/VSC periods. This heavily influenced position changes for mid-field runners.  
- **Tyre Performance** –  
  - **HARD** tyres showed minimal degradation (−0.031 s/lap in first 15 laps), favoring consistent long runs.  
  - **MEDIUM** tyres degraded more quickly (+0.084 s/lap), making them potent for short, aggressive stints but vulnerable over distance.  
- **Standout Tyre Management** – Leclerc, Russell, and Antonelli posted some of the lowest degradation rates, enabling strategic flexibility late in the race.  
- **Undercut Opportunities** – Multiple large undercut gains (>3s) were recorded, with Boraschi vs Stroll (+9.2s) and Verstappen vs Hamilton (+8.4s) being the most decisive.  
- **Key Battles** – The Russell–Verstappen–Antonelli trio ran within a second of each other in the closing laps, with Russell narrowly holding the lead under pressure.  

