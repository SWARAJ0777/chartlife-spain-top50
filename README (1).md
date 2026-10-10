# ChartLife Spain — Top 50 Lifecycle Intelligence

**Content Maturity, Release Lifecycle & Playlist Rotation Analysis of Spain's Top 50 Songs**

> Lifecycle intelligence for Spain's Top 50: rebuilds 544 songs from 555 daily snapshots to measure survival, playlist churn and content maturity. Finds album drops drive 20% of entries, with a 13-day median stay. It includes Kaplan-Meier analysis, six KPIs, an interactive Streamlit dashboard, a research paper and a government executive summary.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red) ![Status](https://img.shields.io/badge/status-complete-brightgreen)

*Client: Atlantic Recording Corporation · Programme: Unified Mentor · Author: Swaraj Dattaram Tawade*

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Deliverables](#2-deliverables)
3. [Headline results](#3-headline-results)
4. [Dataset and data quality](#4-dataset-and-data-quality)
5. [Methodology](#5-methodology)
6. [KPIs](#6-kpis)
7. [Dashboard guide](#7-dashboard-guide)
8. [How to run](#8-how-to-run)
9. [Troubleshooting](#9-troubleshooting)
10. [Recommendations](#10-recommendations)
11. [Assumptions and ambiguities](#11-assumptions-and-ambiguities)
12. [Limitations](#12-limitations)
13. [Verification and reproducibility](#13-verification-and-reproducibility)
14. [Tech stack](#14-tech-stack)
15. [Future work](#15-future-work)

---

## 1. Project overview

### Business context
The Spanish music market has distinctive traits: strong Latin and regional genres, sensitivity to release freshness, fast playlist rotation and distinct behaviour around explicit versus clean content. Atlantic Recording Corporation records the Spain Top 50 playlist every day but lacked **lifecycle intelligence**: how long songs survive, whether the list favours fresh or mature tracks, and how song attributes shape entry and exit. Without it, release strategy risks being copied from other regions.

### Questions answered
| # | Question | Short answer |
|---|---|---|
| Q1 | How long do songs survive on Spain's Top 50? | Median new entry stays **13 days**; 60% reach day 7, 39% day 30, 22% day 90 |
| Q2 | Fresh releases or mature tracks? | Both: fresh *flow* (1.67 entries a day) but a mature *stock* (songs older than 90 days hold 44% of slots) |
| Q3 | How do attributes affect entry speed and exit? | Entry rank and album-drop entry dominate; track length, release size and format add nothing once controlled |
| Q4 | Does explicit content mature differently? | No overall difference (1.07×, 95% CI 0.79–1.43) |
| Q5 | How do album and single tracks behave? | Singles last 1.73× longer overall, but only 1.23× (not significant) once album drops are set aside |

---

## 2. Deliverables

| # | File | What it is |
|---|---|---|
| 1 | `docs/Research_Paper_Spain_Top50_Lifecycle.docx` | Research paper (16 pages): methodology, data preparation, EDA, 13 figures, key findings, insights, limitations, recommendations |
| 2 | `Spain_Top50_Streamlit_Dashboard.py` | Streamlit dashboard — a **single self-contained file** (analysis engine + dashboard) |
| 3 | `docs/Executive_Summary_Spain_Top50.docx` | 2-page, decision-focused summary for government stakeholders |
| — | `Atlantic_Spain.csv` | Input dataset (27,800 rows) |
| — | `notebook/Spain_Top50_Lifecycle_Colab.ipynb` | *(optional)* complete reproducible workflow for Google Colab |

### Recommended folder layout
```
chartlife-spain-top50/
├── Spain_Top50_Streamlit_Dashboard.py
├── Atlantic_Spain.csv
├── requirements.txt
├── runtime.txt
├── README.md
├── docs/
│   ├── Research_Paper_Spain_Top50_Lifecycle.docx
│   └── Executive_Summary_Spain_Top50.docx
└── notebook/                      (optional)
    └── Spain_Top50_Lifecycle_Colab.ipynb
```

---

## 3. Headline results

**Lifecycle length**
- Median new entry survives **13 days**; the mean of completed lifecycles is 45.6 days but the median only 10.5: a long right tail (longest-lived track: 538 days).
- Barbell shape: 40% of songs are gone within a week and hold 2.4% of slot-days; the 21% that last over 90 days hold **73%**.

**Rotation**
- 1.67 entries a day (3.33% churn); **17 album-drop days (3.1% of days) bring 20% of all entries.**
- Album-drop tracks have a median life of 5 days versus 26 for other entries (ratio 0.36, CI 0.22–0.54).
- 61% of entries land at ranks 41–50 and 81% of exits leave from there; the Top 10 changes by only 0.59 tracks a day.

**Content attributes**
- Single vs album: 1.73× → **1.23×** (CI 0.90–1.74, not significant) excluding album drops.
- Explicit vs clean: 1.07× (CI 0.79–1.43, not significant). A possible 1.41× premium outside drops is flagged as *exploratory*.
- In a multivariable model of surviving 30+ days: album-drop entry OR 0.17, entry rank OR 0.77 per 10 positions worse, single OR 1.09 (not significant).

**Early signal**
- 69% of songs entering in the Top 10 are still listed after 30 days, versus 26%–38% for other entry bands.
- Popularity is a lagging indicator (peaks a median 10 days after entry) and still sits at 93% of a song's peak on its last day: songs are displaced, not abandoned.

---

## 4. Dataset and data quality

**File:** `Atlantic_Spain.csv` — 27,800 rows, 555 daily snapshots, 2024-05-18 to 2025-11-27.

| Column | Description |
|---|---|
| `date` | Snapshot date (format `dd-mm-yyyy`) |
| `position` | Playlist rank (1–50) |
| `song` | Song title |
| `artist` | Artist(s), joined with `&` |
| `popularity` | Popularity score (0–100) from the Atlantic API |
| `duration_ms` | Song duration in milliseconds |
| `album_type` | `single` or `album` |
| `total_tracks` | Number of tracks on the release |
| `is_explicit` | Explicit-content flag |
| `album_cover_url` | Artwork URL (not used analytically) |

### Validation findings and treatment
| Check | Finding | Treatment |
|---|---|---|
| Exactly 50 entries per day | 1 Mar 2025 has 100 rows (two scrapes of one day) | Keep the last row per (date, position) → every day has exactly 50 |
| Calendar gaps | 4 missing days: 2025-03-14, 2025-03-25, 2025-07-11, 2025-08-13 | Not imputed; transitions across a gap excluded from churn averages |
| Missing values | None | — |
| `popularity` = 0 | 106 rows on 25 days, 87% on a track's first two days | Treated as "not yet scored" (missing) |
| Duration outliers | 91–555 seconds, all plausible | Kept |
| Naming consistency | 583 raw song–artist strings → 544 tracks | Normalised case, accents, spacing and featured-artist credits |
| Attribute drift | `album_type` changes for 41 tracks, `total_tracks` for 48 | Most frequent value per track |

---

## 5. Methodology

1. **Validation & normalisation** — 50 entries per day, normalised names, first appearance per song.
2. **Lifecycle construction** — per track: entry date, exit date (last day listed), total days on playlist (distinct days, re-entries included), peak position, time to peak (days from first appearance to first day at best rank).
3. **Stage classification** — evaluated per song-day, in this order:

   | Stage | Rule |
   |---|---|
   | New Entry | first 7 listed days of an observed spell |
   | Peak Phase | 7-day median rank in the Top 10 |
   | Growth Phase | trailing 7-day rank slope ≤ −0.75 positions/day (improving) |
   | Decline Phase | trailing 7-day rank slope ≥ +0.75 positions/day (deteriorating) |
   | Mature Phase | everything else (stable, mid-table) |

4. **Rotation & churn** — daily entries/exits (equal, because the list is fixed at 50), churn = entries ÷ 50 on valid consecutive-day transitions, rank volatility (mean |daily rank move|), monthly comparison, **album-drop bursts** (≥3 tracks of one primary artist debuting on the same day).
5. **Content attributes vs lifecycle** — explicit vs clean, single vs album, song length (quartiles), release size.
6. **Popularity vs maturity** — popularity by age, time to peak popularity, decay before exit.
7. **Statistics** — Kaplan–Meier survival and log-rank tests (censoring-aware), Mann–Whitney, bootstrap 95% CIs for ratios of mean days, restricted mean survival time (90 days), a logistic model adjusting for album-drop entry and entry rank, Spearman correlations. With 6 primary comparisons a Bonferroni threshold of p<0.008 is used; results between that and 0.05 are labelled exploratory.

**Censoring.** 50 songs were already listed on day 1 and 50 were still listed on the last day. Duration KPIs therefore use *completed* lifecycles (n=444), entry speed uses the 494 tracks whose entry was observed, and survival curves treat still-listed songs as censored.

---

## 6. KPIs

| KPI | Definition | Value |
|---|---|---|
| Average Days on Playlist | Mean days listed, completed lifecycles | **45.6** (median 10.5) |
| Entry-to-Peak Time | Mean days from first appearance to best rank | **10.3** (median 0.5) |
| Playlist Churn Rate | Mean daily entries ÷ 50 | **3.33%** a day |
| Retention Stability Index | Mean of 1 − (mean abs. daily rank change ÷ 49) | **0.927** |
| Explicit Content Lifecycle Score | Mean days explicit ÷ mean days clean | **1.07×** |
| Single vs Album Longevity Ratio | Mean days single ÷ mean days album | **1.73×** |

The brief names these KPIs but not their formulas; the definitions above are this project's documented readings.

---

## 7. Dashboard guide

### Modules (tabs)
| Tab | What it shows |
|---|---|
| Overview | Six KPI tiles, survival curve, barbell chart, key findings |
| Song lifecycle timeline | Gantt-style timeline of the longest-lived / best-ranked / newest songs; single-song rank and popularity trajectory |
| Entry vs exit flow | Daily entries and exits, where songs enter and leave, weekday pattern, album-drop events table |
| Lifecycle stages | Stage distribution, stage mix over time, stage profile |
| Content maturity | Explicit vs clean and single vs album survival curves, box plots, song length and release size, statistical comparison table, CSV download |
| Churn analytics | Monthly rotation, distribution of daily entries, stability vs volatility |
| Data quality & method | Validation results and definitions |

### Filters (sidebar)
- **Date range**
- **Explicit content:** All / Explicit only / Clean only
- **Album type:** single, album
- **Lifecycle stage:** New Entry, Growth, Peak, Mature, Decline
- **Dataset upload** (optional)

**How the filters work.** Date, explicit and album filters restrict the daily rows. The stage filter applies to the stage, timeline and content-maturity views; entry/exit flow and churn always use all stages (an entry is by definition a "New Entry"). Song-level metrics describe songs that appear in the filtered window, while their lifecycle is always measured on the full history. KPI tiles show the change versus the unfiltered values.

**Performance.** The first load takes about 20 seconds while lifecycles and statistics are computed; results are cached, so filter changes then respond in about a second.

---

## 8. How to run

### 8.1 Prerequisites
- **Python 3.11 or newer** (the pinned pandas 3.x requires it)
- A terminal (PowerShell, Command Prompt, macOS Terminal or Linux shell); VS Code is optional but recommended

### 8.2 Create the two small config files
Put these next to the dashboard file.

`requirements.txt`
```
streamlit==1.65.0
plotly==7.1.0
pandas==3.0.5
numpy==2.5.3
scipy==1.18.1
statsmodels==0.15.0
```

`runtime.txt`
```
python-3.11
```

(`statsmodels` is optional: it only powers the adjusted odds-ratio model; the dashboard runs without it.)

### 8.3 Windows + VS Code (step by step)
1. Create a folder, e.g. `chartlife-spain-top50`, and place `Spain_Top50_Streamlit_Dashboard.py`, `Atlantic_Spain.csv`, `requirements.txt` and `runtime.txt` in it.
2. Open the folder in VS Code (*File ▸ Open Folder*), then open a terminal (*Terminal ▸ New Terminal*).
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
4. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```powershell
   streamlit run Spain_Top50_Streamlit_Dashboard.py
   ```
6. Your browser opens at **http://localhost:8501**. Wait about 20 seconds for the first load. Stop the app with `Ctrl + C`.

### 8.4 macOS / Linux
```bash
cd chartlife-spain-top50
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Spain_Top50_Streamlit_Dashboard.py
```

### 8.5 Using your own CSV
Leave `Atlantic_Spain.csv` out of the folder (or use another file) and **upload** the CSV from the dashboard sidebar. The file must have the ten columns listed in [section 4](#4-dataset-and-data-quality) and dates in `dd-mm-yyyy` format.

### 8.6 Deploy on Streamlit Community Cloud
1. Push the folder (dashboard file, CSV, `requirements.txt`, `runtime.txt`) to a GitHub repository.
2. On [share.streamlit.io](https://share.streamlit.io) choose **New app**, select the repository and branch, and set **Main file path** to `Spain_Top50_Streamlit_Dashboard.py`.
3. Under *Advanced settings* select **Python 3.11** and deploy. The pinned `requirements.txt` keeps the build reproducible.

### 8.7 Google Colab (research workflow)
Open `notebook/Spain_Top50_Lifecycle_Colab.ipynb` in Colab and choose *Runtime ▸ Run all*; upload `Atlantic_Spain.csv` when asked. The notebook runs the whole analysis (validation → lifecycles → stages → churn → statistics → KPIs → 13 figures), exports the result tables and a zip of outputs, and its last cell can optionally launch the dashboard (`RUN_DASHBOARD = True`).

### 8.8 Reading the documents
The research paper and executive summary are standard Word files (`.docx`); open them in Microsoft Word, LibreOffice or Google Docs (*File ▸ Open ▸ Upload*).

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment first, or run `python -m streamlit run Spain_Top50_Streamlit_Dashboard.py` |
| PowerShell says scripts are disabled | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt` |
| A pinned version cannot be found or installed | Use Python 3.11+; if it still fails, try `pip install streamlit plotly pandas numpy scipy statsmodels` (newer versions are untested with this project) |
| Page shows "Upload `Atlantic_Spain.csv` in the sidebar" | Put the CSV in the same folder as the `.py` file (or in a `data/` subfolder), or upload it in the sidebar |
| `Port 8501 is already in use` | `streamlit run Spain_Top50_Streamlit_Dashboard.py --server.port 8502` |
| First load feels slow | Expected (~20 s); results are cached afterwards |
| "No data for this combination of filters" | Widen the date range or re-enable stages / album types |
| Error with your own CSV | Check the column names and the `dd-mm-yyyy` date format |

---

## 10. Recommendations

| # | Recommendation | Evidence |
|---|---|---|
| R1 | Treat an album as a **focus track plus a tail** and stagger the tail | Drop entries: 5-day median vs 26; 75% of a typical drop's tracks leave within a week, yet 9 of 18 drops produced a 60+ day track |
| R2 | Use **early rank as the go / no-go signal** (checkpoints at day 3 and day 7) | Top-10 entries: 69% day-30 survival vs 26%–38% lower down |
| R3 | **Front-load marketing**; do not steer by popularity in the first fortnight | Half of songs peak on entry day; popularity peaks a median 10 days later |
| R4 | **Protect the catalogue core** | Songs older than 90 days hold 44% of slots; long-lived songs have a stability index of 0.968 |
| R5 | Plan release calendars around competitor drops; **validate day-of-week effects** before using them | 12 of 17 drops registered on Mondays, possibly a data-compilation artefact |
| R6 | Do not change explicit/clean strategy on lifecycle grounds | No overall difference (1.07×) |
| R7 | **Test** UK/US playbooks with the same pipeline rather than assume them | Cross-market claims cannot be tested with one market |
| R8 | Fix and enrich the data feed | Duplicated day, missing days, zero scores; no release dates, genres or stream counts |

The full reasoning, with actions and monitoring metrics, is in the research paper (section 6).

---

## 11. Assumptions and ambiguities

| Ambiguity in the brief | Assumption |
|---|---|
| 50 entries per day, but one day has 100 rows | Two scrapes of one day; the later row per position is kept |
| Missing calendar days | Not imputed; excluded from consecutive-day transitions |
| Stage thresholds ("rank improving", "stable mid-rank") | 7-day rank slope ±0.75 positions/day; sensitivity analysis in the paper (Appendix B) |
| "Exit date" | Last day listed; re-entries are tracked as separate spells |
| "Total days on playlist" | Distinct days listed, not calendar span |
| Explicit Content Lifecycle Score and Retention Stability Index have no formula | Defined in [section 6](#6-kpis) |
| Filter list repeats "Explicit content toggle" | Treated as one control (All / Explicit only / Clean only) |
| Statements about Spain versus the UK and US | Not testable with one market; reported as such |

---

## 12. Limitations

- **Observational data:** associations are not causal (album-drop entry may reflect fan curiosity, algorithms or marketing intensity).
- **One market, 559 days:** no cross-market comparison and only one Christmas season.
- **Censoring:** 50 songs enter before and 50 survive past the window; mean-based KPIs use completed lifecycles and lean toward shorter stays. Kaplan–Meier and restricted-mean analyses, and a subset with 90 days of follow-up, give the same conclusions.
- **Top-50 truncation:** leaving the list does not mean disappearing from listening.
- **Track identity** is inferred from title and primary artist; re-releases may be merged.
- **Analyst-defined thresholds:** stage slopes and the three-track album-drop rule (the drop finding is stable for 2–5 tracks).
- **Popularity** is a lagging, smoothed score, not a stream count.
- **Multiple testing:** only the album-drop, single-vs-album (overall) and entry-rank results clear the Bonferroni threshold.

---

## 13. Verification and reproducibility

- The six KPIs were **independently recomputed from the raw CSV** with separate code and matched exactly.
- The dashboard was tested headlessly across all filters (explicit, album type, stage, date ranges including a single week), against a live server, and in a clean virtual environment.
- Parameters (stage slope threshold, album-drop size, trend window) are declared at the top of the analysis section of the dashboard file.
- The research paper and executive summary are generated from the same analysis outputs, so their numbers match the dashboard.

---

## 14. Tech stack

Python 3.11 · pandas · NumPy · SciPy · statsmodels · Plotly · Streamlit · Matplotlib (notebook figures) · Jupyter / Google Colab

---

## 15. Future work

- Add release dates, genres, stream counts and artist origin to measure true time-to-peak and domestic versus international share.
- Run the same pipeline on UK and US Top 50 data for a direct cross-market comparison.
- Pilot the focus-track-plus-staggered-tail release strategy and measure it against the drop baseline.
- Add automated snapshot-integrity checks (exactly 50 rows, no gaps) to the data feed.
- Add a survival-model view (hazard by entry rank and release type) to the dashboard.

---

## Author

**Swaraj Dattaram Tawade** — Data Scientist, Navi Mumbai, India
Email: swarajtawade198@gmail.com

*Dataset and project brief supplied by Unified Mentor for Atlantic Recording Corporation.*
