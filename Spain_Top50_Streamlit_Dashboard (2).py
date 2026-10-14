"""
Spain Top 50 - Content Maturity, Release Lifecycle & Playlist Rotation  |  Streamlit dashboard (single file)
Atlantic Recording Corporation | Unified Mentor project

This one file contains BOTH the analysis engine (validation, lifecycles, stages, churn, statistics, KPIs)
and the Streamlit dashboard, so nothing else is needed except the data file.

Setup (Windows / macOS / Linux, Python 3.11+):
    python -m venv .venv
    .venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
    python -m pip install --upgrade pip
    pip install streamlit plotly pandas numpy scipy statsmodels
    python -m streamlit run Spain_Top50_Streamlit_Dashboard.py

    Note: use "python -m streamlit run ..." if the bare "streamlit" command is
    not on your PATH (common on Windows).

Data: put Atlantic_Spain.csv in the same folder as this file (or in a "data" subfolder), or upload it from the
dashboard sidebar.

Key design decisions of the engine (all documented in the research paper):
  * Track identity = normalised song title + normalised PRIMARY artist.
  * A date with more than 50 rows (duplicate scrape) keeps the LAST row per (date, position).
  * popularity == 0 is treated as "not yet scored" (missing), not as a score.
  * Missing calendar days are not imputed; transitions across a gap are excluded from churn averages.
  * The panel is left- and right-censored; duration KPIs use completed lifecycles and survival curves
    treat still-listed songs as censored.
"""
from __future__ import annotations


# =====================================================================================================
# PART 1 - ANALYSIS ENGINE
# =====================================================================================================
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Parameters (documented; change here to run sensitivity analyses)
# --------------------------------------------------------------------------
PLAYLIST_SIZE = 50
NEW_ENTRY_DAYS = 7          # brief: New Entry = <= 7 days on playlist
PEAK_RANK = 10              # brief: Peak Phase = Top-10 stability
TREND_WINDOW = 7            # trailing observed days used for rank trend
TREND_MIN_OBS = 4           # minimum observations for a slope
SLOPE_THRESHOLD = 0.75      # positions per day (|slope| above => Growth / Decline)
BURST_MIN_TRACKS = 3        # >=3 same-artist debuts on one day = album-drop burst
STAGES = ["New Entry", "Growth Phase", "Peak Phase", "Mature Phase", "Decline Phase"]
DATE_FORMAT = "%d-%m-%Y"


# --------------------------------------------------------------------------
# 1. Loading, validation & normalisation
# --------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower().strip())


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robustly convert strings/ints/bools to real booleans ('False' string -> False)."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(bool)
    return (s.astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y", "t"]))


def load_and_clean(source) -> tuple[pd.DataFrame, dict]:
    """Load the raw CSV (path or file-like) and return (clean_daily, qa_report)."""
    raw = pd.read_csv(source)
    qa: dict = {}
    qa["raw_rows"] = int(len(raw))
    qa["columns"] = list(raw.columns)
    qa["missing_values"] = {c: int(v) for c, v in raw.isna().sum().items() if v}
    qa["exact_duplicate_rows"] = int(raw.duplicated().sum())

    df = raw.copy()
    df["_row"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT)
    qa["first_date"] = df["date"].min()
    qa["last_date"] = df["date"].max()

    # --- exactly 50 entries per day -------------------------------------
    per_day = df.groupby("date").size()
    over = per_day[per_day > PLAYLIST_SIZE]
    under = per_day[per_day < PLAYLIST_SIZE]
    qa["days_over_50"] = {str(k.date()): int(v) for k, v in over.items()}
    qa["days_under_50"] = {str(k.date()): int(v) for k, v in under.items()}
    qa["dup_date_position_rows"] = int(df.duplicated(["date", "position"]).sum())
    df = (df.sort_values("_row")
            .drop_duplicates(["date", "position"], keep="last")
            .sort_values(["date", "position"]).reset_index(drop=True))
    ok = df.groupby("date")["position"].apply(lambda s: sorted(s) == list(range(1, PLAYLIST_SIZE + 1)))
    qa["days_valid_after_cleaning"] = int(ok.sum())
    qa["days_total"] = int(len(ok))
    qa["rows_after_cleaning"] = int(len(df))

    # --- calendar completeness -------------------------------------------
    full = pd.date_range(df["date"].min(), df["date"].max())
    missing_days = sorted(set(full) - set(df["date"]))
    qa["missing_calendar_days"] = [str(d.date()) for d in missing_days]
    qa["calendar_span_days"] = int(len(full))

    # --- popularity glitches ---------------------------------------------
    zero_pop = df["popularity"].eq(0)
    srt = df.sort_values(["date", "position"])
    grp_key = (srt["song"].map(_norm_text) + "|" +
               srt["artist"].map(lambda a: _norm_text(str(a).split("&")[0])))
    first_two_sorted = srt.groupby(grp_key).cumcount().lt(2)
    first_two = first_two_sorted.reindex(df.index).fillna(False)
    qa["popularity_zero_share_in_first_2_days"] = float((zero_pop & first_two).sum() / max(zero_pop.sum(), 1))
    qa["popularity_zero_rows"] = int(zero_pop.sum())
    qa["popularity_zero_days"] = int(df.loc[zero_pop, "date"].nunique())
    df.loc[zero_pop, "popularity"] = np.nan
    qa["popularity_low_outliers_kept(<30)"] = int((df["popularity"] < 30).sum())

    # --- duration outliers (flagged, kept: plausible tracks) --------------
    q1, q3 = df["duration_ms"].quantile([.25, .75])
    iqr = q3 - q1
    qa["duration_iqr_outliers"] = int(((df["duration_ms"] < q1 - 1.5 * iqr) |
                                       (df["duration_ms"] > q3 + 1.5 * iqr)).sum())
    qa["duration_min_max_sec"] = (round(df["duration_ms"].min() / 1000), round(df["duration_ms"].max() / 1000))

    # --- naming normalisation / track key ----------------------------------
    qa["raw_song_artist_pairs"] = int(df[["song", "artist"]].drop_duplicates().shape[0])
    df["song_norm"] = df["song"].map(_norm_text)
    df["artist_norm"] = df["artist"].map(_norm_text)
    df["primary_artist"] = df["artist"].map(lambda s: _norm_text(str(s).split("&")[0]))
    df["key"] = df["song_norm"] + "|" + df["primary_artist"]
    qa["unique_tracks_after_normalisation"] = int(df["key"].nunique())
    qa["same_day_key_collisions"] = int(df.groupby(["date", "key"]).size().gt(1).sum())

    # display names = most common raw spelling
    disp_song = df.groupby("key")["song"].agg(lambda s: s.mode().iloc[0])
    disp_art = df.groupby("key")["artist"].agg(lambda s: s.mode().iloc[0])
    df["song_display"] = df["key"].map(disp_song)
    df["artist_display"] = df["key"].map(disp_art)
    df["label"] = df["song_display"] + " — " + df["artist_display"]

    df["is_explicit"] = _to_bool_series(df["is_explicit"])
    df["explicit_label"] = np.where(df["is_explicit"], "Explicit", "Clean")
    df["album_type"] = df["album_type"].astype(str).str.lower().str.strip()
    qa["album_type_values"] = df["album_type"].value_counts().to_dict()
    qa["single_with_multi_tracks_rows"] = int(((df["album_type"] == "single") & (df["total_tracks"] > 1)).sum())
    qa["tracks_with_changing_attributes"] = {
        c: int(v) for c, v in df.groupby("key")[["is_explicit", "album_type", "total_tracks", "duration_ms"]]
        .nunique().gt(1).sum().items()}
    df = df.drop(columns=["_row", "album_cover_url"], errors="ignore")
    return df, qa


# --------------------------------------------------------------------------
# 2. Song-day panel: spells, age, rank trend, lifecycle stage
# --------------------------------------------------------------------------
def _trailing_stats(x: np.ndarray, y: np.ndarray, win: int, min_obs: int):
    """Trailing OLS slope of y on x and trailing median of y (window of `win` obs)."""
    n = len(y)
    slope = np.full(n, np.nan)
    med = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - win + 1)
        xs, ys = x[lo:i + 1], y[lo:i + 1]
        if len(ys) >= 3:
            med[i] = np.median(ys)
        if len(ys) >= min_obs:
            xm, ym = xs.mean(), ys.mean()
            slope[i] = ((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum()
    return slope, med


def build_song_day(daily: pd.DataFrame, slope_threshold: float = SLOPE_THRESHOLD) -> pd.DataFrame:
    """Add spell, age, rank-trend, stage and entry/exit flags to the daily panel."""
    sd = daily.sort_values(["key", "date"]).reset_index(drop=True).copy()
    dates = np.sort(sd["date"].unique())
    didx = {d: i for i, d in enumerate(dates)}
    sd["didx"] = sd["date"].map(didx)
    start, end = dates[0], dates[-1]

    # contiguous spells (consecutive OBSERVED snapshots)
    gap = sd.groupby("key")["didx"].diff().fillna(1).ne(1)
    sd["spell"] = gap.groupby(sd["key"]).cumsum().astype(int) + 1
    sd["spell_day"] = sd.groupby(["key", "spell"]).cumcount() + 1
    first_seen = sd.groupby("key")["date"].transform("min")
    sd["first_seen"] = first_seen
    sd["age_days"] = (sd["date"] - first_seen).dt.days
    sd["left_censored"] = first_seen.eq(start)          # already listed on day 1
    sd["is_first_spell"] = sd["spell"].eq(1)

    # entry / exit flags (calendar transitions)
    sd["entry_flag"] = sd["spell_day"].eq(1) & ~(sd["left_censored"] & sd["is_first_spell"])
    sd["entry_flag"] &= sd["date"].ne(start)
    sd["return_flag"] = sd["entry_flag"] & sd["spell"].gt(1)
    last_in_spell = sd.groupby(["key", "spell"])["didx"].transform("max").eq(sd["didx"])
    sd["exit_after_flag"] = last_in_spell & sd["date"].ne(end)
    next_date = pd.Series(dates[1:], index=dates[:-1])
    sd["exit_date"] = pd.NaT
    sd.loc[sd["exit_after_flag"], "exit_date"] = sd.loc[sd["exit_after_flag"], "date"].map(next_date)
    sd["exit_date"] = pd.to_datetime(sd["exit_date"])

    # rank trend / stability within spells
    slopes = np.full(len(sd), np.nan)
    meds = np.full(len(sd), np.nan)
    for (_, _), idx in sd.groupby(["key", "spell"]).indices.items():
        s, m = _trailing_stats(sd["didx"].values[idx].astype(float), sd["position"].values[idx].astype(float),
                               TREND_WINDOW, TREND_MIN_OBS)
        slopes[idx], meds[idx] = s, m
    sd["rank_slope"] = slopes            # positions/day; negative = improving
    sd["rank_median7"] = meds
    sd["rank_change"] = sd.groupby(["key", "spell"])["position"].diff()
    sd["pop_change7"] = sd["popularity"] - sd.groupby(["key", "spell"])["popularity"].shift(TREND_WINDOW)

    sd["stage"] = classify_stage(sd, slope_threshold)
    sd["age_bucket"] = pd.cut(sd["age_days"], [-1, 7, 30, 90, 10_000],
                              labels=["0–7 d", "8–30 d", "31–90 d", ">90 d"])
    sd["month"] = sd["date"].dt.to_period("M").astype(str)
    return sd


def classify_stage(sd: pd.DataFrame, slope_threshold: float = SLOPE_THRESHOLD) -> pd.Series:
    """Rule-based stage per song-day. Order of precedence:
    New Entry (<=7 days into a spell whose entry was observed) >
    Peak (7-day median rank in Top 10) > Growth (rank improving) > Decline (rank deteriorating) > Mature."""
    new = sd["spell_day"].le(NEW_ENTRY_DAYS) & ~(sd["left_censored"] & sd["is_first_spell"])
    ref_rank = sd["rank_median7"].fillna(sd["position"])
    peak = ref_rank.le(PEAK_RANK)
    growth = sd["rank_slope"].le(-slope_threshold)
    decline = sd["rank_slope"].ge(slope_threshold)
    stage = np.select([new, peak, growth, decline],
                      ["New Entry", "Peak Phase", "Growth Phase", "Decline Phase"], default="Mature Phase")
    return pd.Series(pd.Categorical(stage, categories=STAGES, ordered=True), index=sd.index)


# --------------------------------------------------------------------------
# 3. Song lifecycle table
# --------------------------------------------------------------------------
def _mode(s: pd.Series):
    m = s.mode()
    return m.iloc[0] if len(m) else np.nan


def build_lifecycles(sd: pd.DataFrame) -> pd.DataFrame:
    start, end = sd["date"].min(), sd["date"].max()
    g = sd.groupby("key", sort=False)
    lc = g.agg(song=("song_display", "first"), artist=("artist_display", "first"),
               entry_date=("date", "min"), exit_date=("date", "max"),
               days_on_playlist=("date", "nunique"),
               n_spells=("spell", "max"),
               peak_position=("position", "min"),
               mean_position=("position", "mean"),
               entry_position=("position", "first"),
               exit_position=("position", "last"),
               is_explicit=("is_explicit", _mode),
               album_type=("album_type", _mode),
               total_tracks=("total_tracks", _mode),
               duration_ms=("duration_ms", _mode),
               peak_popularity=("popularity", "max"),
               left_censored=("left_censored", "first")).reset_index()
    lc["span_days"] = (lc["exit_date"] - lc["entry_date"]).dt.days + 1
    lc["reentries"] = lc["n_spells"] - 1
    lc["right_censored"] = lc["exit_date"].eq(end)
    lc["incident"] = ~lc["left_censored"]
    lc["completed"] = lc["incident"] & ~lc["right_censored"]

    # time to peak (first day at the best position)
    pk = (sd.merge(lc[["key", "peak_position"]], on="key")
            .query("position == peak_position").groupby("key")["date"].min().rename("peak_date"))
    lc = lc.merge(pk, on="key")
    lc["time_to_peak"] = (lc["peak_date"] - lc["entry_date"]).dt.days
    lc["days_top10"] = lc["key"].map(sd[sd["position"] <= 10].groupby("key").size()).fillna(0).astype(int)
    lc["days_top3"] = lc["key"].map(sd[sd["position"] <= 3].groupby("key").size()).fillna(0).astype(int)
    lc["reached_top10"] = lc["peak_position"].le(10)

    # rank stability: 1 - mean |daily rank change| / 49
    mad = sd.assign(a=sd["rank_change"].abs()).groupby("key")["a"].mean()
    lc["mean_abs_rank_change"] = lc["key"].map(mad)
    lc["rank_stability"] = 1 - lc["mean_abs_rank_change"] / (PLAYLIST_SIZE - 1)

    # popularity dynamics
    pop = sd.dropna(subset=["popularity"])
    first3 = pop.groupby("key").head(3).groupby("key")["popularity"].mean()
    last3 = pop.groupby("key").tail(3).groupby("key")["popularity"].mean()
    lc["pop_entry"] = lc["key"].map(first3)
    lc["pop_last"] = lc["key"].map(last3)
    lc["pop_change"] = lc["pop_last"] - lc["pop_entry"]
    pp = (pop.merge(lc[["key", "peak_popularity"]], on="key").query("popularity == peak_popularity")
             .groupby("key")["date"].min().rename("peak_pop_date"))
    lc = lc.merge(pp, on="key", how="left")
    lc["time_to_peak_pop"] = (lc["peak_pop_date"] - lc["entry_date"]).dt.days
    lc["peak_pop_timing_pct"] = lc["time_to_peak_pop"] / (lc["span_days"] - 1).replace(0, np.nan)

    # stages
    lc["final_stage"] = lc["key"].map(sd.groupby("key")["stage"].last())
    lc["dominant_stage"] = lc["key"].map(sd.groupby("key")["stage"].agg(
        lambda s: s.value_counts().idxmax()))
    for st in STAGES:
        lc["days_" + st.split()[0].lower()] = lc["key"].map(
            sd[sd["stage"] == st].groupby("key").size()).fillna(0).astype(int)

    # buckets
    lc["explicit_label"] = np.where(lc["is_explicit"].astype(bool), "Explicit", "Clean")
    lc["album_size_bucket"] = pd.cut(lc["total_tracks"], [0, 1, 6, 12, 20, 1000],
                                     labels=["1 (single)", "2–6", "7–12", "13–20", "21+"])
    q = lc["duration_ms"].quantile([0, .25, .5, .75, 1]).values
    lbl = [f"{q[i]/60000:.1f}–{q[i+1]/60000:.1f} min" for i in range(4)]
    lc["duration_bucket"] = pd.cut(lc["duration_ms"], q, labels=lbl, include_lowest=True)
    lc["duration_min"] = lc["duration_ms"] / 60000
    # release-burst flag: >=3 tracks of the same primary artist debut on the same day (album drop)
    lc["primary_artist"] = lc["key"].str.split("|").str[-1]
    first_entries = sd[sd["entry_flag"] & sd["spell"].eq(1)]
    cluster = first_entries.groupby(["date", "primary_artist"]).size().rename("entry_cluster_size")
    lc = lc.merge(cluster, left_on=["entry_date", "primary_artist"], right_index=True, how="left")
    lc["entry_cluster_size"] = lc["entry_cluster_size"].where(lc["incident"])
    lc["burst_entry"] = lc["entry_cluster_size"].ge(BURST_MIN_TRACKS)
    lc["lifespan_class"] = pd.cut(lc["days_on_playlist"], [0, 7, 30, 90, 10_000],
                                  labels=["Flash (≤7 d)", "Short (8–30 d)", "Medium (31–90 d)", "Long (>90 d)"])
    lc["entry_month"] = lc["entry_date"].dt.to_period("M").astype(str)
    return lc


# --------------------------------------------------------------------------
# 4. Flow / churn analytics
# --------------------------------------------------------------------------
def build_flows(sd: pd.DataFrame, size_of_full_list: bool = False) -> pd.DataFrame:
    """Daily entries, exits, churn rate and rank volatility for the rows supplied
    (works on any filtered subset of the song-day panel).

    churn_rate = entries / songs on the day (=/50 when nothing is filtered).
    Transitions that span a missing calendar day are flagged churn_valid=False.
    """
    d = pd.DataFrame({"date": np.sort(sd["date"].unique())})
    d["songs"] = d["date"].map(sd.groupby("date").size())
    d["entries"] = d["date"].map(sd[sd["entry_flag"]].groupby("date").size()).fillna(0).astype(int)
    d["returns"] = d["date"].map(sd[sd["return_flag"]].groupby("date").size()).fillna(0).astype(int)
    d["debuts"] = d["entries"] - d["returns"]
    d["exits"] = d["date"].map(sd[sd["exit_date"].notna()].groupby("exit_date").size()).fillna(0).astype(int)
    denom = PLAYLIST_SIZE if size_of_full_list else d["songs"].replace(0, np.nan)
    d["churn_rate"] = d["entries"] / denom
    d["exit_rate"] = d["exits"] / denom
    d["rank_volatility"] = d["date"].map(sd.assign(a=sd["rank_change"].abs()).groupby("date")["a"].mean())
    d["span_days"] = d["date"].diff().dt.days
    d["churn_valid"] = d["span_days"].eq(1)
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["weekday"] = d["date"].dt.day_name()
    # day-to-day list similarity (Jaccard) from entries: |A∩B| = 50 - entries
    d["retention"] = 1 - d["churn_rate"]
    return d


def monthly_rotation(flows: pd.DataFrame) -> pd.DataFrame:
    v = flows[flows["churn_valid"]]
    m = v.groupby("month").agg(days=("date", "size"), avg_entries=("entries", "mean"),
                               avg_churn_rate=("churn_rate", "mean"),
                               avg_rank_volatility=("rank_volatility", "mean"),
                               churn_std=("churn_rate", "std")).reset_index()
    # version-agnostic debut-share computation
    sums = v.groupby("month")[["debuts", "entries"]].sum()
    debut_share = sums["debuts"] / sums["entries"].replace(0, np.nan)
    m["debut_share"] = m["month"].map(debut_share)
    return m


def release_events(sd: pd.DataFrame, lc: pd.DataFrame) -> pd.DataFrame:
    """One row per album-drop burst: date, artist, tracks entering and how they fared.

    Censoring-aware: tracks still listed on the last snapshot are NOT counted as gone.
    share_gone_in_7d is NaN when fewer than 8 days of follow-up exist after the drop."""
    end = sd["date"].max()
    e = lc[lc["burst_entry"].fillna(False)].copy()
    rows = []
    for (dt, art), g in e.groupby(["entry_date", "primary_artist"]):
        obs = int((end - dt).days + 1)
        gone7 = float(((g["days_on_playlist"] <= 7) & (~g["right_censored"])).mean()) if obs >= 8 else np.nan
        rows.append({"date": dt, "artist": g["artist"].mode().iloc[0].split("&")[0].strip(), "tracks_entered": len(g),
                     "best_peak": int(g["peak_position"].min()),
                     "median_days": float(g["days_on_playlist"].median()),
                     "max_days": int(g["days_on_playlist"].max()),
                     "still_listed_now": int(g["right_censored"].sum()),
                     "observed_days": obs,
                     "share_gone_in_7d": gone7,
                     "album_share": float((g["album_type"] == "album").mean())})
    if not rows:
        return pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "artist": pd.Series(dtype="object"),
            "tracks_entered": pd.Series(dtype="int64"),
            "best_peak": pd.Series(dtype="int64"),
            "median_days": pd.Series(dtype="float64"),
            "max_days": pd.Series(dtype="int64"),
            "still_listed_now": pd.Series(dtype="int64"),
            "observed_days": pd.Series(dtype="int64"),
            "share_gone_in_7d": pd.Series(dtype="float64"),
            "album_share": pd.Series(dtype="float64"),
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def tier_turnover(sd: pd.DataFrame) -> pd.DataFrame:
    """Average days a song stays at each rank band & share of entries landing in each band."""
    bands = pd.cut(sd["position"], [0, 10, 25, 40, 50], labels=["1–10", "11–25", "26–40", "41–50"])
    out = sd.assign(band=bands).groupby("band", observed=True).agg(
        rows=("position", "size"),
        entries=("entry_flag", "sum"),
        avg_abs_daily_move=("rank_change", lambda s: s.abs().mean())).reset_index()
    out["entry_share"] = out["entries"] / out["entries"].sum()
    return out


# --------------------------------------------------------------------------
# 5. Statistics helpers
# --------------------------------------------------------------------------
def km_curve(durations, events):
    """Kaplan-Meier estimator. events=1 if exit observed, 0 if still listed (censored)."""
    t = np.asarray(durations, float)
    e = np.asarray(events, int)
    times = np.unique(t[e == 1])
    surv, s = [], 1.0
    for ti in times:
        n = (t >= ti).sum()
        dd = ((t == ti) & (e == 1)).sum()
        s *= 1 - dd / n
        surv.append(s)
    return pd.DataFrame({"time": np.r_[0, times], "survival": np.r_[1.0, surv]})


def km_median(curve: pd.DataFrame):
    below = curve[curve["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else np.nan


def logrank_test(t1, e1, t2, e2):
    """Two-sample log-rank test. Returns (chi2, p)."""
    t = np.r_[t1, t2].astype(float); e = np.r_[e1, e2].astype(int)
    g = np.r_[np.zeros(len(t1)), np.ones(len(t2))]
    O = E = V = 0.0
    for ti in np.unique(t[e == 1]):
        at_risk = t >= ti
        n = at_risk.sum(); n1 = (at_risk & (g == 0)).sum()
        d = ((t == ti) & (e == 1)).sum(); d1 = ((t == ti) & (e == 1) & (g == 0)).sum()
        O += d1; E += d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = (O - E) ** 2 / V if V > 0 else np.nan
    return chi2, float(stats.chi2.sf(chi2, 1)) if V > 0 else (np.nan, np.nan)


def rmst(durations, events, tau: float = 90.0) -> float:
    """Restricted mean survival time up to tau days (area under the Kaplan-Meier curve);
    censoring-robust measure of 'average days on playlist within the first tau days'."""
    c = km_curve(durations, events)
    t, sv = c["time"].values, c["survival"].values
    m = t < tau
    t, sv = t[m], sv[m]
    return float((sv * np.diff(np.r_[t, tau])).sum())


def rmst_diff(a_dur, a_ev, b_dur, b_ev, tau: float = 90.0, n_boot: int = 1000, seed: int = 7):
    """RMST(a) - RMST(b) with percentile bootstrap 95% CI."""
    a_dur, a_ev, b_dur, b_ev = map(np.asarray, (a_dur, a_ev, b_dur, b_ev))
    est = rmst(a_dur, a_ev, tau) - rmst(b_dur, b_ev, tau)
    if n_boot <= 0:
        return est, np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        ia, ib = rng.integers(0, len(a_dur), len(a_dur)), rng.integers(0, len(b_dur), len(b_dur))
        out.append(rmst(a_dur[ia], a_ev[ia], tau) - rmst(b_dur[ib], b_ev[ib], tau))
    return est, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bootstrap_ratio(a, b, stat=np.mean, n_boot=4000, seed=42):
    """Point estimate and 95% bootstrap CI of stat(a)/stat(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    est = stat(a) / stat(b) if stat(b) != 0 else np.nan
    bs = []
    for _ in range(n_boot):
        sa, sb = rng.choice(a, len(a)), rng.choice(b, len(b))
        denom = stat(sb)
        bs.append(stat(sa) / denom if denom != 0 else np.nan)
    bs = np.asarray(bs, float)
    return est, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


# --------------------------------------------------------------------------
# 6. KPIs
# --------------------------------------------------------------------------
def compute_kpis(lc: pd.DataFrame, flows: pd.DataFrame) -> dict:
    """All six KPIs required by the brief."""
    comp = lc[lc["completed"]]
    inc = lc[lc["incident"]]
    k = {}
    k["avg_days_completed"] = comp["days_on_playlist"].mean() if len(comp) else np.nan
    k["median_days_completed"] = comp["days_on_playlist"].median() if len(comp) else np.nan
    k["avg_days_all"] = lc["days_on_playlist"].mean()
    k["entry_to_peak_mean"] = inc["time_to_peak"].mean() if len(inc) else np.nan
    k["entry_to_peak_median"] = inc["time_to_peak"].median() if len(inc) else np.nan
    v = flows[flows["churn_valid"]]
    k["churn_rate"] = v["churn_rate"].mean() if len(v) else np.nan
    k["avg_daily_entries"] = v["entries"].mean() if len(v) else np.nan
    k["rsi"] = lc["rank_stability"].mean()
    k["rank_volatility"] = v["rank_volatility"].mean() if len(v) else np.nan
    ex, cl = comp[comp["is_explicit"].astype(bool)], comp[~comp["is_explicit"].astype(bool)]
    k["explicit_score"] = ex["days_on_playlist"].mean() / cl["days_on_playlist"].mean() if len(ex) and len(cl) else np.nan
    si, al = comp[comp["album_type"] == "single"], comp[comp["album_type"] == "album"]
    k["single_album_ratio"] = si["days_on_playlist"].mean() / al["days_on_playlist"].mean() if len(si) and len(al) else np.nan
    k["n_tracks"] = len(lc)
    k["n_completed"] = len(comp)
    return k


def run_pipeline(source):
    """Convenience wrapper: returns dict with every analytical table."""
    daily, qa = load_and_clean(source)
    sd = build_song_day(daily)
    lc = build_lifecycles(sd)
    flows = build_flows(sd)
    return {"daily": daily, "qa": qa, "sd": sd, "lc": lc, "flows": flows,
            "monthly": monthly_rotation(flows), "kpis": compute_kpis(lc, flows),
            "events": release_events(sd, lc)}


# --------------------------------------------------------------------------
# 7. Findings
# --------------------------------------------------------------------------
def compare_groups(lc: pd.DataFrame, col: str, a, b, label: str, n_boot: int = 4000, n_boot_rmst: int = 1000) -> dict:
    """Longevity comparison of two groups of INCIDENT songs."""
    inc = lc[lc["incident"]].copy()
    inc["event"] = (~inc["right_censored"]).astype(int)
    A, B = inc[inc[col] == a], inc[inc[col] == b]
    chi2, p_lr = logrank_test(A["days_on_playlist"], A["event"], B["days_on_playlist"], B["event"])
    ca, cb = A[A["completed"]], B[B["completed"]]
    if len(ca) and len(cb):
        mw = stats.mannwhitneyu(ca["days_on_playlist"], cb["days_on_playlist"])
        p_mw = float(mw.pvalue)
    else:
        p_mw = np.nan
    est, lo, hi = bootstrap_ratio(ca["days_on_playlist"], cb["days_on_playlist"], n_boot=n_boot)
    rd, rlo, rhi = rmst_diff(A["days_on_playlist"], A["event"], B["days_on_playlist"], B["event"], 90, n_boot_rmst)
    return {"rmst90_a": rmst(A["days_on_playlist"], A["event"]), "rmst90_b": rmst(B["days_on_playlist"], B["event"]),
            "rmst90_diff": rd, "rmst90_lo": rlo, "rmst90_hi": rhi, "label": label, "a": str(a), "b": str(b),
            "n_a": len(A), "n_b": len(B),
            "km_median_a": km_median(km_curve(A["days_on_playlist"], A["event"])),
            "km_median_b": km_median(km_curve(B["days_on_playlist"], B["event"])),
            "mean_a": ca["days_on_playlist"].mean() if len(ca) else np.nan,
            "mean_b": cb["days_on_playlist"].mean() if len(cb) else np.nan,
            "median_a": ca["days_on_playlist"].median() if len(ca) else np.nan,
            "median_b": cb["days_on_playlist"].median() if len(cb) else np.nan,
            "ratio": est, "ci_lo": lo, "ci_hi": hi, "p_logrank": p_lr, "p_mwu": p_mw}


def _km_at(curve, t):
    sub = curve[curve["time"] <= t]
    return float(sub["survival"].iloc[-1]) if len(sub) else np.nan


def _safe_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(np.isnan(y)) or np.all(np.isnan(x)):
        return (np.nan, np.nan)
    try:
        r = stats.spearmanr(x, y)
        return (float(r[0]), float(r[1]))
    except Exception:
        return (np.nan, np.nan)


def _safe_kruskal(*arrays):
    arrays = [np.asarray(a, float) for a in arrays if len(a) > 0]
    if len(arrays) < 2:
        return np.nan
    try:
        return float(stats.kruskal(*arrays).pvalue)
    except Exception:
        return np.nan


def compute_findings(R: dict) -> dict:
    sd, lc, fl, ev, qa = R["sd"], R["lc"], R["flows"], R["events"], R["qa"]
    F: dict = {"qa": qa}
    v = fl[fl["churn_valid"]]
    inc = lc[lc["incident"]].copy()
    inc["event"] = (~inc["right_censored"]).astype(int)
    comp = lc[lc["completed"]]
    nb = lc[lc["burst_entry"] != True]  # noqa: E712  (non-burst incl. censored songs)
    burst_dates = set(ev["date"]) if len(ev) else set()

    # ---- overview -------------------------------------------------------
    F["overview"] = {
        "days": qa["days_total"], "first": str(qa["first_date"].date()), "last": str(qa["last_date"].date()),
        "tracks": len(lc), "artists": int(sd["primary_artist"].nunique()),
        "incident": len(inc), "left_censored": int(lc["left_censored"].sum()),
        "right_censored": int(lc["right_censored"].sum()), "completed": len(comp),
        "explicit_slot_share": float(sd["is_explicit"].mean()),
        "single_slot_share": float((sd["album_type"] == "single").mean()),
        "reentry_track_share": float((lc["reentries"] > 0).mean()),
    }

    # ---- survival / lifecycle length -----------------------------------
    km = km_curve(inc["days_on_playlist"], inc["event"])
    F["survival"] = {"median": km_median(km), **{f"S{d}": _km_at(km, d) for d in (7, 14, 30, 90, 180)},
                     "mean_completed": float(comp["days_on_playlist"].mean()) if len(comp) else np.nan,
                     "median_completed": float(comp["days_on_playlist"].median()) if len(comp) else np.nan,
                     "p25_completed": float(comp["days_on_playlist"].quantile(.25)) if len(comp) else np.nan,
                     "p75_completed": float(comp["days_on_playlist"].quantile(.75)) if len(comp) else np.nan,
                     "max_days": int(lc["days_on_playlist"].max())}
    cls_songs = lc["lifespan_class"].value_counts(normalize=True).sort_index()
    slot = sd.merge(lc[["key", "lifespan_class"]], on="key").groupby("lifespan_class", observed=True).size() / len(sd)
    x_ = lc["days_on_playlist"].values
    gini = (2 * np.sum(np.arange(1, len(x_) + 1) * np.sort(x_)) / (len(x_) * np.sum(x_))) - (len(x_) + 1) / len(x_) if len(x_) else np.nan
    F["barbell"] = {"songs_share": {str(k): float(x) for k, x in cls_songs.items()},
                    "slotday_share": {str(k): float(x) for k, x in slot.items()},
                    "top10pct_songs_slot_share": float(lc.sort_values("days_on_playlist", ascending=False)
                                                       .head(int(len(lc) * .1))["days_on_playlist"].sum() / len(sd)) if len(sd) else np.nan,
                    "gini_days": float(gini)}

    # ---- entry / peak behaviour ----------------------------------------
    F["entry_peak"] = {
        "entry_to_peak_mean": float(inc["time_to_peak"].mean()) if len(inc) else np.nan,
        "entry_to_peak_median": float(inc["time_to_peak"].median()) if len(inc) else np.nan,
        "peak_on_entry_day_share": float((inc["time_to_peak"] == 0).mean()) if len(inc) else np.nan,
        "peak_within_3d_share": float((inc["time_to_peak"] <= 3).mean()) if len(inc) else np.nan,
        "entry_position_median": float(inc["entry_position"].median()) if len(inc) else np.nan,
        "entered_top10_share": float((inc["entry_position"] <= 10).mean()) if len(inc) else np.nan,
        "entered_41_50_share": float((inc["entry_position"] > 40).mean()) if len(inc) else np.nan,
        "reached_top10_share": float(inc["reached_top10"].mean()) if len(inc) else np.nan,
        "reached_top10_n": int(inc["reached_top10"].sum()),
        "median_peak_position": float(inc["peak_position"].median()) if len(inc) else np.nan,
        "ttp_mean_burst": float(inc[inc["burst_entry"] == True]["time_to_peak"].mean()) if (inc["burst_entry"] == True).any() else np.nan,  # noqa: E712
        "ttp_mean_nonburst": float(inc[inc["burst_entry"] != True]["time_to_peak"].mean()) if (inc["burst_entry"] != True).any() else np.nan,  # noqa: E712
        "ttp_median_nonburst": float(inc[inc["burst_entry"] != True]["time_to_peak"].median()) if (inc["burst_entry"] != True).any() else np.nan,  # noqa: E712
    }
    inc["entry_band"] = pd.cut(inc["entry_position"], [0, 10, 25, 40, 50], labels=["1–10", "11–25", "26–40", "41–50"])
    rows = []
    for b, g in inc.groupby("entry_band", observed=True):
        c_ = km_curve(g["days_on_playlist"], g["event"])
        rows.append({"band": str(b), "n": len(g), "peak_at_entry": float((g["time_to_peak"] == 0).mean()),
                     "km_median": km_median(c_), "S7": _km_at(c_, 7), "S30": _km_at(c_, 30), "S90": _km_at(c_, 90),
                     "burst_share": float((g["burst_entry"] == True).mean())})  # noqa: E712
    F["entry_band_table"] = rows
    f3 = sd[(~sd["left_censored"]) & (sd["spell"] == 1) & (sd["spell_day"] <= 3)].groupby("key").agg(
        r3=("position", "mean"), n=("position", "size"))
    f3 = f3[f3["n"] == 3]
    t = inc.merge(f3, left_on="key", right_index=True)
    t["band"] = pd.cut(t["r3"], [0, 10, 25, 40, 50], labels=["1–10", "11–25", "26–40", "41–50"])
    rows = []
    for b, g in t.groupby("band", observed=True):
        c_ = km_curve(g["days_on_playlist"], g["event"])
        rows.append({"band": str(b), "n": len(g), "km_median": km_median(c_), "S30": _km_at(c_, 30), "S90": _km_at(c_, 90)})
    F["day3_rank_table"] = rows

    # ---- stages ---------------------------------------------------------
    F["stages"] = {
        "slot_share": {k: float(x) for k, x in sd["stage"].value_counts(normalize=True).items()},
        "slot_share_postwarmup": {k: float(x) for k, x in sd[~sd["left_censored"]]["stage"].value_counts(normalize=True).items()},
        "sensitivity": {str(th): {k: float(x) for k, x in classify_stage(sd, th).value_counts(normalize=True).items()}
                        for th in (0.5, 0.75, 1.0)},
        "final_stage_share": {k: float(x) for k, x in lc["final_stage"].value_counts(normalize=True).items()},
        "dominant_stage_n": {k: int(x) for k, x in lc["dominant_stage"].value_counts().items()},
        "by_album_type": {a: {k: float(x) for k, x in g["stage"].value_counts(normalize=True).items()} for a, g in sd.groupby("album_type")},
        "by_explicit": {a: {k: float(x) for k, x in g["stage"].value_counts(normalize=True).items()} for a, g in sd.groupby("explicit_label")},
        "popularity_by_stage": {k: float(x) for k, x in sd.groupby("stage", observed=True)["popularity"].mean().items()},
        "pop_change7_by_stage": {k: (None if pd.isna(x) else float(x)) for k, x in sd.groupby("stage", observed=True)["pop_change7"].mean().items()},
        "position_by_stage": {k: float(x) for k, x in sd.groupby("stage", observed=True)["position"].mean().items()},
    }

    # ---- churn / rotation ----------------------------------------------
    b_days, n_days = v[v["date"].isin(burst_dates)], v[~v["date"].isin(burst_dates)]
    top10 = sd[sd["position"] <= 10].groupby("date")["key"].apply(set)
    t10_changes = [len(top10.iloc[i] - top10.iloc[i - 1]) for i in range(1, len(top10))
                   if (top10.index[i] - top10.index[i - 1]).days == 1]
    ex = sd[sd["exit_after_flag"]]
    tier = tier_turnover(sd)
    F["churn"] = {
        "valid_days": len(v), "excluded_transitions": int((~fl["churn_valid"]).sum()),
        "entries_per_day": float(v["entries"].mean()) if len(v) else np.nan,
        "median_entries": float(v["entries"].median()) if len(v) else np.nan,
        "churn_rate": float(v["churn_rate"].mean()) if len(v) else np.nan,
        "churn_std": float(v["churn_rate"].std()) if len(v) else np.nan,
        "zero_entry_days": int((v["entries"] == 0).sum()) if len(v) else 0,
        "zero_entry_share": float((v["entries"] == 0).mean()) if len(v) else np.nan,
        "burst_days": len(b_days), "burst_day_entries_mean": float(b_days["entries"].mean()) if len(b_days) else np.nan,
        "burst_entry_share": float(b_days["entries"].sum() / v["entries"].sum()) if len(v) and v["entries"].sum() else np.nan,
        "nonburst_entries_per_day": float(n_days["entries"].mean()) if len(n_days) else np.nan,
        "nonburst_churn_rate": float(n_days["churn_rate"].mean()) if len(n_days) else np.nan,
        "rank_vol_mean": float(v["rank_volatility"].mean()) if len(v) else np.nan,
        "rank_vol_burst": float(b_days["rank_volatility"].mean()) if len(b_days) else np.nan,
        "rank_vol_nonburst": float(n_days["rank_volatility"].mean()) if len(n_days) else np.nan,
        "debut_share": float(v["debuts"].sum() / v["entries"].sum()) if len(v) and v["entries"].sum() else np.nan,
        "returns": int(v["returns"].sum()) if len(v) else 0, "debuts": int(v["debuts"].sum()) if len(v) else 0,
        "mean_tenure_littles_law": float(PLAYLIST_SIZE / v["entries"].mean()) if len(v) and v["entries"].mean() else np.nan,
        "entry_share_by_band": {str(r["band"]): float(r["entry_share"]) for _, r in tier.iterrows()},
        "avg_abs_move_by_band": {str(r["band"]): float(r["avg_abs_daily_move"]) for _, r in tier.iterrows()},
        "exit_from_41_50_share": float((ex["position"] > 40).mean()) if len(ex) else np.nan,
        "exit_median_position": float(ex["position"].median()) if len(ex) else np.nan,
        "top10_changes_per_day": float(np.mean(t10_changes)) if t10_changes else np.nan,
        "top10_unique_tracks": int(sd[sd["position"] <= 10]["key"].nunique()),
        "number_one_tracks": int(sd[sd["position"] == 1]["key"].nunique()),
        "burst_on_monday": (int((pd.Series(sorted(burst_dates)).dt.dayofweek == 0).sum()) if burst_dates else 0),
        "burst_total": len(burst_dates),
        "weekday_entries_nonburst": {k: float(x) for k, x in n_days.groupby("weekday")["entries"].mean().items()} if len(n_days) else {},
        "weekday_debuts_nonburst": {k: float(x) for k, x in n_days.groupby("weekday")["debuts"].mean().items()} if len(n_days) else {},
        "monthly": R["monthly"].to_dict("records"),
    }
    nbm = v.assign(b=v["date"].isin(burst_dates)) if len(v) else v
    F["churn"]["monthly_nonburst_churn"] = ({k: float(x) for k, x in nbm[~nbm["b"]].groupby("month")["churn_rate"].mean().items()}
                                             if len(nbm) else {})
    if len(R["monthly"]):
        F["churn"]["monthly_churn_min"] = [R["monthly"].sort_values("avg_churn_rate").iloc[0]["month"],
                                           float(R["monthly"]["avg_churn_rate"].min())]
        F["churn"]["monthly_churn_max"] = [R["monthly"].sort_values("avg_churn_rate").iloc[-1]["month"],
                                           float(R["monthly"]["avg_churn_rate"].max())]
    else:
        F["churn"]["monthly_churn_min"] = [None, np.nan]
        F["churn"]["monthly_churn_max"] = [None, np.nan]

    # ---- content attributes --------------------------------------------
    lc2 = lc.copy()
    lc2["burst_label"] = np.where(lc2["burst_entry"] == True, "burst", "non-burst")  # noqa: E712
    F["compare"] = {
        "explicit_all": compare_groups(lc2, "explicit_label", "Explicit", "Clean", "Explicit vs clean (all)"),
        "album_all": compare_groups(lc2, "album_type", "single", "album", "Single vs album (all)"),
        "burst_all": compare_groups(lc2, "burst_label", "burst", "non-burst", "Burst vs non-burst entry"),
    }
    nbl = lc2[lc2["burst_label"] == "non-burst"]
    F["compare"]["explicit_nonburst"] = compare_groups(nbl, "explicit_label", "Explicit", "Clean", "Explicit vs clean (non-burst)")
    F["compare"]["album_nonburst"] = compare_groups(nbl, "album_type", "single", "album", "Single vs album (non-burst)")
    F["compare"]["explicit_single"] = compare_groups(lc2[lc2["album_type"] == "single"], "explicit_label", "Explicit", "Clean", "Explicit vs clean (singles)")
    F["compare"]["explicit_album"] = compare_groups(lc2[lc2["album_type"] == "album"], "explicit_label", "Explicit", "Clean", "Explicit vs clean (albums)")
    inc2 = lc2[lc2["incident"]]
    F["attr_profile"] = {}
    for col in ("explicit_label", "album_type", "burst_label"):
        g = inc2.groupby(col).agg(n=("key", "size"), entry_pos=("entry_position", "median"), peak=("peak_position", "median"),
                                  top10=("reached_top10", "mean"), ttp_median=("time_to_peak", "median"),
                                  ttp_mean=("time_to_peak", "mean"), stability=("rank_stability", "mean"),
                                  top10_days=("days_top10", "mean"), pop_peak=("peak_popularity", "mean"))
        F["attr_profile"][col] = {str(i): {k: float(x) for k, x in r.items()} for i, r in g.iterrows()}
    burst_album_share = (float((lc2[lc2["burst_entry"] == True]["album_type"] == "album").mean())
                          if (lc2["burst_entry"] == True).any() else np.nan)  # noqa: E712
    F["burst_album_overlap"] = {"burst_tracks": int(lc2["burst_entry"].sum()),
                                "burst_album_share": burst_album_share,
                                "album_tracks_in_burst_share": float(lc2[(lc2["album_type"] == "album") & lc2["incident"]]["burst_entry"].mean()) if ((lc2["album_type"] == "album") & lc2["incident"]).any() else np.nan,
                                "explicit_share_single": float(inc2[inc2["album_type"] == "single"]["is_explicit"].mean()) if len(inc2[inc2["album_type"] == "single"]) else np.nan,
                                "explicit_share_album": float(inc2[inc2["album_type"] == "album"]["is_explicit"].mean()) if len(inc2[inc2["album_type"] == "album"]) else np.nan}
    cp = comp.copy()
    cpn = cp[cp["burst_entry"] != True]  # noqa: E712
    dur_tab = inc2.groupby("duration_bucket", observed=True).agg(
        n=("key", "size"), median=("days_on_playlist", "median"), mean=("days_on_playlist", "mean"),
        stability=("rank_stability", "mean"), top10=("reached_top10", "mean"), burst=("burst_entry", "mean")).reset_index()
    dur_tab["duration_bucket"] = dur_tab["duration_bucket"].astype(str)
    size_tab = inc2.groupby("album_size_bucket", observed=True).agg(
        n=("key", "size"), median=("days_on_playlist", "median"), mean=("days_on_playlist", "mean"),
        stability=("rank_stability", "mean"), top10=("reached_top10", "mean"), burst=("burst_entry", "mean")).reset_index()
    size_tab["album_size_bucket"] = size_tab["album_size_bucket"].astype(str)
    sp = lambda x, y: _safe_spearman(x, y)  # noqa: E731
    stab = lc.dropna(subset=["rank_stability"])
    F["duration_size"] = {
        "duration_table": dur_tab.to_dict("records"), "size_table": size_tab.to_dict("records"),
        "spearman_duration_days": sp(cp["duration_ms"], cp["days_on_playlist"]),
        "spearman_duration_days_nonburst": sp(cpn["duration_ms"], cpn["days_on_playlist"]),
        "spearman_tracks_days": sp(cp["total_tracks"], cp["days_on_playlist"]),
        "spearman_tracks_days_nonburst": sp(cpn["total_tracks"], cpn["days_on_playlist"]),
        "spearman_tracks_stability": sp(stab["total_tracks"], stab["rank_stability"]),
        "spearman_duration_stability": sp(stab["duration_ms"], stab["rank_stability"]),
        "kruskal_duration_p": _safe_kruskal(*[g["days_on_playlist"].values for _, g in cp.groupby("duration_bucket", observed=True)]),
        "kruskal_size_p": _safe_kruskal(*[g["days_on_playlist"].values for _, g in cp.groupby("album_size_bucket", observed=True)]),
        "spearman_entrypos_days": sp(cp["entry_position"], cp["days_on_playlist"]),
    }

    # ---- burst sensitivity ---------------------------------------------
    fe = sd[sd["entry_flag"] & sd["spell"].eq(1)].groupby(["date", "primary_artist"]).size().rename("c").reset_index()
    sens = []
    for thr in (2, 3, 4, 5):
        tt = inc.merge(fe, left_on=["entry_date", "primary_artist"], right_on=["date", "primary_artist"], how="left")
        tt["b"] = tt["c"].ge(thr)
        A_, B_ = tt[tt["b"]], tt[~tt["b"]]
        r_ = bootstrap_ratio(A_[A_["completed"]]["days_on_playlist"], B_[B_["completed"]]["days_on_playlist"], n_boot=1500)
        sens.append({"threshold": thr, "n_burst": len(A_),
                     "km_median_burst": km_median(km_curve(A_["days_on_playlist"], A_["event"])),
                     "km_median_other": km_median(km_curve(B_["days_on_playlist"], B_["event"])),
                     "ratio": r_[0], "ci_lo": r_[1], "ci_hi": r_[2]})
    F["burst_sensitivity"] = sens
    F["events"] = (ev.assign(date=ev["date"].dt.strftime("%Y-%m-%d")).to_dict("records") if len(ev) else [])

    # ---- popularity -----------------------------------------------------
    pc = sd[~sd["left_censored"]].groupby(sd["spell_day"].clip(upper=120))["popularity"].mean()
    pp = sd.dropna(subset=["popularity"])
    cm = lc[lc["completed"] & (lc["days_on_playlist"] >= 30)]["key"]
    x = sd[sd["key"].isin(cm)].merge(lc[["key", "peak_popularity"]], on="key")
    if len(x):
        x["k"] = (x["date"] - x.groupby("key")["date"].transform("max")).dt.days
        x["rel"] = x["popularity"] / x["peak_popularity"]
        rel = x[x["k"].between(-45, 0)].groupby("k")["rel"].mean()
    else:
        rel = pd.Series(dtype=float)
    e_pop = comp.dropna(subset=["pop_entry"])
    F["popularity"] = {
        "by_age": {int(d): float(pc.loc[d]) for d in (1, 2, 3, 7, 14, 30, 60, 90, 120) if d in pc.index},
        "peak_pop_days_median": float(inc["time_to_peak_pop"].median()) if len(inc) else np.nan,
        "peak_pop_days_mean": float(inc["time_to_peak_pop"].mean()) if len(inc) else np.nan,
        "peak_pop_on_entry_share": float((inc["time_to_peak_pop"] == 0).mean()) if len(inc) else np.nan,
        "peak_pop_within3_share": float((inc["time_to_peak_pop"] <= 3).mean()) if len(inc) else np.nan,
        "peak_pop_timing_pct_median": float(inc["peak_pop_timing_pct"].median()) if len(inc) else np.nan,
        "rel_to_peak_45d_before_exit": float(rel.loc[-45]) if -45 in rel.index else np.nan,
        "rel_to_peak_at_exit": float(rel.loc[0]) if 0 in rel.index else np.nan,
        "rel_curve": {int(k): float(v_) for k, v_ in rel.items()},
        "decay_songs": int(x["key"].nunique()) if len(x) else 0,
        "spearman_position_popularity": sp(pp["position"], pp["popularity"]),
        "spearman_entry_pop_days": sp(e_pop["pop_entry"], e_pop["days_on_playlist"]),
        "mean_pop_change_completed": float(comp["pop_change"].mean()) if len(comp) else np.nan,
    }

    # ---- concentration, age mix, seasonality ----------------------------
    s_art = sd.groupby("primary_artist").size().sort_values(ascending=False)
    top = s_art.head(5)
    disp = sd.groupby("primary_artist")["artist_display"].agg(lambda s: s.mode().iloc[0].split("&")[0].strip())
    F["concentration"] = {"top5": [(disp[a], float(n / len(sd))) for a, n in top.items()],
                          "top10_share": float(s_art.head(10).sum() / len(sd)) if len(sd) else np.nan,
                          "n_artists": int(len(s_art)),
                          "hhi": float(((s_art / len(sd)) ** 2).sum()) if len(sd) else np.nan}
    steady = sd[sd["date"] >= sd["date"].min() + pd.Timedelta(days=91)]
    if len(steady):
        am = steady.groupby(["month", "age_bucket"], observed=True).size().unstack().fillna(0)
        am = am.div(am.sum(1), axis=0)
        gt90_min = float(am[">90 d"].min()) if ">90 d" in am.columns else np.nan
        gt90_max = float(am[">90 d"].max()) if ">90 d" in am.columns else np.nan
        new_07 = float(am["0–7 d"].mean()) if "0–7 d" in am.columns else np.nan
    else:
        gt90_min = gt90_max = new_07 = np.nan
    F["age_mix"] = {"mean_share": {str(k): float(x) for k, x in steady["age_bucket"].value_counts(normalize=True).items()} if len(steady) else {},
                    "gt90_min": gt90_min, "gt90_max": gt90_max, "new_0_7_mean": new_07}
    xmas = lc[lc["song"].str.contains("Navidad|Christmas", case=False, na=False)]
    F["seasonal"] = {"n": len(xmas), "median_days": float(xmas["days_on_playlist"].median()) if len(xmas) else np.nan,
                     "max_days": int(xmas["days_on_playlist"].max()) if len(xmas) else 0}
    F["rsi"] = {"mean": float(stab["rank_stability"].mean()) if len(stab) else np.nan, "n": len(stab),
                "by_lifespan": {str(k): float(x) for k, x in lc.groupby("lifespan_class", observed=True)["rank_stability"].mean().items()}}

    # ---- extra evidence --------------------------------------------------
    kmb = {}
    for lab, mask in (("burst", inc["burst_entry"] == True), ("other", inc["burst_entry"] != True)):  # noqa: E712
        g = inc[mask]; c_ = km_curve(g["days_on_playlist"], g["event"])
        kmb[lab] = {"n": len(g), "median": km_median(c_), "S7": _km_at(c_, 7), "S30": _km_at(c_, 30), "S90": _km_at(c_, 90)}
    F["km_burst"] = kmb
    drops = sorted(burst_dates)
    ne = inc[inc["burst_entry"] != True].copy()  # noqa: E712
    ne["after_drop"] = ne["entry_date"].map(lambda d: any(0 <= (d - b).days <= 3 for b in drops)) if drops else False
    A_, B_ = ne[ne["after_drop"] == True], ne[ne["after_drop"] == False]  # noqa: E712
    if len(A_) >= 5 and len(B_) >= 5:
        chi, p_ = logrank_test(A_["days_on_playlist"], A_["event"], B_["days_on_playlist"], B_["event"])
        ca_, cb_ = km_curve(A_["days_on_playlist"], A_["event"]), km_curve(B_["days_on_playlist"], B_["event"])
        F["drop_window"] = {"n_after": len(A_), "n_other": len(B_), "km_median_after": km_median(ca_),
                            "km_median_other": km_median(cb_), "S30_after": _km_at(ca_, 30),
                            "S30_other": _km_at(cb_, 30), "p_logrank": p_}
    xs = sd[sd["key"].isin(lc[lc["completed"]]["key"])].copy()
    if len(xs):
        xs["k"] = (xs["date"] - xs.groupby("key")["date"].transform("max")).dt.days
        slope_by_k = xs[xs["k"].between(-45, 0)].groupby("k")["rank_slope"].mean()
    else:
        slope_by_k = pd.Series(dtype=float)
    F["rank_signal"] = {int(k_): float(slope_by_k.loc[k_]) for k_ in (-45, -30, -15, -10, -5, 0) if k_ in slope_by_k.index}

    # ---- robustness ------------------------------------------------------
    end = sd["date"].max()
    fu = inc[inc["entry_date"] <= end - pd.Timedelta(days=90)].copy()
    fu["burst_label"] = np.where(fu["burst_entry"] == True, "burst", "non-burst")  # noqa: E712
    rob = {"followup90_n": len(fu),
           "followup90_completed_share": float(fu["completed"].mean()) if len(fu) else np.nan}
    for key_, col_, a_, b_ in [("explicit", "explicit_label", "Explicit", "Clean"),
                               ("album", "album_type", "single", "album"),
                               ("burst", "burst_label", "burst", "non-burst")]:
        A_, B_ = fu[fu[col_] == a_], fu[fu[col_] == b_]
        ca_, cb_ = A_[A_["completed"]], B_[B_["completed"]]
        _, p_ = logrank_test(A_["days_on_playlist"], A_["event"], B_["days_on_playlist"], B_["event"])
        r_ = bootstrap_ratio(ca_["days_on_playlist"], cb_["days_on_playlist"], n_boot=2000)
        rob[f"followup90_{key_}"] = {"ratio": r_[0], "lo": r_[1], "hi": r_[2], "p_logrank": p_,
                                     "n_a": len(A_), "n_b": len(B_)}
    fn_ = fu[fu["burst_label"] == "non-burst"]
    for key_, col_, a_, b_ in [("explicit_nonburst", "explicit_label", "Explicit", "Clean"),
                               ("album_nonburst", "album_type", "single", "album")]:
        A_, B_ = fn_[fn_[col_] == a_], fn_[fn_[col_] == b_]
        ca_, cb_ = A_[A_["completed"]], B_[B_["completed"]]
        _, p_ = logrank_test(A_["days_on_playlist"], A_["event"], B_["days_on_playlist"], B_["event"])
        r_ = bootstrap_ratio(ca_["days_on_playlist"], cb_["days_on_playlist"], n_boot=2000)
        rob[f"followup90_{key_}"] = {"ratio": r_[0], "lo": r_[1], "hi": r_[2], "p_logrank": p_}
    # multivariable logistic model
    try:
        import statsmodels.api as sm
        f30 = inc[inc["entry_date"] <= end - pd.Timedelta(days=30)].copy()
        f30["y"] = (f30["span_days"] > 30).astype(int)
        if len(f30) > 20 and f30["y"].nunique() > 1:
            def _fit(cols):
                X = pd.DataFrame({
                    "explicit": f30["is_explicit"].astype(int),
                    "single": (f30["album_type"] == "single").astype(int),
                    "album_drop_entry": (f30["burst_entry"] == True).astype(int),  # noqa: E712
                    "duration_min": f30["duration_min"],
                    "entry_rank_per10": f30["entry_position"] / 10,
                })[cols]
                m = sm.Logit(f30["y"], sm.add_constant(X)).fit(disp=0)
                ci = m.conf_int()
                return {c: {"OR": float(np.exp(m.params[c])),
                            "lo": float(np.exp(ci.loc[c, 0])),
                            "hi": float(np.exp(ci.loc[c, 1])),
                            "p": float(m.pvalues[c])} for c in cols}
            rob["logit"] = {"n": len(f30), "base_rate": float(f30["y"].mean()),
                            "m1_single": _fit(["single"]),
                            "m2_single_burst": _fit(["single", "album_drop_entry"]),
                            "m3_full": _fit(["explicit", "single", "album_drop_entry", "duration_min", "entry_rank_per10"])}
        else:
            rob["logit"] = None
    except ImportError:
        rob["logit"] = None
    except Exception:
        rob["logit"] = None
    F["robustness"] = rob

    # ---- trends ---------------------------------------------------------
    full_m = R["monthly"][R["monthly"]["days"] >= 25].reset_index(drop=True)
    nbm_ = nbm[~nbm["b"]].groupby("month")["churn_rate"].mean() if len(nbm) else pd.Series(dtype=float)
    nbm_ = nbm_[nbm_.index.isin(full_m["month"])] if len(full_m) else nbm_
    if len(n_days) and n_days["weekday"].nunique() > 1:
        try:
            kw_w = stats.kruskal(*[g["entries"].values for _, g in n_days.groupby("weekday")])
            kw_p = float(kw_w.pvalue)
        except Exception:
            kw_p = np.nan
    else:
        kw_p = np.nan
    ex_m = sd[sd["date"] >= "2024-06-01"].groupby("month")["is_explicit"].mean()
    rho_all = _safe_spearman(np.arange(len(full_m)), full_m["avg_churn_rate"].values) if len(full_m) else (np.nan, np.nan)
    rho_nb = _safe_spearman(np.arange(len(nbm_)), nbm_.values) if len(nbm_) else (np.nan, np.nan)
    rho_ex = _safe_spearman(np.arange(len(ex_m)), ex_m.values) if len(ex_m) else (np.nan, np.nan)
    F["trends"] = {
        "monthly_trend_rho": rho_all[0], "monthly_trend_p": rho_all[1],
        "monthly_nonburst_trend_rho": rho_nb[0], "monthly_nonburst_trend_p": rho_nb[1],
        "weekday_kruskal_p": kw_p,
        "explicit_share_trend_rho": rho_ex[0], "explicit_share_trend_p": rho_ex[1],
        "explicit_share_2024H2": float(sd[(sd["date"] >= "2024-06-01") & (sd["date"] < "2025-01-01")]["is_explicit"].mean()) if ((sd["date"] >= "2024-06-01") & (sd["date"] < "2025-01-01")).any() else np.nan,
        "explicit_share_2025": float(sd[sd["date"] >= "2025-01-01"]["is_explicit"].mean()) if (sd["date"] >= "2025-01-01").any() else np.nan,
        "monthly_churn_max_over_min": float(full_m["avg_churn_rate"].max() / full_m["avg_churn_rate"].min()) if len(full_m) and full_m["avg_churn_rate"].min() else np.nan,
        "full_month_min": [full_m.sort_values("avg_churn_rate").iloc[0]["month"], float(full_m["avg_churn_rate"].min())] if len(full_m) else [None, np.nan],
        "full_month_max": [full_m.sort_values("avg_churn_rate").iloc[-1]["month"], float(full_m["avg_churn_rate"].max())] if len(full_m) else [None, np.nan],
        "peak_on_entry_nonburst": float((inc[inc["burst_entry"] != True]["time_to_peak"] == 0).mean()) if (inc["burst_entry"] != True).any() else np.nan,  # noqa: E712
        "peak_on_entry_burst": float((inc[inc["burst_entry"] == True]["time_to_peak"] == 0).mean()) if (inc["burst_entry"] == True).any() else np.nan,  # noqa: E712
        "n_primary_tests": 6,
    }
    if len(ev):
        F["events_summary"] = {
            "n": len(ev), "median_share_gone_7d": float(ev["share_gone_in_7d"].median()),
            "n_with_60d_survivor": int((ev["max_days"] >= 60).sum()),
            "n_followup_lt7": int(ev["share_gone_in_7d"].isna().sum()),
            "tracks_in_drops": int(ev["tracks_entered"].sum()),
        }
    else:
        F["events_summary"] = {"n": 0, "median_share_gone_7d": np.nan, "n_with_60d_survivor": 0,
                               "n_followup_lt7": 0, "tracks_in_drops": 0}
    F["kpis"] = R["kpis"]
    F["kpi_ci"] = {"explicit": F["compare"]["explicit_all"], "single_album": F["compare"]["album_all"]}
    return F


# expose the engine as `la.<name>` for the dashboard code below
import types as _types
la = _types.SimpleNamespace(**{k: v for k, v in list(globals().items()) if not k.startswith("__")})


# =====================================================================================================
# PART 2 - STREAMLIT DASHBOARD
# =====================================================================================================
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


warnings.filterwarnings("ignore")
st.set_page_config(page_title="Spain Top 50 – Lifecycle Intelligence", page_icon="🎧", layout="wide")

PALETTE = {"Explicit": "#c0392b", "Clean": "#1f5fa8", "single": "#2e8b57", "album": "#d9822b"}
STAGE_COL = {"New Entry": "#7a5195", "Growth Phase": "#2e8b57", "Peak Phase": "#1f5fa8",
             "Mature Phase": "#9aa5b1", "Decline Phase": "#c0392b"}
st.markdown("""
<style>
.block-container {padding-top: 1.4rem;}
div[data-testid="stMetric"] {background:#f5f7fa; border:1px solid #e1e6ec; padding:10px 14px; border-radius:10px;}
div[data-testid="stMetricValue"] {font-size:1.55rem;}
.small-note {font-size:0.82rem; color:#5c6670;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Building lifecycles …")
def load_pipeline(file_bytes: bytes | None, path: str):
    import io
    src = io.BytesIO(file_bytes) if file_bytes else path
    return la.run_pipeline(src)


@st.cache_data(show_spinner="Computing report findings …")
def load_findings(file_bytes: bytes | None, path: str):
    return la.compute_findings(load_pipeline(file_bytes, path))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = next((p for p in (os.path.join(BASE_DIR, "Atlantic_Spain.csv"),
                                 os.path.join(BASE_DIR, "data", "Atlantic_Spain.csv"))
                     if os.path.exists(p)), os.path.join(BASE_DIR, "Atlantic_Spain.csv"))
st.sidebar.title("🎧 Spain Top 50")
st.sidebar.caption("Lifecycle · Maturity · Rotation")
uploaded = st.sidebar.file_uploader("Dataset (CSV)", type="csv",
                                    help="Leave empty to use Atlantic_Spain.csv next to the app.")
if uploaded is None and not os.path.exists(DEFAULT_PATH):
    st.info("Upload `Atlantic_Spain.csv` in the sidebar to begin.")
    st.stop()
file_bytes = uploaded.getvalue() if uploaded else None
R = load_pipeline(file_bytes, DEFAULT_PATH)
sd, lc, qa = R["sd"], R["lc"], R["qa"]

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
dmin, dmax = sd["date"].min().date(), sd["date"].max().date()
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    d0, d1 = pd.Timestamp(dmin), pd.Timestamp(dmax)
explicit_mode = st.sidebar.radio("Explicit content", ["All", "Explicit only", "Clean only"], horizontal=True)
album_sel = st.sidebar.multiselect("Album type", ["single", "album"], default=["single", "album"])
stage_sel = st.sidebar.multiselect("Lifecycle stage", la.STAGES, default=la.STAGES,
                                   help="Applies to stage, timeline and content-maturity views. "
                                        "Entry/exit flow and churn always use all stages (an entry is by definition a 'New Entry').")
st.sidebar.markdown(
    "<div class='small-note'><b>How filters work.</b> Date, explicit and album filters restrict the daily rows. "
    "Song-level metrics describe songs that appear in the filtered window; their lifecycle is always measured on the full history. "
    "Churn = entries ÷ songs on the day (÷50 when unfiltered).</div>", unsafe_allow_html=True)

mask = sd["date"].between(d0, d1) & sd["album_type"].isin(album_sel or ["single", "album"])
if explicit_mode == "Explicit only":
    mask &= sd["is_explicit"]
elif explicit_mode == "Clean only":
    mask &= ~sd["is_explicit"]
sd_flow = sd[mask]
sd_stage = sd_flow[sd_flow["stage"].isin(stage_sel or la.STAGES)]
if sd_flow.empty or sd_stage.empty:
    st.warning("No data for this combination of filters.")
    st.stop()
lc_f = lc[lc["key"].isin(sd_stage["key"].unique())]
flows = la.build_flows(sd_flow, size_of_full_list=False)
kp = la.compute_kpis(lc_f, flows)
kp_all = R["kpis"]
filtered = (len(sd_flow) != len(sd)) or (len(sd_stage) != len(sd_flow))

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("Spain Top 50 – Content Maturity, Release Lifecycle & Playlist Rotation")
st.caption(f"{d0.date()} → {d1.date()}  ·  {len(sd_stage):,} song-days  ·  {len(lc_f):,} tracks  ·  Atlantic Recording Corporation")


def fmt(x, f="{:.1f}"):
    if x is None:
        return "–"
    try:
        if pd.isna(x):
            return "–"
    except (TypeError, ValueError):
        pass
    try:
        return f.format(x)
    except (TypeError, ValueError):
        return "–"


def delta(k, f="{:+.1f}"):
    if not filtered:
        return None
    try:
        if pd.isna(kp[k]) or pd.isna(kp_all[k]):
            return None
    except (TypeError, ValueError, KeyError):
        return None
    return f.format(kp[k] - kp_all[k]) + " vs full"


c = st.columns(6)
c[0].metric("Avg days on playlist", fmt(kp["avg_days_completed"]), delta("avg_days_completed"), delta_color="off",
            help="Mean days of completed lifecycles (entry and exit both observed). Median is far lower – see Overview.")
c[1].metric("Entry-to-peak (median d)", fmt(kp["entry_to_peak_median"]), delta("entry_to_peak_median"), delta_color="off",
            help="Median days from first appearance to best rank (songs whose entry was observed). Mean: %s d" % fmt(kp["entry_to_peak_mean"]))
c[2].metric("Playlist churn / day", fmt((kp["churn_rate"] * 100) if pd.notna(kp["churn_rate"]) else np.nan, "{:.2f}%"), None,
            help="Mean daily entries ÷ songs on the list (valid consecutive-day transitions).")
c[3].metric("Retention Stability Index", fmt(kp["rsi"], "{:.3f}"), delta("rsi", "{:+.3f}"), delta_color="off",
            help="Mean of 1 − (mean |daily rank move| ÷ 49) across songs. 1 = never moves.")
c[4].metric("Explicit lifecycle score", fmt(kp["explicit_score"], "{:.2f}×"), None,
            help="Mean days explicit ÷ mean days clean (completed lifecycles). >1 = explicit lasts longer. Check significance in 'Content maturity'.")
c[5].metric("Single vs album longevity", fmt(kp["single_album_ratio"], "{:.2f}×"), None,
            help="Mean days single ÷ mean days album (completed lifecycles). Largely driven by album-drop bursts – see 'Content maturity'.")

tabs = st.tabs(["Overview", "Song lifecycle timeline", "Entry vs exit flow", "Lifecycle stages",
                "Content maturity", "Churn analytics", "Data quality & method"])

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tabs[0]:
    a, b = st.columns([1.1, 1])
    with a:
        inc = lc_f[lc_f["incident"]].copy()
        inc["event"] = (~inc["right_censored"]).astype(int)
        if len(inc) >= 10:
            km = la.km_curve(inc["days_on_playlist"], inc["event"])
            fig = go.Figure(go.Scatter(x=km["time"], y=km["survival"], mode="lines",
                                       line=dict(shape="hv", color="#1f5fa8", width=3)))
            fig.add_hline(y=.5, line_dash="dash", line_color="grey")
            fig.update_layout(title=f"How long do new entries survive? (median {la.km_median(km):.0f} days)",
                              height=360, xaxis_title="Days on Top 50", yaxis_title="Share still listed",
                              yaxis_tickformat=".0%", xaxis_range=[0, 200], margin=dict(t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Too few new entries in this selection for a survival curve.")
    with b:
        order = ["Flash (≤7 d)", "Short (8–30 d)", "Medium (31–90 d)", "Long (>90 d)"]
        s_share = lc_f["lifespan_class"].value_counts(normalize=True).reindex(order).fillna(0)
        slot = sd_stage.merge(lc[["key", "lifespan_class"]], on="key").groupby("lifespan_class", observed=True).size()
        slot = (slot / slot.sum()).reindex(order).fillna(0) if slot.sum() else slot
        fig = go.Figure()
        fig.add_bar(x=order, y=s_share.values, name="Share of songs", marker_color="#1f5fa8")
        fig.add_bar(x=order, y=slot.values, name="Share of slot-days", marker_color="#d9822b")
        fig.update_layout(title="Barbell: many short lives, few long ones", barmode="group", height=360,
                          yaxis_tickformat=".0%", margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    e1, e2, e3 = st.columns(3)
    e1.metric("Median days (completed)", fmt(kp["median_days_completed"]))
    e2.metric("Avg new entries / day", fmt(kp["avg_daily_entries"], "{:.2f}"))
    e3.metric("Tracks in selection", f"{len(lc_f):,}")
    with st.expander("Key findings for the full period (from the research paper)", expanded=False):
        try:
            F = load_findings(file_bytes, DEFAULT_PATH)
            st.markdown(f"""
- **Barbell structure.** {F['barbell']['songs_share'].get('Flash (≤7 d)', 0):.0%} of songs are gone within a week and hold only {F['barbell']['slotday_share'].get('Flash (≤7 d)', 0):.0%} of slot-days; the {F['barbell']['songs_share'].get('Long (>90 d)', 0):.0%} that last >90 days hold {F['barbell']['slotday_share'].get('Long (>90 d)', 0):.0%}.
- **Album drops drive rotation.** {F['churn']['burst_days']} album-drop days ({F['churn']['burst_days']/max(F['churn']['valid_days'],1):.0%} of days) produce {F['churn']['burst_entry_share']:.0%} of entries; drop tracks have a median life of {F['compare']['burst_all']['km_median_a']:.0f} days vs {F['compare']['burst_all']['km_median_b']:.0f} days for other entries.
- **Explicit vs clean:** no overall difference (ratio {F['compare']['explicit_all']['ratio']:.2f}, 95% CI {F['compare']['explicit_all']['ci_lo']:.2f}–{F['compare']['explicit_all']['ci_hi']:.2f}).
- **Single vs album:** ratio {F['compare']['album_all']['ratio']:.2f}, but only {F['compare']['album_nonburst']['ratio']:.2f} (not significant) once album-drop entries are set aside.
- **Revolving door at the bottom:** {F['churn']['entry_share_by_band'].get('41–50', 0):.0%} of entries land at ranks 41–50 and {F['churn']['exit_from_41_50_share']:.0%} of exits leave from there.
""")
        except Exception as e:
            st.info(f"Findings summary unavailable: {e}")

# --------------------------------------------------------------------------
# Song lifecycle timeline
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Song lifecycle timeline")
    colA, colB, colC = st.columns([1, 1, 1])
    n_show = colA.slider("Songs on the timeline", 10, 60, 30, step=5)
    sort_by = colB.selectbox("Rank songs by", ["Days on playlist", "Best rank", "Entry date"])
    colour_by = colC.selectbox("Colour by", ["Explicit / clean", "Single / album", "Dominant stage"])
    sub = lc_f.copy()
    sub = sub.sort_values({"Days on playlist": "days_on_playlist", "Best rank": "peak_position",
                           "Entry date": "entry_date"}[sort_by],
                          ascending=(sort_by != "Days on playlist")).head(n_show)
    sub["label"] = sub["song"].str.slice(0, 32) + " – " + sub["artist"].str.slice(0, 22)
    sub["finish"] = sub["exit_date"] + pd.Timedelta(days=1)
    cc = {"Explicit / clean": "explicit_label", "Single / album": "album_type",
          "Dominant stage": "dominant_stage"}[colour_by]
    cmap = PALETTE if cc != "dominant_stage" else STAGE_COL
    fig = px.timeline(sub, x_start="entry_date", x_end="finish", y="label", color=sub[cc].astype(str),
                      color_discrete_map=cmap,
                      hover_data={"days_on_playlist": True, "peak_position": True, "time_to_peak": True,
                                  "label": False, "finish": False})
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=max(380, 17 * n_show + 120), margin=dict(t=30), legend_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Bars span first to last appearance (re-entries included). Songs already listed on day 1 or still listed on the last day are censored at the data edges.")

    st.markdown("##### Single-song trajectory")
    options = lc_f.sort_values("days_on_playlist", ascending=False)
    labels = (options["song"] + " – " + options["artist"]).tolist()
    if labels:
        pick = st.selectbox("Song", labels[:400], index=0)
        key = options.iloc[labels.index(pick)]["key"]
        s = sd[sd["key"] == key].sort_values("date")
        fig = go.Figure()
        for stg in la.STAGES:
            ss = s[s["stage"] == stg]
            if len(ss):
                fig.add_trace(go.Scatter(x=ss["date"], y=ss["position"], mode="markers", name=stg,
                                         marker=dict(color=STAGE_COL[stg], size=7)))
        fig.add_trace(go.Scatter(x=s["date"], y=s["position"], mode="lines",
                                 line=dict(color="#c9d0d8", width=1), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=s["date"], y=s["popularity"], name="Popularity", yaxis="y2",
                                 line=dict(color="#d9822b", width=1.5, dash="dot")))
        fig.update_layout(height=380, yaxis=dict(autorange="reversed", title="Rank (1 = top)", range=[51, 0]),
                          yaxis2=dict(title="Popularity", overlaying="y", side="right", showgrid=False),
                          margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        r = lc[lc["key"] == key].iloc[0]
        m = st.columns(5)
        m[0].metric("Entry", str(r["entry_date"].date()))
        m[1].metric("Exit", str(r["exit_date"].date()))
        m[2].metric("Days listed", int(r["days_on_playlist"]))
        m[3].metric("Peak rank", int(r["peak_position"]))
        m[4].metric("Days to peak", int(r["time_to_peak"]))

# --------------------------------------------------------------------------
# Flow
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Entry vs exit flow")
    fl = flows.copy()
    fig = go.Figure()
    fig.add_bar(x=fl["date"], y=fl["entries"], name="Entries", marker_color="#1f5fa8")
    fig.add_bar(x=fl["date"], y=-fl["exits"], name="Exits", marker_color="#c0392b")
    fig.add_trace(go.Scatter(x=fl["date"], y=fl["entries"].rolling(28, min_periods=14).mean(),
                             name="Entries, 28-day mean", line=dict(color="#d9822b", width=3)))
    fig.update_layout(barmode="relative", height=400, yaxis_title="Songs per day (exits shown negative)",
                      margin=dict(t=20), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("The list is fixed at 50, so entries equal exits in the full data. With attribute filters they differ, showing how that content type gains or loses share.")
    a, b = st.columns(2)
    with a:
        bands = ["1–10", "11–25", "26–40", "41–50"]
        ent_src = sd_flow[sd_flow["entry_flag"]]
        ex_src = sd_flow[sd_flow["exit_after_flag"]]
        ent = pd.cut(ent_src["position"], [0, 10, 25, 40, 50], labels=bands).value_counts(normalize=True).reindex(bands).fillna(0) if len(ent_src) else pd.Series(0, index=bands)
        ex = pd.cut(ex_src["position"], [0, 10, 25, 40, 50], labels=bands).value_counts(normalize=True).reindex(bands).fillna(0) if len(ex_src) else pd.Series(0, index=bands)
        fig = go.Figure()
        fig.add_bar(x=bands, y=ent.values, name="Entry rank", marker_color="#1f5fa8")
        fig.add_bar(x=bands, y=ex.values, name="Last rank before exit", marker_color="#d9822b")
        fig.update_layout(title="Where songs enter and leave", barmode="group", yaxis_tickformat=".0%",
                          height=340, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with b:
        vv = fl[fl["churn_valid"]]
        wd = vv.groupby("weekday")[["entries", "debuts"]].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]) if len(vv) else pd.DataFrame(columns=["entries", "debuts"])
        fig = go.Figure()
        fig.add_bar(x=wd.index, y=wd["entries"].fillna(0), name="All entries", marker_color="#1f5fa8")
        fig.add_bar(x=wd.index, y=wd["debuts"].fillna(0), name="First-time debuts", marker_color="#7a5195")
        fig.update_layout(title="Average entries by weekday", barmode="group", height=340, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    ev = la.release_events(sd, lc)
    if len(ev):
        ev = ev[(ev["date"] >= d0) & (ev["date"] <= d1)]
    st.markdown("##### Album-drop events (≥3 tracks of one artist debuting on the same day)")
    if len(ev):
        show = ev.assign(date=ev["date"].dt.date,
                         share_gone_in_7d=ev["share_gone_in_7d"].map(lambda x: "n/a (<8 d observed)" if pd.isna(x) else f"{x:.0%}"),
                         album_share=ev["album_share"].map(lambda x: f"{x:.0%}"))
        st.caption("*Days are censored for tracks still listed on the last snapshot.")
        st.dataframe(show.drop(columns=["observed_days"]).rename(columns={
            "tracks_entered": "tracks entered", "best_peak": "best rank", "median_days": "median days*",
            "max_days": "longest-lived track*", "still_listed_now": "still listed on last day",
            "share_gone_in_7d": "gone within 7 d", "album_share": "from albums"}),
            width="stretch", hide_index=True)
    else:
        st.info("No album-drop events in the selected window.")

# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Lifecycle stage distribution")
    a, b = st.columns([1, 1.6])
    share = sd_stage["stage"].value_counts(normalize=True).reindex(la.STAGES).fillna(0)
    with a:
        fig = go.Figure(go.Bar(y=share.index[::-1], x=share.values[::-1], orientation="h",
                               marker_color=[STAGE_COL[s] for s in share.index[::-1]],
                               text=[f"{v:.1%}" for v in share.values[::-1]], textposition="outside"))
        fig.update_layout(title="Share of slot-days", height=360, xaxis_tickformat=".0%",
                          margin=dict(t=50, r=40), xaxis_range=[0, (share.max() * 1.25) if share.max() else 1])
        st.plotly_chart(fig, width="stretch")
    with b:
        freq = st.radio("Aggregate by", ["Week", "Month"], horizontal=True)
        t = sd_stage.assign(period=sd_stage["date"].dt.to_period("W" if freq == "Week" else "M").dt.start_time)
        t = t.groupby(["period", "stage"], observed=True).size().reset_index(name="n")
        fig = px.area(t, x="period", y="n", color="stage", groupnorm="percent",
                      color_discrete_map=STAGE_COL, category_orders={"stage": la.STAGES})
        fig.update_layout(title="Stage mix over time", height=360, yaxis_title="% of slot-days",
                          margin=dict(t=50), legend_title="")
        st.plotly_chart(fig, width="stretch")
    prof = sd_stage.groupby("stage", observed=True).agg(
        slot_days=("key", "size"), mean_rank=("position", "mean"),
        mean_popularity=("popularity", "mean"), mean_rank_slope=("rank_slope", "mean")).reindex(la.STAGES)
    prof["share"] = prof["slot_days"] / prof["slot_days"].sum() if prof["slot_days"].sum() else np.nan
    st.dataframe(prof.round(2).rename(columns={"slot_days": "slot-days", "mean_rank": "mean rank",
                                               "mean_popularity": "mean popularity",
                                               "mean_rank_slope": "rank slope (pos/day)"}),
                 width="stretch")
    fin = lc_f["final_stage"].value_counts(normalize=True).reindex(la.STAGES).fillna(0)
    st.caption("Stage rules: New Entry = first 7 listed days (entry observed) · Peak = 7-day median rank in Top 10 · "
               "Growth/Decline = trailing 7-day rank slope ≤ −0.75 / ≥ +0.75 positions per day · otherwise Mature. "
               f"Stage at last appearance: {', '.join(f'{k} {v:.0%}' for k, v in fin.items())}.")

# --------------------------------------------------------------------------
# Content maturity
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Content maturity comparisons")
    inc = lc_f[lc_f["incident"]].copy()
    inc["event"] = (~inc["right_censored"]).astype(int)
    a, b = st.columns(2)
    for col_, grp, title in [(a, "explicit_label", "Explicit vs clean"), (b, "album_type", "Single vs album")]:
        with col_:
            fig = go.Figure()
            for val in inc[grp].dropna().unique():
                g = inc[inc[grp] == val]
                if len(g) >= 5:
                    k = la.km_curve(g["days_on_playlist"], g["event"])
                    fig.add_trace(go.Scatter(x=k["time"], y=k["survival"], mode="lines",
                                             line=dict(shape="hv", width=3, color=PALETTE.get(val)),
                                             name=f"{val} (n={len(g)}, median {la.km_median(k):.0f} d)"))
            fig.update_layout(title=f"Survival – {title}", height=340, xaxis_range=[0, 200],
                              yaxis_tickformat=".0%", margin=dict(t=50),
                              legend=dict(orientation="h", y=-0.2), xaxis_title="Days on Top 50")
            st.plotly_chart(fig, width="stretch")
    inc["entry_type"] = np.where(inc["burst_entry"] == True, "Album-drop entry", "Other entry")  # noqa: E712
    a, b = st.columns(2)
    with a:
        if len(inc):
            fig = px.box(inc, x="explicit_label", y="days_on_playlist", color="album_type",
                         color_discrete_map=PALETTE, log_y=True, points=False)
            fig.update_layout(title="Days on playlist by content type (log scale)", height=340,
                              xaxis_title="", yaxis_title="Days", margin=dict(t=50), legend_title="")
            st.plotly_chart(fig, width="stretch")
    with b:
        if len(inc):
            fig = px.box(inc, x="entry_type", y="days_on_playlist", color="entry_type", log_y=True, points=False,
                         color_discrete_map={"Album-drop entry": "#c0392b", "Other entry": "#2e8b57"})
            fig.update_layout(title="Album-drop entries vs other entries", height=340, xaxis_title="",
                              yaxis_title="Days", showlegend=False, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        d = lc_f[lc_f["incident"]].groupby("duration_bucket", observed=True).agg(
            median=("days_on_playlist", "median"), n=("key", "size")).reset_index()
        if len(d):
            fig = px.bar(d, x=d["duration_bucket"].astype(str), y="median", text="median",
                         hover_data=["n"], color_discrete_sequence=["#1f5fa8"])
            fig.update_layout(title="Song length (quartiles) vs median days", height=320,
                              xaxis_title="", yaxis_title="Median days", margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
    with b:
        d = lc_f[lc_f["incident"]].groupby("album_size_bucket", observed=True).agg(
            median=("days_on_playlist", "median"), n=("key", "size"),
            burst=("burst_entry", "mean")).reset_index()
        if len(d):
            fig = px.bar(d, x=d["album_size_bucket"].astype(str), y="median", text="median",
                         hover_data=["n", "burst"], color_discrete_sequence=["#d9822b"])
            fig.update_layout(title="Release size (tracks) vs median days", height=320,
                              xaxis_title="", yaxis_title="Median days", margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")

    st.markdown("##### Statistical comparison (selected songs; ratio of mean days, completed lifecycles)")
    rows = []
    lc_cmp = lc_f.assign(burst_label=np.where(lc_f["burst_entry"] == True, "burst", "non-burst"))  # noqa: E712
    nb = lc_cmp[lc_cmp["burst_label"] == "non-burst"]
    specs = [(lc_cmp, "explicit_label", "Explicit", "Clean", "Explicit vs clean – all"),
             (nb, "explicit_label", "Explicit", "Clean", "Explicit vs clean – excl. album-drop entries"),
             (lc_cmp, "album_type", "single", "album", "Single vs album – all"),
             (nb, "album_type", "single", "album", "Single vs album – excl. album-drop entries"),
             (lc_cmp, "burst_label", "burst", "non-burst", "Album-drop vs other entry")]
    for df_, col_, a_, b_, lab in specs:
        i_ = df_[df_["incident"]]
        if (i_[col_] == a_).sum() >= 8 and (i_[col_] == b_).sum() >= 8 and df_[df_["completed"]][col_].isin([a_, b_]).any():
            try:
                r = la.compare_groups(df_, col_, a_, b_, lab, n_boot=800, n_boot_rmst=0)
                km_a = f"{r['km_median_a']:.0f}" if pd.notna(r['km_median_a']) else "n/a"
                km_b = f"{r['km_median_b']:.0f}" if pd.notna(r['km_median_b']) else "n/a"
                ratio_s = f"{r['ratio']:.2f}" if pd.notna(r['ratio']) else "n/a"
                ci_s = f"{r['ci_lo']:.2f}–{r['ci_hi']:.2f}" if pd.notna(r['ci_lo']) and pd.notna(r['ci_hi']) else "n/a"
                p_s = ("<0.001" if (pd.notna(r["p_logrank"]) and r["p_logrank"] < .001)
                       else (f"{r['p_logrank']:.3f}" if pd.notna(r["p_logrank"]) else "n/a"))
                rows.append({"Comparison": lab, "n (A/B)": f"{r['n_a']}/{r['n_b']}",
                             "KM median A / B (d)": f"{km_a} / {km_b}",
                             "Ratio A÷B": ratio_s, "95% CI": ci_s, "Log-rank p": p_s})
            except Exception:
                pass
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("Not enough songs in both groups under the current filters.")
    st.caption("A÷B: first named group over second. 'Album-drop entry' = ≥3 tracks of the same artist debuting on the same day. "
               "The explicit-vs-clean and single-vs-album gaps depend heavily on this confounder – compare the 'all' and 'excl.' rows.")
    st.download_button("Download song lifecycle table (CSV)",
                       lc_f.drop(columns=["key"]).to_csv(index=False).encode(),
                       "song_lifecycles_filtered.csv")

# --------------------------------------------------------------------------
# Churn
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Playlist churn analytics")
    v = flows[flows["churn_valid"]]
    m = la.monthly_rotation(flows) if len(v) else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean daily churn", fmt((kp["churn_rate"] * 100) if pd.notna(kp["churn_rate"]) else np.nan, "{:.2f}%"))
    c2.metric("Median entries / day", fmt(v["entries"].median() if len(v) else np.nan, "{:.0f}"))
    c3.metric("Days with zero entries", fmt(((v["entries"] == 0).mean() * 100) if len(v) else np.nan, "{:.0f}%"))
    c4.metric("Mean |rank move| / day", fmt(kp["rank_volatility"], "{:.2f}"))
    if len(m):
        fig = go.Figure()
        fig.add_bar(x=m["month"], y=m["avg_churn_rate"] * 100, name="Churn % (left)", marker_color="#1f5fa8")
        fig.add_trace(go.Scatter(x=m["month"], y=m["avg_rank_volatility"], name="Rank volatility (right)",
                                 yaxis="y2", line=dict(color="#c0392b", width=3)))
        fig.update_layout(title="Monthly rotation", height=360, yaxis_title="Mean daily churn (%)",
                          yaxis2=dict(overlaying="y", side="right", title="Mean |rank move|", showgrid=False),
                          legend=dict(orientation="h", y=-0.15), margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    with a:
        if len(v):
            fig = px.histogram(v, x="entries", nbins=25, color_discrete_sequence=["#1f5fa8"])
            fig.update_layout(title="Distribution of daily entries (heavy right tail = album drops)",
                              height=330, margin=dict(t=50), xaxis_title="Entries per day", yaxis_title="Days")
            st.plotly_chart(fig, width="stretch")
    with b:
        if len(v):
            fig = px.scatter(v, x="rank_volatility", y="churn_rate", hover_data=["date"],
                             color_discrete_sequence=["#7a5195"], opacity=.7)
            fig.update_layout(title="Stability vs volatility (each dot = a day)", height=330,
                              margin=dict(t=50), xaxis_title="Mean |daily rank move|",
                              yaxis_title="Churn rate", yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
    if len(m):
        mm = m.assign(debuts=(m["debut_share"] * 100).round(0),
                      churn=(m["avg_churn_rate"] * 100).round(2),
                      volatility=m["avg_rank_volatility"].round(2),
                      entries=m["avg_entries"].round(2))[["month", "days", "entries", "churn", "volatility", "debuts"]]
        st.dataframe(mm.rename(columns={"entries": "avg entries/day", "churn": "churn %",
                                        "volatility": "rank volatility", "debuts": "% debuts (vs returns)"}),
                     hide_index=True, width="stretch")
    st.caption("Transitions that span a missing calendar day are excluded. Returns = songs re-entering after an earlier exit (often hovering at ranks 45–50).")

# --------------------------------------------------------------------------
# Data quality & method
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Data quality & method")
    q = pd.DataFrame([
        ("Raw rows", f"{qa['raw_rows']:,}"), ("Rows after cleaning", f"{qa['rows_after_cleaning']:,}"),
        ("Snapshot days (50 entries each)", f"{qa['days_valid_after_cleaning']} of {qa['days_total']}"),
        ("Days with >50 rows (duplicate scrape)", ", ".join(f"{k} ({v})" for k, v in qa["days_over_50"].items()) or "none"),
        ("Missing calendar days", ", ".join(qa["missing_calendar_days"]) or "none"),
        ("Exact duplicate rows", qa["exact_duplicate_rows"]),
        ("Popularity = 0 rows (treated as missing)", f"{qa['popularity_zero_rows']} ({qa['popularity_zero_share_in_first_2_days']:.0%} on a track's first 2 days)"),
        ("Distinct song/artist spellings → tracks", f"{qa['raw_song_artist_pairs']} → {qa['unique_tracks_after_normalisation']}"),
        ("Tracks with changing album_type / total_tracks", f"{qa['tracks_with_changing_attributes']['album_type']} / {qa['tracks_with_changing_attributes']['total_tracks']}"),
    ], columns=["Check", "Result"]).astype(str)
    st.dataframe(q, hide_index=True, width="stretch")
    st.markdown("""
**Definitions**
- **Track identity**: normalised title + normalised primary artist (case, accents, spacing).
- **Days on playlist**: number of snapshot days listed (re-entries included). **Completed lifecycle**: entry and exit both inside the data window.
- **Censoring**: 50 songs were already listed on day 1 and 50 on the last day; duration KPIs use completed lifecycles only; survival curves treat still-listed songs as censored.
- **Retention Stability Index**: mean over songs of 1 − mean|daily rank change|/49.
- **Explicit Content Lifecycle Score / Single-vs-Album Ratio**: ratios of mean days on playlist (completed lifecycles).
- **Album-drop burst**: ≥3 tracks of the same primary artist debut on one day.

**Limitations**: no release dates, genres or stream counts; a single market, so “faster rotation than UK/US” cannot be tested; the popularity score is a lagging API metric; stage thresholds are analyst-defined (sensitivity in the paper).
""")