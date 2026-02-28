"""
data_loader.py
--------------
Loads, links, and feature-engineers all season data for the shots dashboard.
Cached with st.cache_data so it only runs once per Streamlit session.
"""

import glob
import os
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "cbbd_data")


def _load_dir(subdir: str, **kwargs) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, "*.csv")))
    if not paths:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_csv(p, low_memory=False, **kwargs) for p in paths],
        ignore_index=True,
    )


# ── Shot zone classification ──────────────────────────────────────────────────
# Court coords: x=0-940 (full-court length), y=0-500 (court width)
# Basket 1 ≈ (52, 250),  Basket 2 ≈ (888, 250)
# Corner-3 region: within ~15ft of a baseline AND ~10ft of a sideline
_CORNER_X_LO = 150
_CORNER_X_HI = 790
_CORNER_Y_LO = 100
_CORNER_Y_HI = 400
_PAINT_DIST = 10   # feet; non-rim jumpers inside this distance = paint


def _classify_zone(row) -> str:
    sr = row["shot_range"]
    if sr == "free_throw":
        return "free_throw"
    if sr == "rim":
        return "rim"
    if row["is_three"]:
        x, y = row["x"], row["y"]
        if pd.notna(x) and pd.notna(y):
            near_baseline = (x < _CORNER_X_LO) or (x > _CORNER_X_HI)
            near_sideline = (y < _CORNER_Y_LO) or (y > _CORNER_Y_HI)
            if near_baseline and near_sideline:
                return "corner_3"
        return "non_corner_3"
    # jumper
    dist = row["distance"]
    if pd.notna(dist) and dist <= _PAINT_DIST:
        return "paint"
    return "non_paint_2"


# ── Possession-bucket classification ──────────────────────────────────────────
def _classify_poss_bucket(
    ptype: str,
    shot_num: int,
    time_since_prev_end: float,
    time_oreb_to_fga: float,
    is_last_in_poss: bool,
) -> str:
    """
    shot_num:         cumulative FGA on this offensive trip (cum_fga_in_trip).
    is_last_in_poss:  True for the terminal shot in this possession_id (the
                      made bucket or the final miss after all scramble attempts).
    time_oreb_to_fga: pipeline-computed seconds from OREB to the first FGA in
                      this possession.  Null (~30% of second_chance shots) means
                      the pipeline could not determine OREB timing — those fall
                      through to HC bucketing rather than being mislabeled putbacks.

    A shot is a true putback only when:
      - the possession is second_chance / scramble_putback,
      - torf <= 5 s (quick scramble off the OREB), AND
      - it is the LAST shot in that possession (is_last_in_poss == True).
    Only the terminal shot counts so that each OREB opportunity is one scoring
    attempt.  Intermediate misses in the same scramble window get
    "scramble_intermediate" and are excluded from all HC FG% buckets (they are
    structurally guaranteed misses — the possession continued after them).
    Note: torf is a possession-level field reflecting the first FGA's timing.
    Multi-shot scrambles where later shots slip past 5 s are rare enough in
    practice that start_seconds is not a reliable per-shot alternative (it has
    zero nulls and a wide distribution suggesting unreliable merge matches).
    """
    if ptype in ("second_chance", "scramble_putback"):
        if pd.notna(time_oreb_to_fga) and time_oreb_to_fga <= 5:
            if is_last_in_poss:
                return "putback"
            # Intermediate miss in a multi-shot scramble: structurally cannot be
            # a make (possession continued), so excluding from all HC FG% buckets
            # prevents these guaranteed-zero shots from diluting HC efficiency.
            return "scramble_intermediate"
        # torf > 5 s or null: offense has reset — fall through to trip-based HC bucketing
    elif ptype == "transition":
        if pd.notna(time_since_prev_end) and time_since_prev_end <= 8:
            return "transition"
        # Pipeline mislabeled as transition; fall through to HC bucketing
    # HC set: bucket by cumulative FGA on this offensive trip
    if shot_num <= 1:
        return "first_hc"
    if shot_num == 2:
        return "second_hc"
    return "third_plus_hc"


# ── Shot-clock bucket ─────────────────────────────────────────────────────────
def _classify_clock_bucket(ptype: str, time_since_prev_end: float, time_oreb_to_fga: float) -> str:
    """
    Uses time_since_prev_end as a shot-clock proxy for transition/half-court shots.
    For second_chance/scramble_putback uses time_oreb_to_fga (pipeline-computed
    seconds from OREB to first FGA) which is the correct OREB-timing measure.

    second_chance/scramble_putback shots within 5 s of the OREB → "extra"
    (putback pace).  Beyond 5 s (or null) the offense has reset, so fall
    through to normal shot-clock bucketing.
    """
    if ptype in ("second_chance", "scramble_putback"):
        if pd.notna(time_oreb_to_fga) and time_oreb_to_fga <= 5:
            return "extra"
        # fell out of scramble window — fall through to normal timing below
    if pd.notna(time_since_prev_end):
        if time_since_prev_end <= 8:
            return "transition_pace"
        if time_since_prev_end <= 25:
            return "normal"
        if time_since_prev_end <= 30:
            return "late_clock"
    return "extra"


# ── Main data-loading entry point ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading season data…")
def load_all_data():
    """
    Returns
    -------
    shots_enriched : DataFrame  – every FGA with possession & zone context
    possessions    : DataFrame  – raw possessions_enriched
    four_factors   : DataFrame  – per-team, per-game four-factors
    games          : DataFrame  – game metadata
    """
    # ── raw tables ────────────────────────────────────────────────────────────
    shots = _load_dir("shots")
    possessions = _load_dir("possessions_enriched")
    four_factors = _load_dir("four_factors")
    games = _load_dir("games")

    # de-duplicate games (same game appears in both home & away daily files)
    if not games.empty:
        games = games.drop_duplicates(subset="id")

    # ── correct possession_type on the raw possessions table ──────────────────
    # The pipeline labels ~50% of possessions as "transition" because it sets
    # start_seconds = shot_seconds for quick plays (time_to_first_fga = 0).
    # Use time_since_prev_end to filter out false positives: if a "transition"
    # possession had the shot > 8 s after the previous possession ended, it is
    # really a half-court possession.
    if not possessions.empty:
        poss_adj = possessions[possessions["possession_team"].notna()].copy()
        poss_adj = poss_adj.sort_values(["gameId", "possession_id"])
        poss_adj["_prev_end"] = poss_adj.groupby("gameId")["end_seconds"].shift(1)
        poss_adj["_tspe"] = (poss_adj["_prev_end"] - poss_adj["start_seconds"]).clip(lower=0)
        mask_false_trans = (
            (poss_adj["possession_type"] == "transition")
            & (poss_adj["_tspe"] > 8)
        )
        poss_adj.loc[mask_false_trans, "possession_type"] = "half_court"
        # Merge corrected type back onto original possessions index
        possessions = possessions.copy()
        possessions.loc[poss_adj.index, "possession_type"] = poss_adj["possession_type"]

    # ── shots: separate FTs (kept for PPP), process FGAs ─────────────────────
    shots_ft = shots[shots["shot_range"] == "free_throw"].copy()
    shots    = shots[shots["shot_range"] != "free_throw"].copy()
    shots["shot_zone"] = shots.apply(_classify_zone, axis=1)

    # is_assisted: True when the shot was credited to an assisting player.
    # Use assisted_by.notna() rather than the `assisted` boolean — the bool has
    # only ~21% coverage while assisted_by fires on ~52% of makes (matching the
    # expected D1 median) and is always null on misses (clean semantics).
    shots["is_assisted"] = shots["assisted_by"].notna()

    # ── shooter 3pt grade (Red / Yellow / Green closeout priority) ────────────
    # Grade each shooter by their season 3pt FG% — used in defensive analysis
    # to assess *who* a team is allowing to shoot 3s, not just how many.
    #
    # Qualification: 3+ 3pt attempts per game AND 5+ games played.
    # Below that volume the sample is too small to trust; treat as Green.
    #
    # Thresholds (among qualified shooters, ~top-30th percentile = Red):
    #   Red    ≥ 37%  — must close hard, genuine 3pt threat
    #   Yellow  32–37% — close out but stay under control
    #   Green  < 32%  — can sag; also all low-volume / unqualified shooters
    _games_played = (
        shots.groupby("shooter_id")["gameId"]
        .nunique()
        .rename("_gp")
    )
    _three_stats = (
        shots[shots["is_three"] == True]
        .groupby("shooter_id")
        .agg(_att3=("made", "count"), _made3=("made", "sum"))
    )
    _shooter_grades = (
        _three_stats
        .join(_games_played, how="left")
        .assign(
            _att3_pg=lambda d: d["_att3"] / d["_gp"].clip(lower=1),
            _fg3=lambda d: 100.0 * d["_made3"] / d["_att3"].clip(lower=1),
        )
    )
    def _grade(row):
        if row["_att3_pg"] >= 3 and row["_gp"] >= 5:
            if row["_fg3"] >= 37:
                return "red"
            if row["_fg3"] >= 32:
                return "yellow"
        return "green"

    _shooter_grades["shooter_3pt_grade"] = _shooter_grades.apply(_grade, axis=1)
    shots = shots.merge(
        _shooter_grades[["shooter_3pt_grade"]],
        left_on="shooter_id",
        right_index=True,
        how="left",
    )
    shots["shooter_3pt_grade"] = shots["shooter_3pt_grade"].fillna("green")

    # ── link shots → possessions ──────────────────────────────────────────────
    shots_enriched = _link_shots_to_possessions(shots, possessions)

    # ── attach FT rows with possession context (for PPP numerator) ───────────
    # FTs are kept separate from FGAs so trip/bucket logic above is unaffected.
    # Only possession_type is needed from the join; other derived columns
    # (poss_bucket, clock_bucket, cum_fga_in_trip) exist but are always ignored
    # because every metric filters by shot_zone or poss_bucket, and "free_throw"
    # is not in any of those allow-lists.
    if not shots_ft.empty and not possessions.empty:
        shots_ft["shot_zone"]         = "free_throw"
        shots_ft["is_assisted"]       = False
        shots_ft["shooter_3pt_grade"] = "green"
        shots_ft_enriched = _link_shots_to_possessions(shots_ft, possessions)
        shots_enriched = pd.concat(
            [shots_enriched, shots_ft_enriched], ignore_index=True
        )

    return shots_enriched, possessions, four_factors, games


# ── Shot ↔ Possession join ────────────────────────────────────────────────────
def _link_shots_to_possessions(
    shots: pd.DataFrame, poss: pd.DataFrame
) -> pd.DataFrame:
    """
    For every FGA, finds its possession using a fully-vectorized range join.

    Strategy
    --------
    Time counts DOWN (1200 → 0 per half).  We build a monotone global sort
    key:  game_rank * PERIOD_SLOTS + (period-1) * CLOCK_SLOTS + elapsed_sec
    where elapsed_sec = 1200 - secondsRemaining  (0 = start, 1200 = end).

    This makes the key globally ascending, so a single merge_asof
    direction='backward' correctly finds the most recent possession that
    started at or before each shot.  We then validate the shot falls
    inside [end_seconds, start_seconds].
    """
    CLOCK_SLOTS  = 1500   # > max seconds per period (1200 reg + OT buffer)
    PERIOD_SLOTS = CLOCK_SLOTS * 12  # room for up to 12 periods

    poss_clean = poss[poss["possession_team"].notna()].copy()

    # build unified game rank from both tables
    all_games = sorted(
        set(shots["gameId"].unique()) | set(poss_clean["gameId"].unique())
    )
    game_rank = {g: i for i, g in enumerate(all_games)}

    def _gkey(df, seconds_col):
        rank  = df["gameId"].map(game_rank).fillna(0).astype(np.int64)
        per   = df["period"].astype(float).fillna(1).astype(np.int64).clip(1)
        sec   = df[seconds_col].astype(float).fillna(0)
        elapsed = (1200.0 - sec).clip(0, CLOCK_SLOTS - 1).astype(np.int64)
        return rank * PERIOD_SLOTS + (per - 1) * CLOCK_SLOTS + elapsed

    shots = shots.copy()
    shots["_gkey"] = _gkey(shots, "secondsRemaining")

    poss_clean["_pgkey"] = _gkey(poss_clean, "start_seconds")

    shots_s = shots.sort_values("_gkey")
    poss_s  = poss_clean.sort_values("_pgkey")

    poss_cols = [
        "_pgkey",
        "possession_id", "possession_team", "possession_type",
        "start_seconds", "end_seconds", "time_oreb_to_fga",
    ]

    merged = pd.merge_asof(
        shots_s,
        poss_s[poss_cols],
        left_on="_gkey",
        right_on="_pgkey",
        direction="backward",
    )

    # validate: shot must fall inside [end_seconds, start_seconds]
    merged = merged[
        merged["secondsRemaining"] >= merged["end_seconds"].fillna(-1)
    ].copy()

    # ── shot number within possession ─────────────────────────────────────────
    merged = merged.sort_values(
        ["gameId", "possession_id", "secondsRemaining"],
        ascending=[True, True, False],
    )
    merged["shot_num_in_poss"] = (
        merged.groupby(["gameId", "possession_id"]).cumcount() + 1
    )
    # Last shot in each possession: the terminal outcome (made or final miss).
    # Used for putback classification so each OREB opportunity counts once.
    poss_size = merged.groupby(["gameId", "possession_id"])["shot_num_in_poss"].transform("max")
    merged["is_last_in_poss"] = (merged["shot_num_in_poss"] == poss_size)

    # ── time since previous possession ended ──────────────────────────────────
    # For "transition" possessions the pipeline sets start_seconds = shot_seconds
    # (i.e. time_in_poss is always 0), making intra-possession timing useless.
    # Instead we compute the game-clock gap from the *previous* possession's
    # end_seconds to this shot, which is a reliable proxy for shot-clock usage.
    poss_prev = (
        poss_clean[["gameId", "possession_id", "end_seconds"]]
        .drop_duplicates(["gameId", "possession_id"])
        .sort_values(["gameId", "possession_id"])
        .copy()
    )
    poss_prev["prev_end_seconds"] = poss_prev.groupby("gameId")["end_seconds"].shift(1)
    poss_prev = poss_prev[["gameId", "possession_id", "prev_end_seconds"]]

    merged = merged.merge(poss_prev, on=["gameId", "possession_id"], how="left")

    # ── trip-level cumulative FGA ──────────────────────────────────────────────
    # Possession-level chaining is unreliable because the pipeline often inserts
    # an opponent possession record between the missed shot and the OREB, making
    # the "previous possession" by ID (or clock order) the opponent's.
    # Instead we track trips at the shot level:
    #   - A shot whose possession_type is second_chance / scramble_putback
    #     continues the same offensive trip as the prior FGA by that team.
    #   - Any other possession_type starts a new trip.
    # Sorting shots chronologically (descending secondsRemaining = forward in time)
    # and cumcounting within (gameId, team, trip_id) gives cum_fga_in_trip.
    merged = merged.sort_values(
        ["gameId", "period", "team", "secondsRemaining"],
        ascending=[True, True, True, False],
    )
    # Gap (seconds) from the previous same-team shot within the same period.
    # Used to catch "false" second_chance labels — cases where the pipeline tags
    # a possession as second_chance but the team's prior shot was a make (so the
    # opponent had the ball in between).  The CBB shot clock resets to 20s after
    # an OREB, so any genuine putback/reset shot must arrive within ~30s of the
    # prior team shot.  A larger gap means at least one full opponent possession
    # occurred and this should be treated as a new trip.
    grp = merged.groupby(["gameId", "period", "team"])
    merged["_prev_sr"]   = grp["secondsRemaining"].shift(1)
    merged["_prev_made"] = grp["made"].shift(1)
    merged["_shot_gap"]  = (merged["_prev_sr"] - merged["secondsRemaining"]).clip(lower=0)

    # A possession continues the same trip only when ALL three hold:
    #   1. Pipeline labels it as a second-chance / scramble continuation
    #   2. The prior same-team shot was a MISS  (can't get OREB after a make)
    #   3. The gap from the prior shot is ≤ 30s (rules out a full opponent possession)
    is_continuation = (
        merged["possession_type"].isin(["second_chance", "scramble_putback"])
        & (merged["_prev_made"] == False)
        & (merged["_shot_gap"] <= 30)
    )
    merged["_new_trip"] = (~is_continuation).astype(int)
    merged["_trip_id"] = merged.groupby(["gameId", "period", "team"])["_new_trip"].cumsum()
    merged["cum_fga_in_trip"] = (
        merged.groupby(["gameId", "period", "team", "_trip_id"]).cumcount() + 1
    )
    merged.drop(columns=["_new_trip", "_trip_id", "_prev_sr", "_prev_made", "_shot_gap"], inplace=True)

    # time_since_prev_end: positive = seconds elapsed since ball changed hands
    merged["time_since_prev_end"] = (
        merged["prev_end_seconds"] - merged["secondsRemaining"]
    ).clip(lower=0)

    # ── derived labels (vectorised) ───────────────────────────────────────────
    pt   = merged["possession_type"].fillna("nan").astype(str)
    sn   = merged["cum_fga_in_trip"].fillna(1).astype(int)
    tspe = merged["time_since_prev_end"]
    torf = merged["time_oreb_to_fga"]
    ilip = merged["is_last_in_poss"].fillna(False)

    merged["poss_bucket"] = [
        _classify_poss_bucket(p, s, t, o, n) for p, s, t, o, n in zip(pt, sn, tspe, torf, ilip)
    ]
    merged["clock_bucket"] = [
        _classify_clock_bucket(p, t, o) for p, t, o in zip(pt, tspe, torf)
    ]

    merged.drop(columns=["_gkey", "_pgkey", "prev_end_seconds"], inplace=True, errors="ignore")
    return merged
