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
def _classify_poss_bucket(ptype: str, shot_num: int, time_since_prev_end: float) -> str:
    """
    time_since_prev_end: game-clock seconds elapsed from previous possession's
    end to this shot.  NaN when unknown (start of period, etc.).

    The pipeline labels ~50% of possessions as "transition" because it sets
    start_seconds = shot_seconds for any quick shot, making time_in_poss = 0.
    We filter this: a possession is only treated as "transition" if the shot
    was actually taken within 8 s of the previous possession ending.
    """
    if ptype in ("second_chance", "scramble_putback"):
        return "putback"
    if ptype == "transition":
        if pd.notna(time_since_prev_end) and time_since_prev_end <= 8:
            return "transition"
        # Pipeline mislabeled as transition; treat as first half-court attempt
        return "first_hc"
    # half_court (and any other type)
    if shot_num == 1:
        return "first_hc"
    if shot_num == 2:
        return "second_hc"
    return "third_plus_hc"


# ── Shot-clock bucket ─────────────────────────────────────────────────────────
def _classify_clock_bucket(ptype: str, time_since_prev_end: float) -> str:
    """
    Uses time_since_prev_end (game clock elapsed from previous possession's end
    to the shot) as a proxy for shot-clock time used.  This is the only reliable
    timing measure because for transition possessions the pipeline sets
    start_seconds = shot_seconds (so time_in_poss is always 0).
    """
    if ptype in ("second_chance", "scramble_putback"):
        return "extra"
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

    # ── shots: drop FTs, add zone ─────────────────────────────────────────────
    shots = shots[shots["shot_range"] != "free_throw"].copy()
    shots["shot_zone"] = shots.apply(_classify_zone, axis=1)

    # ── link shots → possessions ──────────────────────────────────────────────
    shots_enriched = _link_shots_to_possessions(shots, possessions)

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
        "start_seconds", "end_seconds",
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

    # time_since_prev_end: positive = seconds elapsed since ball changed hands
    merged["time_since_prev_end"] = (
        merged["prev_end_seconds"] - merged["secondsRemaining"]
    ).clip(lower=0)

    # ── derived labels (vectorised) ───────────────────────────────────────────
    pt   = merged["possession_type"].fillna("nan").astype(str)
    sn   = merged["shot_num_in_poss"].fillna(1).astype(int)
    tspe = merged["time_since_prev_end"]

    merged["poss_bucket"] = [
        _classify_poss_bucket(p, s, t) for p, s, t in zip(pt, sn, tspe)
    ]
    merged["clock_bucket"] = [
        _classify_clock_bucket(p, t) for p, t in zip(pt, tspe)
    ]

    merged.drop(columns=["_gkey", "_pgkey", "prev_end_seconds"], inplace=True, errors="ignore")
    return merged
