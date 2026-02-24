"""
metrics.py
----------
All aggregation and metric-computation helpers for the shots dashboard.
All public functions take the full enriched shots DataFrame + a team name
+ side ('Offense' | 'Defense') and return tidy DataFrames or dicts.
"""

import numpy as np
import pandas as pd

# ── Ordered labels ─────────────────────────────────────────────────────────────
ZONE_ORDER = ["rim", "paint", "non_paint_2", "corner_3", "non_corner_3"]
ZONE_LABELS = {
    "rim":          "Rim",
    "paint":        "Paint (non-rim)",
    "non_paint_2":  "Mid-Range",
    "corner_3":     "Corner 3",
    "non_corner_3": "Above-Break 3",
}

BUCKET_ORDER = ["transition", "first_hc", "putback", "second_hc", "third_plus_hc"]
BUCKET_LABELS = {
    "transition":      "Transition",
    "first_hc":        "1st HC Attempt",
    "putback":         "Putbacks / Scrambles",
    "second_hc":       "2nd HC Attempt",
    "third_plus_hc":   "3rd+ HC Attempt",
}

CLOCK_ORDER = ["transition_pace", "normal", "late_clock", "extra"]
CLOCK_LABELS = {
    "transition_pace": "≤8s (Transition pace)",
    "normal":          "9–25s (Normal)",
    "late_clock":      "25–30s (Late clock)",
    "extra":           "Extra (Putback/2nd chance)",
}


# ── Internal helpers ───────────────────────────────────────────────────────────
def _efg(df: pd.DataFrame) -> float:
    fga = len(df)
    if fga == 0:
        return np.nan
    fgm = df["made"].sum()
    threes_made = df.loc[df["is_three"] == True, "made"].sum()
    return 100.0 * (fgm + 0.5 * threes_made) / fga


def _filter(shots: pd.DataFrame, team: str, side: str) -> pd.DataFrame:
    if side == "Offense":
        return shots[shots["team"] == team].copy()
    return shots[shots["opponent"] == team].copy()


# ── Four Factors ───────────────────────────────────────────────────────────────
def compute_four_factors(
    ff: pd.DataFrame, shots: pd.DataFrame, team: str, side: str
) -> dict:
    """
    Aggregate season-level four-factors for *team*.

    Note
    ----
    FGM and 3PM in the four_factors table are zeroed out for ~51 % of rows
    (pipeline data issue).  eFG % is therefore computed from the shots table
    directly, which is always accurate.  FT Rate uses FTA/FGA (both populated).
    TO %, ORB %, and Tempo are read from the four_factors table (accurate).
    """
    # ── eFG% from shots ───────────────────────────────────────────────────────
    if side == "Offense":
        sh = shots[shots["team"] == team]
    else:
        sh = shots[shots["opponent"] == team]
    sh = sh[sh["shot_zone"] != "free_throw"]  # exclude FTs (already done in loader)

    fga_shots = len(sh)
    fgm_shots = sh["made"].sum()
    made_3     = sh.loc[sh["is_three"] == True, "made"].sum()
    efg = (
        round(100.0 * (fgm_shots + 0.5 * made_3) / fga_shots, 1)
        if fga_shots > 0
        else np.nan
    )

    # ── other factors from four_factors table ─────────────────────────────────
    if ff.empty:
        return {"efg": efg, "to_pct": np.nan, "orb_pct": np.nan,
                "ft_rate": np.nan, "tempo": np.nan, "games": 0}

    if side == "Offense":
        tf = ff[ff["team"] == team]
    else:
        tf = ff[ff["opponent"] == team]

    if tf.empty:
        return {"efg": efg, "to_pct": np.nan, "orb_pct": np.nan,
                "ft_rate": np.nan, "tempo": np.nan, "games": 0}

    fga_ff  = tf["FGA"].sum()
    tov     = tf["TOV"].sum()
    poss    = tf["Possessions"].sum()
    orb     = tf["ORB"].sum()
    opp_drb = tf["Opp_DRB"].sum()
    fta     = tf["FTA"].sum()   # FTA is populated even in bad rows

    return {
        "efg":     efg,
        "to_pct":  round(100.0 * tov / poss, 1) if poss > 0 else np.nan,
        "orb_pct": round(100.0 * orb / (orb + opp_drb), 1) if (orb + opp_drb) > 0 else np.nan,
        "ft_rate": round(100.0 * fta / fga_ff, 1) if fga_ff > 0 else np.nan,
        "tempo":   round(tf["Tempo"].mean(), 1),
        "games":   tf["game_id"].nunique() if "game_id" in tf.columns else len(tf),
    }


def compute_team_record(games: pd.DataFrame, team: str) -> str:
    """Return 'W-L' string for the team."""
    if games.empty:
        return ""
    home = games[games["homeTeam"] == team]
    away = games[games["awayTeam"] == team]
    # use == True to safely handle object dtype with mixed NaN values
    w = int((home["homeWinner"] == True).sum()) + int((away["awayWinner"] == True).sum())
    total = len(home) + len(away)
    return f"{w}-{total - w}"


def compute_team_conference(games: pd.DataFrame, team: str) -> str:
    if games.empty:
        return ""
    home = games[games["homeTeam"] == team]
    if not home.empty:
        return str(home["homeConference"].iloc[0])
    away = games[games["awayTeam"] == team]
    if not away.empty:
        return str(away["awayConference"].iloc[0])
    return ""


# ── Shot-zone metrics ──────────────────────────────────────────────────────────
def compute_zone_metrics(shots: pd.DataFrame, team: str, side: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per shot zone:
        zone | fga | fga_pct | fg_pct | efg_pct | ast_pct
    """
    df = _filter(shots, team, side)
    df = df[df["shot_zone"].isin(ZONE_ORDER)]
    total = len(df)

    rows = []
    for z in ZONE_ORDER:
        zdf = df[df["shot_zone"] == z]
        fga = len(zdf)
        fgm = zdf["made"].sum()
        ast = zdf["assisted"].sum()
        rows.append(
            {
                "zone_key": z,
                "zone": ZONE_LABELS[z],
                "fga":     fga,
                "fga_pct": round(100.0 * fga / total, 1) if total > 0 else 0.0,
                "fg_pct":  round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
                "efg_pct": round(_efg(zdf), 1),
                "ast_pct": round(100.0 * ast / fga, 1) if fga > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ── Possession-bucket metrics ──────────────────────────────────────────────────
def compute_bucket_metrics(shots: pd.DataFrame, team: str, side: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per possession bucket:
        bucket | fga | fga_pct | fg_pct | efg_pct | ast_pct
    """
    df = _filter(shots, team, side)
    df = df[df["poss_bucket"].isin(BUCKET_ORDER)]
    total = len(df)

    rows = []
    for b in BUCKET_ORDER:
        bdf = df[df["poss_bucket"] == b]
        fga = len(bdf)
        fgm = bdf["made"].sum()
        ast = bdf["assisted"].sum()
        rows.append(
            {
                "bucket_key": b,
                "bucket":   BUCKET_LABELS[b],
                "fga":      fga,
                "fga_pct":  round(100.0 * fga / total, 1) if total > 0 else 0.0,
                "fg_pct":   round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
                "efg_pct":  round(_efg(bdf), 1),
                "ast_pct":  round(100.0 * ast / fga, 1) if fga > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ── Shot-clock × zone cross-tab ────────────────────────────────────────────────
def compute_clock_zone_metrics(
    shots: pd.DataFrame, team: str, side: str
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame: zone × clock_bucket with FGA count, fg_pct, efg_pct.
    Only includes FGA shot zones (no free throws).
    """
    df = _filter(shots, team, side)
    df = df[
        df["shot_zone"].isin(ZONE_ORDER) & df["clock_bucket"].isin(CLOCK_ORDER)
    ]

    rows = []
    for z in ZONE_ORDER:
        for c in CLOCK_ORDER:
            sub = df[(df["shot_zone"] == z) & (df["clock_bucket"] == c)]
            fga = len(sub)
            fgm = sub["made"].sum()
            rows.append(
                {
                    "zone":         ZONE_LABELS[z],
                    "clock_bucket": CLOCK_LABELS[c],
                    "fga":          fga,
                    "fg_pct":       round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
                    "efg_pct":      round(_efg(sub), 1),
                }
            )
    return pd.DataFrame(rows)


# ── Assisted / unassisted by zone ──────────────────────────────────────────────
def compute_assisted_metrics(
    shots: pd.DataFrame, team: str, side: str
) -> pd.DataFrame:
    """
    Returns DataFrame: zone | assisted_fga_pct | unassisted_fga_pct |
                             assisted_fg_pct   | unassisted_fg_pct
    """
    df = _filter(shots, team, side)
    df = df[df["shot_zone"].isin(ZONE_ORDER)]

    rows = []
    for z in ZONE_ORDER:
        zdf = df[df["shot_zone"] == z]
        a   = zdf[zdf["assisted"] == True]
        u   = zdf[zdf["assisted"] == False]
        fga = len(zdf)
        rows.append(
            {
                "zone_key": z,
                "zone": ZONE_LABELS[z],
                "fga":           fga,
                "ast_share":     round(100.0 * len(a) / fga, 1) if fga > 0 else np.nan,
                "ast_fg_pct":    round(100.0 * a["made"].sum() / len(a), 1) if len(a) > 0 else np.nan,
                "unast_fg_pct":  round(100.0 * u["made"].sum() / len(u), 1) if len(u) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ── Possession-type summary (from possessions_enriched directly) ──────────────
def compute_poss_type_summary(
    poss: pd.DataFrame, games: pd.DataFrame, team: str, side: str
) -> pd.DataFrame:
    """
    Summarise raw possessions (including non-shooting) by possession_type.
    Returns: poss_type | count | share | made_fg_pct | to_pct
    """
    if poss.empty or games.empty:
        return pd.DataFrame()

    if side == "Offense":
        df = poss[poss["possession_team"] == team].copy()
    else:
        # For defense: all possessions in games where team played,
        # excluding the team's own possessions (i.e., opponent possessions)
        team_game_ids = set(
            games.loc[
                (games["homeTeam"] == team) | (games["awayTeam"] == team), "id"
            ]
        )
        df = poss[
            (poss["gameId"].isin(team_game_ids))
            & (poss["possession_team"] != team)
            & (poss["possession_team"].notna())
        ].copy()

    ptype_order = ["transition", "half_court", "second_chance", "scramble_putback"]
    df = df[df["possession_type"].isin(ptype_order)]
    total = len(df)

    rows = []
    for pt in ptype_order:
        sub = df[df["possession_type"] == pt]
        n = len(sub)
        made = (sub["raw_outcome"] == "made_fg").sum()
        to   = sub["raw_outcome"].isin(["turnover", "steal"]).sum()
        rows.append(
            {
                "poss_type": pt.replace("_", " ").title(),
                "count":    n,
                "share":    round(100.0 * n / total, 1) if total > 0 else 0.0,
                "fg_pct":   round(100.0 * made / n, 1) if n > 0 else np.nan,
                "to_pct":   round(100.0 * to / n, 1) if n > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)
