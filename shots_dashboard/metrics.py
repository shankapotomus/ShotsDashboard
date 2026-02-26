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
    Aggregate season-level four-factors + transition pts/100 for *team*.

    Note
    ----
    FGM and 3PM in the four_factors table are zeroed out for ~51 % of rows
    (pipeline data issue).  eFG % is therefore computed from the shots table
    directly, which is always accurate.  FT Rate uses FTA/FGA (both populated).
    TO %, ORB % are read from the four_factors table (accurate).
    Transition Pts/100 = (2*trans_FGM + trans_3PM) / total_poss * 100.
    """
    # ── eFG% + transition pts from shots ──────────────────────────────────────
    if side == "Offense":
        sh = shots[shots["team"] == team]
    else:
        sh = shots[shots["opponent"] == team]
    sh = sh[sh["shot_zone"] != "free_throw"]

    fga_shots = len(sh)
    fgm_shots = sh["made"].sum()
    made_3    = sh.loc[sh["is_three"] == True, "made"].sum()
    efg = (
        round(100.0 * (fgm_shots + 0.5 * made_3) / fga_shots, 1)
        if fga_shots > 0
        else np.nan
    )

    # transition points: 2 pts per any FGM + 1 bonus pt per 3PM
    sh_t      = sh[sh["poss_bucket"] == "transition"]
    trans_fgm = int(sh_t["made"].sum())
    trans_3pm = int(sh_t.loc[sh_t["is_three"] == True, "made"].sum())
    trans_pts = 2 * trans_fgm + trans_3pm

    # ── other factors from four_factors table ─────────────────────────────────
    _empty = {"efg": efg, "to_pct": np.nan, "orb_pct": np.nan,
              "ft_rate": np.nan, "trans_pts_per100": np.nan, "games": 0}
    if ff.empty:
        return _empty

    tf = ff[ff["team"] == team] if side == "Offense" else ff[ff["opponent"] == team]
    if tf.empty:
        return _empty

    fga_ff  = tf["FGA"].sum()
    tov     = tf["TOV"].sum()
    poss    = tf["Possessions"].sum()
    orb     = tf["ORB"].sum()
    opp_drb = tf["Opp_DRB"].sum()
    fta     = tf["FTA"].sum()

    return {
        "efg":              efg,
        "to_pct":           round(100.0 * tov / poss, 1) if poss > 0 else np.nan,
        "orb_pct":          round(100.0 * orb / (orb + opp_drb), 1) if (orb + opp_drb) > 0 else np.nan,
        "ft_rate":          round(100.0 * fta / fga_ff, 1) if fga_ff > 0 else np.nan,
        "trans_pts_per100": round(100.0 * trans_pts / poss, 1) if poss > 0 else np.nan,
        "games":            tf["game_id"].nunique() if "game_id" in tf.columns else len(tf),
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
        # ast_pct = % of MAKES that were assisted (is_assisted only fires on makes)
        ast = zdf["is_assisted"].sum()
        rows.append(
            {
                "zone_key": z,
                "zone": ZONE_LABELS[z],
                "fga":     fga,
                "fga_pct": round(100.0 * fga / total, 1) if total > 0 else 0.0,
                "fg_pct":  round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
                "efg_pct": round(_efg(zdf), 1),
                "ast_pct": round(100.0 * ast / fgm, 1) if fgm > 0 else np.nan,
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
        # ast_pct = % of MAKES that were assisted (is_assisted only fires on makes)
        ast = bdf["is_assisted"].sum()
        rows.append(
            {
                "bucket_key": b,
                "bucket":   BUCKET_LABELS[b],
                "fga":      fga,
                "fga_pct":  round(100.0 * fga / total, 1) if total > 0 else 0.0,
                "fg_pct":   round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
                "efg_pct":  round(_efg(bdf), 1),
                "ast_pct":  round(100.0 * ast / fgm, 1) if fgm > 0 else np.nan,
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
    Returns DataFrame: zone | fga | fg_pct | ast_on_makes

    ast_on_makes = % of made FGs that were assisted (is_assisted / made).
    is_assisted uses assisted_by.notna() which has ~52% coverage on makes —
    matching the expected D1 median — and is always False on misses.
    """
    df = _filter(shots, team, side)
    df = df[df["shot_zone"].isin(ZONE_ORDER)]

    rows = []
    for z in ZONE_ORDER:
        zdf  = df[df["shot_zone"] == z]
        fga  = len(zdf)
        fgm  = zdf["made"].sum()
        ast  = zdf["is_assisted"].sum()   # only non-zero on makes
        rows.append(
            {
                "zone_key":    z,
                "zone":        ZONE_LABELS[z],
                "fga":         fga,
                "fg_pct":      round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
                "ast_on_makes": round(100.0 * ast / fgm, 1) if fgm > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ── 3-Point context ────────────────────────────────────────────────────────────
GRADE_ORDER  = ["red", "yellow", "green"]
GRADE_LABELS = {"red": "Red (Close Hard)", "yellow": "Yellow (Respect)", "green": "Green (Let Shoot)"}
GRADE_COLORS = {"red": "#e05c5c", "yellow": "#f5c542", "green": "#4ec97b"}

CLOCK_3PT_ORDER  = ["transition_pace", "normal", "late_clock", "extra"]
CLOCK_3PT_LABELS = {
    "transition_pace": "Transition",
    "normal":          "Normal",
    "late_clock":      "Late Clock",
    "extra":           "Scramble",
}


def compute_threept_context(shots: pd.DataFrame, team: str, side: str) -> dict:
    """
    Returns a dict with three sub-DataFrames for the 3-point context section:

    'location'  — corner_3 vs non_corner_3
        zone | fga | fga_pct | fg_pct | ast_on_makes

    'grade'     — shooter Red / Yellow / Green closeout grade
        grade | fga | fga_pct | fg_pct

    'clock'     — clock bucket distribution of 3pt FGA
        clock | fga | fga_pct | fg_pct

    All denominators are 3pt FGA only (is_three == True).
    ast_on_makes = % of made 3s that were assisted (is_assisted on makes).
    """
    df   = _filter(shots, team, side)
    df3  = df[df["is_three"] == True].copy()
    total = len(df3)

    # ── location ──────────────────────────────────────────────────────────────
    loc_rows = []
    for zone_key, zone_label in [("corner_3", "Corner 3"), ("non_corner_3", "Above-Break 3")]:
        z   = df3[df3["shot_zone"] == zone_key]
        fga = len(z)
        fgm = z["made"].sum()
        ast = z["is_assisted"].sum()
        loc_rows.append({
            "zone":         zone_label,
            "fga":          fga,
            "fga_pct":      round(100.0 * fga / total, 1) if total > 0 else 0.0,
            "fg_pct":       round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
            "ast_on_makes": round(100.0 * ast / fgm, 1) if fgm > 0 else np.nan,
        })

    # ── shooter grade ─────────────────────────────────────────────────────────
    grade_rows = []
    for g in GRADE_ORDER:
        gdf = df3[df3["shooter_3pt_grade"] == g]
        fga = len(gdf)
        fgm = gdf["made"].sum()
        grade_rows.append({
            "grade":    GRADE_LABELS[g],
            "grade_key": g,
            "fga":      fga,
            "fga_pct":  round(100.0 * fga / total, 1) if total > 0 else 0.0,
            "fg_pct":   round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
        })

    # ── clock ─────────────────────────────────────────────────────────────────
    clock_rows = []
    for c in CLOCK_3PT_ORDER:
        cdf = df3[df3["clock_bucket"] == c]
        fga = len(cdf)
        fgm = cdf["made"].sum()
        clock_rows.append({
            "clock":    CLOCK_3PT_LABELS[c],
            "clock_key": c,
            "fga":      fga,
            "fga_pct":  round(100.0 * fga / total, 1) if total > 0 else 0.0,
            "fg_pct":   round(100.0 * fgm / fga, 1) if fga > 0 else np.nan,
        })

    return {
        "total_3pt_fga":  total,
        "overall_fg_pct": round(100.0 * df3["made"].sum() / total, 1) if total > 0 else np.nan,
        "overall_ast":    round(100.0 * df3["is_assisted"].sum() / df3["made"].sum(), 1)
                          if df3["made"].sum() > 0 else np.nan,
        "corner_pct":     round(100.0 * (df3["shot_zone"] == "corner_3").sum() / total, 1)
                          if total > 0 else np.nan,
        "location":       pd.DataFrame(loc_rows),
        "grade":          pd.DataFrame(grade_rows),
        "clock":          pd.DataFrame(clock_rows),
    }


# ── D1 team identification ────────────────────────────────────────────────────
def get_d1_teams(games: pd.DataFrame) -> set:
    """Return set of school names that appear in 10+ games (D1 classification)."""
    home_g = games.groupby("homeTeam")["id"].nunique()
    away_g = games.groupby("awayTeam")["id"].nunique()
    return set((home_g.add(away_g, fill_value=0)).pipe(lambda s: s[s >= 10]).index)


# ── League-wide stats for percentile ranking ──────────────────────────────────
def compute_league_stats(
    shots: pd.DataFrame, ff: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """
    Vectorised computation of four-factors + transition pts/100 for every D1
    team (defined as any team appearing in 10+ games in the games table).

    Returns tidy DataFrame:
        team | side | efg | to_pct | orb_pct | ft_rate | trans_pts_per100

    Used exclusively for percentile ranking — not displayed directly.
    """
    # ── D1 teams: 10+ games ───────────────────────────────────────────────────
    home_g = games.groupby("homeTeam")["id"].nunique()
    away_g = games.groupby("awayTeam")["id"].nunique()
    d1_teams = set((home_g.add(away_g, fill_value=0)).pipe(lambda s: s[s >= 10]).index)

    sh = shots[shots["shot_zone"] != "free_throw"].copy()
    sh["_made"]  = sh["made"].astype(int)
    sh["_3made"] = ((sh["is_three"] == True) & (sh["made"] == True)).astype(int)

    dfs = []
    for side in ["Offense", "Defense"]:
        tc = "team" if side == "Offense" else "opponent"

        sh_s = sh[sh[tc].isin(d1_teams)]

        # eFG% per team
        efg_df = (
            sh_s.groupby(tc)
            .agg(_fga=("_made", "count"), _fgm=("_made", "sum"), _3pm=("_3made", "sum"))
            .assign(efg=lambda d: 100.0 * (d["_fgm"] + 0.5 * d["_3pm"]) / d["_fga"])
            [["efg"]]
        )

        # transition pts per team
        trans_df = (
            sh_s[sh_s["poss_bucket"] == "transition"]
            .groupby(tc)
            .agg(_fgm=("_made", "sum"), _3pm=("_3made", "sum"))
            .assign(trans_pts=lambda d: 2 * d["_fgm"] + d["_3pm"])
            [["trans_pts"]]
        )

        # TO%, ORB%, FT Rate from ff table
        ff_df = (
            ff[ff[tc].isin(d1_teams)]
            .groupby(tc)
            .agg(
                fga=("FGA", "sum"), tov=("TOV", "sum"), poss=("Possessions", "sum"),
                orb=("ORB", "sum"), opp_drb=("Opp_DRB", "sum"), fta=("FTA", "sum"),
            )
        )

        combo = efg_df.join(trans_df, how="outer").join(ff_df, how="outer")
        combo["trans_pts"] = combo["trans_pts"].fillna(0)
        combo["to_pct"]           = 100.0 * combo["tov"]  / combo["poss"]
        combo["orb_pct"]          = 100.0 * combo["orb"]  / (combo["orb"] + combo["opp_drb"])
        combo["ft_rate"]          = 100.0 * combo["fta"]  / combo["fga"]
        combo["trans_pts_per100"] = 100.0 * combo["trans_pts"] / combo["poss"]
        combo["side"] = side
        combo.index.name = "team"
        dfs.append(
            combo.reset_index()[
                ["team", "side", "efg", "to_pct", "orb_pct", "ft_rate", "trans_pts_per100"]
            ]
        )

    return pd.concat(dfs, ignore_index=True)


# ── Percentile rank helper ─────────────────────────────────────────────────────
# For each metric: which direction is "better"?
_HIGHER_IS_BETTER = {
    "Offense": {
        "efg":              True,
        "to_pct":           False,   # fewer turnovers = better
        "orb_pct":          True,
        "ft_rate":          True,
        "trans_pts_per100": True,
    },
    "Defense": {
        "efg":              False,   # lower allowed = better
        "to_pct":           True,    # more forced TOs = better
        "orb_pct":          False,   # fewer opp OREBs = better
        "ft_rate":          False,   # fewer FTs allowed = better
        "trans_pts_per100": False,   # fewer transition pts allowed = better
    },
}


def percentile_rank(
    league_df: pd.DataFrame, team: str, side: str, metric: str
) -> int | None:
    """
    Returns 0-100 percentile where 100 = best in D1 for that side/metric.
    Returns None if the team or metric is missing.
    """
    sub = league_df[league_df["side"] == side]
    vals = sub[metric].dropna()
    if vals.empty:
        return None

    team_row = sub[sub["team"] == team]
    if team_row.empty:
        return None
    val = team_row[metric].iloc[0]
    if pd.isna(val):
        return None

    # % of D1 teams at or below this value (0-100)
    raw_pct = int(round(100.0 * (vals <= val).sum() / len(vals)))
    higher_is_better = _HIGHER_IS_BETTER.get(side, {}).get(metric, True)
    return raw_pct if higher_is_better else (100 - raw_pct)


def _ordinal(n: int) -> str:
    """1 → '1st', 2 → '2nd', 87 → '87th', etc."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ── Possession-type summary (from possessions_enriched directly) ──────────────
_PTYPE_ORDER = ["transition", "half_court", "second_chance", "scramble_putback"]
_PTYPE_LABELS = {
    "transition":       "Transition",
    "half_court":       "Half Court",
    "second_chance":    "Second Chance",
    "scramble_putback": "Scramble Putback",
}


def compute_poss_type_summary(
    poss: pd.DataFrame,
    games: pd.DataFrame,
    team: str,
    side: str,
    shots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Summarise raw possessions (including non-shooting) by possession_type.

    Returns: poss_type | poss_type_key | count | share | fg_pct | to_pct | ppp

    ``ppp`` = points per possession from field goals (FTs excluded from shots
    data).  Requires the enriched shots DataFrame to be passed in.
    """
    if poss.empty or games.empty:
        return pd.DataFrame()

    if side == "Offense":
        df = poss[poss["possession_team"] == team].copy()
    else:
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

    df = df[df["possession_type"].isin(_PTYPE_ORDER)]
    total = len(df)

    # ── PPP numerator from shots (2*FGM + made_3, per possession_type) ───────
    pts_by_type: dict[str, float] = {}
    if shots is not None and not shots.empty and "possession_type" in shots.columns:
        tc = "team" if side == "Offense" else "opponent"
        sh = shots[shots[tc] == team]
        sh = sh[sh["shot_zone"] != "free_throw"]
        _pts = sh["made"].astype(int) * 2 + (
            (sh["is_three"] == True) & (sh["made"] == True)
        ).astype(int)
        pts_by_type = (
            sh.assign(_pts=_pts)
            .groupby("possession_type")["_pts"]
            .sum()
            .to_dict()
        )

    rows = []
    for pt in _PTYPE_ORDER:
        sub  = df[df["possession_type"] == pt]
        n    = len(sub)
        made = (sub["raw_outcome"] == "made_fg").sum()
        to   = sub["raw_outcome"].isin(["turnover", "steal"]).sum()
        pts  = pts_by_type.get(pt, np.nan)
        ppp  = round(pts / n, 3) if (pd.notna(pts) and n > 0) else np.nan
        rows.append({
            "poss_type":     _PTYPE_LABELS[pt],
            "poss_type_key": pt,
            "count":         n,
            "share":         round(100.0 * n / total, 1) if total > 0 else 0.0,
            "fg_pct":        round(100.0 * made / n, 1) if n > 0 else np.nan,
            "to_pct":        round(100.0 * to / n, 1) if n > 0 else np.nan,
            "ppp":           ppp,
        })
    return pd.DataFrame(rows)


# ── League-wide PPP per possession type (for percentile ranking) ──────────────
def compute_poss_ppp_league(
    shots: pd.DataFrame, poss: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """
    Vectorised PPP + share per possession type for every D1 team.

    Returns tidy DataFrame:
        team | side | possession_type | share | ppp

    Used for percentile ranking of possession-type efficiency and frequency.
    """
    home_g = games.groupby("homeTeam")["id"].nunique()
    away_g = games.groupby("awayTeam")["id"].nunique()
    d1_teams = set((home_g.add(away_g, fill_value=0)).pipe(lambda s: s[s >= 10]).index)

    sh = shots[shots["shot_zone"] != "free_throw"].copy()
    sh["_pts"] = (
        sh["made"].astype(int) * 2
        + ((sh["is_three"] == True) & (sh["made"] == True)).astype(int)
    )
    sh_pt = sh[sh["possession_type"].isin(_PTYPE_ORDER)]

    # Build defending_team column once (vectorised)
    gmap = (
        games[["id", "homeTeam", "awayTeam"]]
        .drop_duplicates("id")
        .set_index("id")
    )
    poss_d1 = poss[poss["possession_type"].isin(_PTYPE_ORDER)].copy()
    poss_d1 = poss_d1.join(gmap, on="gameId", how="left")
    poss_d1["defending_team"] = np.where(
        poss_d1["possession_team"] == poss_d1["homeTeam"],
        poss_d1["awayTeam"],
        poss_d1["homeTeam"],
    )

    dfs = []
    for side in ["Offense", "Defense"]:
        if side == "Offense":
            sh_grp  = "team"
            ps_s    = poss_d1[poss_d1["possession_team"].isin(d1_teams)]
            ps_grp  = "possession_team"
        else:
            sh_grp  = "opponent"
            ps_s    = poss_d1[poss_d1["defending_team"].isin(d1_teams)]
            ps_grp  = "defending_team"

        sh_s = sh_pt[sh_pt[sh_grp].isin(d1_teams)]

        pts_df = (
            sh_s.groupby([sh_grp, "possession_type"])["_pts"]
            .sum()
            .reset_index()
            .rename(columns={sh_grp: "team", "_pts": "pts"})
        )
        poss_ct = (
            ps_s.groupby([ps_grp, "possession_type"])
            .size()
            .reset_index(name="n")
            .rename(columns={ps_grp: "team"})
        )
        total_ct = (
            ps_s.groupby(ps_grp)
            .size()
            .reset_index(name="total")
            .rename(columns={ps_grp: "team"})
        )

        combo = (
            poss_ct
            .merge(pts_df, on=["team", "possession_type"], how="left")
            .merge(total_ct, on="team", how="left")
        )
        combo["pts"]   = combo["pts"].fillna(0)
        combo["ppp"]   = combo["pts"] / combo["n"]
        combo["share"] = 100.0 * combo["n"] / combo["total"]
        combo["side"]  = side
        dfs.append(combo[["team", "side", "possession_type", "share", "ppp"]])

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Directional rules for poss-type percentiles ───────────────────────────────
# For PPP: offense wants higher, defense wants lower (allowing fewer points)
# For share: transition/SC/scramble higher is better for offense (more easy shots);
#            half_court higher is neutral→worse (fewer easy shot chances).
_POSS_HIGHER_IS_BETTER: dict[str, dict[str, dict[str, bool]]] = {
    "ppp": {
        "Offense": {"transition": True,  "half_court": True,  "second_chance": True,  "scramble_putback": True},
        "Defense": {"transition": False, "half_court": False, "second_chance": False, "scramble_putback": False},
    },
    "share": {
        "Offense": {"transition": True,  "half_court": False, "second_chance": True,  "scramble_putback": True},
        "Defense": {"transition": False, "half_court": True,  "second_chance": False, "scramble_putback": False},
    },
}


def poss_percentile_rank(
    league_df: pd.DataFrame,
    team: str,
    side: str,
    poss_type: str,
    metric: str,
) -> int | None:
    """
    Returns 0-100 percentile where 100 = best in D1 for that poss_type/metric/side.
    metric is 'ppp' or 'share'.
    """
    sub  = league_df[
        (league_df["side"] == side) & (league_df["possession_type"] == poss_type)
    ]
    vals = sub[metric].dropna()
    if vals.empty:
        return None
    team_row = sub[sub["team"] == team]
    if team_row.empty:
        return None
    val = team_row[metric].iloc[0]
    if pd.isna(val):
        return None
    raw_pct = int(round(100.0 * (vals <= val).sum() / len(vals)))
    higher_is_better = (
        _POSS_HIGHER_IS_BETTER.get(metric, {})
        .get(side, {})
        .get(poss_type, True)
    )
    return raw_pct if higher_is_better else (100 - raw_pct)
