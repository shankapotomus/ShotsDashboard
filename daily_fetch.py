#!/usr/bin/env python3
"""
daily_fetch.py
--------------
Automated daily pull of college basketball play-by-play data.

By default fetches yesterday's games. Pass --date YYYY-MM-DD to override.

Required env var:
    CBBD_API_KEY   — your CollegeBasketballData.com bearer token

Usage:
    python daily_fetch.py                    # yesterday
    python daily_fetch.py --date 2026-02-18  # specific date
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")
except ImportError:
    _EASTERN = None  # Python < 3.9; falls back to system timezone

import cbbd
import numpy as np
import pandas as pd
from cbbd.rest import ApiException

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEASON = 2026         # 2025-26 season
OUTPUT_DIR = "cbbd_data"
FG_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot"}


# ---------------------------------------------------------------------------
# Text helpers  (dual API format support)
# ---------------------------------------------------------------------------
def _safe_txt(val):
    return str(val).lower() if pd.notna(val) else ""


def _safe_str(val):
    return str(val) if pd.notna(val) else ""


def _is_made(txt):
    return "makes" in txt or (" made " in txt and "missed" not in txt)


def _is_missed(txt):
    return "misses" in txt or "missed" in txt


# ---------------------------------------------------------------------------
# Substitution parser
# ---------------------------------------------------------------------------
def parse_substitution(play_text):
    match = re.search(
        r"(.+?)\s+subbing\s+(in|out)\s+for\s+(.+?)$", play_text, re.IGNORECASE
    )
    if match:
        return {
            "player": match.group(1).strip(),
            "action": match.group(2).lower(),
            "team": match.group(3).strip(),
        }
    return None


# ---------------------------------------------------------------------------
# Lineup tracker
# ---------------------------------------------------------------------------
def track_lineups_with_real_starters(game_df, starters_by_team):
    teams = game_df[game_df["team"].notna()]["team"].unique()
    lineups = {team: set(starters_by_team.get(team, [])) for team in teams}

    game_df_sorted = game_df.sort_values(
        ["period", "secondsRemaining"], ascending=[True, False]
    )
    lineup_log = []
    current_period = current_clock = None
    pending_plays = []

    def process_pending():
        nonlocal pending_plays
        if not pending_plays:
            return
        for play in pending_plays:
            if play.get("playType") == "Substitution":
                team = play.get("team")
                sub = parse_substitution(play.get("playText", ""))
                if sub and team:
                    if sub["action"] == "in":
                        lineups[team].add(sub["player"])
                    else:
                        lineups[team].discard(sub["player"])
        for play in pending_plays:
            lineup_log.append(
                {
                    "play_id": play.get("id"),
                    "period": play.get("period"),
                    "clock": play.get("clock"),
                    "seconds_remaining": play.get("secondsRemaining"),
                    "play_type": play.get("playType"),
                    "team": play.get("team"),
                    "home_lineup": list(lineups.get(teams[0], set())) if len(teams) > 0 else [],
                    "away_lineup": list(lineups.get(teams[1], set())) if len(teams) > 1 else [],
                    "home_lineup_size": len(lineups.get(teams[0], set())) if len(teams) > 0 else 0,
                    "away_lineup_size": len(lineups.get(teams[1], set())) if len(teams) > 1 else 0,
                }
            )
        pending_plays = []

    for _, play in game_df_sorted.iterrows():
        period = play.get("period")
        clock = play.get("clock")
        if (period, clock) != (current_period, current_clock):
            process_pending()
            current_period = period
            current_clock = clock
        pending_plays.append(play.to_dict())

    process_pending()
    return pd.DataFrame(lineup_log), lineups


def lineup_to_key(lineup_list):
    if isinstance(lineup_list, list):
        return " | ".join(sorted(lineup_list))
    return None


def get_lineup_stints(pbp_df):
    stints = []
    prev_home = prev_away = None
    stint_start = stint_start_score = None

    for _, row in pbp_df.iterrows():
        home = tuple(sorted(row["home_lineup"]))
        away = tuple(sorted(row["away_lineup"]))
        if (home, away) != (prev_home, prev_away):
            if prev_home is not None:
                stints.append(
                    {
                        "home_lineup_key": " | ".join(prev_home),
                        "away_lineup_key": " | ".join(prev_away),
                        "start_seconds": stint_start,
                        "end_seconds": row["secondsRemaining"],
                        "start_home_score": stint_start_score[0],
                        "start_away_score": stint_start_score[1],
                        "end_home_score": row["homeScore"],
                        "end_away_score": row["awayScore"],
                    }
                )
            prev_home = home
            prev_away = away
            stint_start = row["secondsRemaining"]
            stint_start_score = (row["homeScore"], row["awayScore"])

    if prev_home is not None:
        stints.append(
            {
                "home_lineup_key": " | ".join(prev_home),
                "away_lineup_key": " | ".join(prev_away),
                "start_seconds": stint_start,
                "end_seconds": 0,
                "start_home_score": stint_start_score[0],
                "start_away_score": stint_start_score[1],
                "end_home_score": pbp_df.iloc[-1]["homeScore"],
                "end_away_score": pbp_df.iloc[-1]["awayScore"],
            }
        )
    return pd.DataFrame(stints)


# ---------------------------------------------------------------------------
# Last free-throw detection
# ---------------------------------------------------------------------------
def _precompute_last_ft_flags(game_df):
    """Pre-compute which MadeFreeThrow rows are the last FT in a sequence.

    NEW format: has 'M of N' pattern — last when M == N (e.g. '2 of 2').
    OLD format: no 'M of N'; detect last FT by checking whether the next
                play is also a MadeFreeThrow at the same clock time.

    Uses 'id' as sort tiebreaker for stable ordering at the same clock.
    """
    sorted_df = game_df.sort_values(
        ["period", "secondsRemaining", "id"], ascending=[True, False, True]
    ).reset_index()
    is_last_ft = {}
    ft_mask = sorted_df["playType"] == "MadeFreeThrow"
    ft_indices = sorted_df.index[ft_mask].tolist()

    for pos in ft_indices:
        row = sorted_df.iloc[pos]
        txt_lower = _safe_txt(row["playText"])
        orig_idx = row["index"]

        # New format: use regex to extract M and N from "M of N"
        m_of_n = re.search(r"(\d+)\s+of\s+(\d+)", txt_lower)
        if m_of_n:
            m_val, n_val = int(m_of_n.group(1)), int(m_of_n.group(2))
            is_last_ft[orig_idx] = (m_val == n_val)
            continue

        # Old format: check if the NEXT row is also a FT at the same clock
        if pos + 1 < len(sorted_df):
            next_row = sorted_df.iloc[pos + 1]
            same_clock = (
                next_row["period"] == row["period"]
                and next_row["secondsRemaining"] == row["secondsRemaining"]
            )
            next_is_ft = next_row["playType"] == "MadeFreeThrow"
            is_last_ft[orig_idx] = not (same_clock and next_is_ft)
        else:
            is_last_ft[orig_idx] = True
    return is_last_ft


# ---------------------------------------------------------------------------
# Technical foul free-throw detection
# ---------------------------------------------------------------------------
def _precompute_tech_ft_flags(game_df):
    """Pre-compute which MadeFreeThrow rows follow a Technical Foul.

    Technical FTs don't change possession — the fouled team shoots then
    retains the ball. Detected by finding a TechnicalFoul play immediately
    preceding the FT at the same clock time.

    Returns dict: original_index -> bool (True = this is a tech FT).
    """
    sorted_df = game_df.sort_values(
        ["period", "secondsRemaining", "id"], ascending=[True, False, True]
    ).reset_index()
    is_tech_ft = {}

    ft_mask = sorted_df["playType"] == "MadeFreeThrow"
    ft_indices = sorted_df.index[ft_mask].tolist()

    for pos in ft_indices:
        row = sorted_df.iloc[pos]
        orig_idx = row["index"]

        found_tech = False
        for back in range(pos - 1, max(pos - 6, -1), -1):
            prev = sorted_df.iloc[back]
            if prev["period"] != row["period"]:
                break
            if prev["secondsRemaining"] != row["secondsRemaining"]:
                break
            if "Technical" in _safe_str(prev.get("playType", "")):
                found_tech = True
                break
            if prev["playType"] in ("MadeFreeThrow", "PersonalFoul"):
                continue
            break

        is_tech_ft[orig_idx] = found_tech

    return is_tech_ft


# ---------------------------------------------------------------------------
# Dead Ball Rebound classification
# ---------------------------------------------------------------------------
def _classify_dead_ball_rebounds(game_df, last_ft_flags):
    """Classify each Dead Ball Rebound to determine whether it ends possession.

    Categories:
      'mid_ft_sequence'    — between FTs of a multi-FT trip -> don't end possession
      'after_made_last_ft' — after a made last FT -> don't end (already ended)
      'same_team_fg_miss'  — same team missed FG (offensive DB reb) -> don't end
      'end_possession'     — default; ends the current possession

    Returns dict: original_index -> category string.
    """
    sorted_df = game_df.sort_values(
        ["period", "secondsRemaining", "id"], ascending=[True, False, True]
    ).reset_index()
    db_reb_class = {}

    db_mask = sorted_df["playType"] == "Dead Ball Rebound"
    db_indices = sorted_df.index[db_mask].tolist()

    for pos in db_indices:
        row = sorted_df.iloc[pos]
        orig_idx = row["index"]
        db_team = row["team"]

        category = "end_possession"
        for back in range(pos - 1, max(pos - 10, -1), -1):
            prev = sorted_df.iloc[back]
            if prev["period"] != row["period"]:
                break
            prev_pt = _safe_str(prev.get("playType", ""))

            if prev_pt in ("Substitution", "Official TV Timeout", ""):
                continue

            if prev_pt == "MadeFreeThrow":
                prev_orig = prev["index"]
                prev_is_last = last_ft_flags.get(prev_orig, False)
                if not prev_is_last:
                    category = "mid_ft_sequence"
                else:
                    prev_txt = _safe_txt(prev.get("playText", ""))
                    if _is_made(prev_txt):
                        category = "after_made_last_ft"
                    else:
                        category = "end_possession"
                break

            if prev_pt in FG_TYPES:
                prev_txt = _safe_txt(prev.get("playText", ""))
                if _is_missed(prev_txt):
                    if pd.notna(prev["team"]) and pd.notna(db_team) and prev["team"] == db_team:
                        category = "same_team_fg_miss"
                    else:
                        category = "end_possession"
                break

            if "Turnover" in prev_pt or prev_pt in ("PersonalFoul",):
                break

        db_reb_class[orig_idx] = category

    return db_reb_class


# ---------------------------------------------------------------------------
# Possession tracker
# ---------------------------------------------------------------------------
def track_possessions_v2(game_df):
    game_df = game_df.sort_values(
        ["period", "secondsRemaining", "id"], ascending=[True, False, True]
    ).copy()
    teams = [t for t in game_df["team"].unique() if pd.notna(t)]

    def other_team(t):
        others = [x for x in teams if x != t]
        return others[0] if others else None

    last_ft_flags = _precompute_last_ft_flags(game_df)
    tech_ft_flags = _precompute_tech_ft_flags(game_df)
    db_reb_classes = _classify_dead_ball_rebounds(game_df, last_ft_flags)
    poss_id = 0
    poss_team = None
    records = []
    last_end_reason = last_end_team = None

    for idx, row in game_df.iterrows():
        pt = _safe_str(row.get("playType"))
        txt = _safe_txt(row.get("playText"))
        team = row.get("team")
        outcome = None
        end_poss = False
        next_team = None

        if pt == "Jumpball":
            if "won" in txt and team and poss_team is None:
                poss_team = team
        elif pt in FG_TYPES:
            if poss_team is None:
                poss_team = team
            if _is_made(txt):
                outcome = "made_fg"
                end_poss = True
                next_team = other_team(poss_team)
        elif pt == "MadeFreeThrow":
            # Technical FT — team retains possession, don't end
            is_tech = tech_ft_flags.get(idx, False)
            if is_tech:
                outcome = "tech_ft"
            else:
                # And-1 detection
                if (
                    team
                    and poss_team is not None
                    and team != poss_team
                    and last_end_reason == "made_fg"
                    and last_end_team == team
                ):
                    poss_id -= 1
                    poss_team = team
                    for rec in reversed(records):
                        if rec["possession_id"] == poss_id + 1:
                            rec["possession_id"] = poss_id
                            rec["possession_team"] = poss_team
                        else:
                            break
                    last_end_reason = "and1_ft"
                elif poss_team is None and team:
                    poss_team = team
                is_last = last_ft_flags.get(idx, False)
                if is_last:
                    if _is_made(txt):
                        outcome = "made_ft"
                        end_poss = True
                        next_team = other_team(poss_team)
                    else:
                        outcome = "missed_last_ft"
        elif "Turnover" in pt:
            if poss_team is None and team:
                poss_team = team
            outcome = "turnover"
            end_poss = True
            next_team = other_team(poss_team)
        elif pt == "Steal":
            outcome = "steal"
        elif pt == "Defensive Rebound":
            outcome = "def_rebound"
            end_poss = True
            next_team = team
        elif pt == "Offensive Rebound":
            outcome = "off_rebound"
        elif pt == "Dead Ball Rebound":
            # Context-aware: only end possession when appropriate
            db_class = db_reb_classes.get(idx, "end_possession")
            outcome = "dead_ball_rebound"
            if db_class == "end_possession":
                end_poss = True
                next_team = team
            # mid_ft_sequence, after_made_last_ft, same_team_fg_miss -> don't end
        elif pt in ("End Period", "End Game"):
            outcome = "end_period"
            end_poss = True
            next_team = None

        records.append(
            {
                "play_id": row.get("id"),
                "gameId": row.get("gameId"),
                "possession_id": poss_id,
                "possession_team": poss_team,
                "play_type": pt,
                "play_text": row.get("playText", ""),
                "team": team,
                "outcome": outcome,
            }
        )

        if end_poss:
            last_end_reason = outcome
            last_end_team = poss_team
            poss_id += 1
            poss_team = next_team
        elif pt not in (
            "PersonalFoul",
            "MadeFreeThrow",
            "Substitution",
            "Official TV Timeout",
            "",
        ):
            last_end_reason = last_end_team = None

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Possession classifier
# ---------------------------------------------------------------------------
def classify_possessions(possessions_df, game_df):
    game_sorted = game_df.sort_values(
        ["period", "secondsRemaining", "id"], ascending=[True, False, True]
    ).reset_index(drop=True)
    game_sorted["play_order"] = range(len(game_sorted))
    time_info = game_sorted[["id", "secondsRemaining", "period", "play_order"]].rename(
        columns={"id": "play_id"}
    )
    poss = possessions_df.merge(time_info, on="play_id", how="left")
    poss = poss.sort_values("play_order").reset_index(drop=True)

    possession_rows = []
    for pid in sorted(poss["possession_id"].unique()):
        grp = poss[poss["possession_id"] == pid].sort_values("play_order")
        plays = grp.to_dict("records")
        game_id = grp["gameId"].iloc[0] if "gameId" in grp.columns else None
        poss_team = grp["possession_team"].iloc[0]
        period = grp["period"].iloc[0]
        start_sec = grp["secondsRemaining"].iloc[0]
        end_sec = grp["secondsRemaining"].iloc[-1]
        duration = start_sec - end_sec

        all_outcomes = [p["outcome"] for p in plays if p["outcome"] is not None]
        final_outcome = all_outcomes[-1] if all_outcomes else None
        outcome_set = set(all_outcomes)
        has_steal = "steal" in outcome_set
        has_oreb = "off_rebound" in outcome_set

        if final_outcome == "turnover":
            refined = "live_ball_turnover" if has_steal else "dead_ball_turnover"
        elif final_outcome == "def_rebound":
            miss_type = "fga"
            found_dreb = False
            for p in reversed(plays):
                if not found_dreb:
                    if p["outcome"] == "def_rebound":
                        found_dreb = True
                    continue
                if p["play_type"] in FG_TYPES:
                    miss_type = "fga"
                    break
                if p["play_type"] == "MadeFreeThrow" and _is_missed(
                    _safe_txt(p.get("play_text"))
                ):
                    miss_type = "fta"
                    break
            refined = f"{miss_type}_def_rebound"
        elif final_outcome in ("made_fg", "made_ft", "end_period", "dead_ball_rebound"):
            refined = final_outcome
        else:
            refined = final_outcome

        for i_p, p in enumerate(plays):
            pt_lower = _safe_txt(p.get("play_text"))
            if (
                p["play_type"] in FG_TYPES
                and "block" in pt_lower
                and _is_missed(pt_lower)
            ):
                remaining = plays[i_p + 1 :]
                has_reb = any(
                    r["outcome"] in ("def_rebound", "off_rebound", "dead_ball_rebound")
                    for r in remaining
                )
                if not has_reb:
                    next_p = poss[poss["possession_id"] == pid + 1]
                    if len(next_p) > 0:
                        refined = "block_oob"
                break

        fga_plays = [p for p in plays if p["play_type"] in FG_TYPES]
        first_fga_sec = fga_plays[0]["secondsRemaining"] if fga_plays else None
        time_to_first_fga = (
            (start_sec - first_fga_sec) if first_fga_sec is not None else None
        )

        oreb_list = [p for p in plays if p["outcome"] == "off_rebound"]
        time_oreb_to_fga = None
        if oreb_list:
            oreb_sec = oreb_list[0]["secondsRemaining"]
            post_oreb_fga = [p for p in fga_plays if p["secondsRemaining"] < oreb_sec]
            if post_oreb_fga:
                time_oreb_to_fga = oreb_sec - post_oreb_fga[0]["secondsRemaining"]

        foul_plays = [
            p
            for p in plays
            if "Foul" in (p.get("play_type") or "")
            and "shooting" not in _safe_txt(p.get("play_text"))
        ]
        foul_within_10s = foul_plays and (start_sec - foul_plays[0]["secondsRemaining"]) <= 10

        if has_oreb:
            poss_type = (
                "scramble_putback"
                if (time_oreb_to_fga is not None and time_oreb_to_fga <= 3)
                else "second_chance"
            )
        elif foul_plays and foul_within_10s and not fga_plays and start_sec <= 120:
            poss_type = "intentional_foul"
        elif time_to_first_fga is not None:
            poss_type = "transition" if time_to_first_fga <= 7 else "half_court"
        else:
            poss_type = "half_court"

        possession_rows.append(
            {
                "gameId": game_id,
                "possession_id": pid,
                "possession_team": poss_team,
                "period": period,
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "duration_sec": duration,
                "raw_outcome": final_outcome,
                "refined_outcome": refined,
                "possession_type": poss_type,
                "has_oreb": has_oreb,
                "time_to_first_fga": time_to_first_fga,
                "time_oreb_to_fga": time_oreb_to_fga,
            }
        )

    result = pd.DataFrame(possession_rows)

    # Filter out end_period possessions — they are tracking artifacts, not
    # real possessions, and inflate the per-team possession count.
    result = result[result["raw_outcome"] != "end_period"].reset_index(drop=True)

    prev_enders = ["start_of_period"]
    for i_r in range(1, len(result)):
        if result.iloc[i_r]["period"] != result.iloc[i_r - 1]["period"]:
            prev_enders.append("start_of_period")
        else:
            prev_enders.append(result.iloc[i_r - 1]["refined_outcome"])
    result["prev_poss_ender"] = prev_enders
    return result


# ---------------------------------------------------------------------------
# Four Factors
# ---------------------------------------------------------------------------
def compute_four_factors(game_df):
    teams = game_df[game_df["team"].notna()]["team"].unique()
    results = {}
    for team in teams:
        tp = game_df[game_df["team"] == team]
        opp = [t for t in teams if t != team]
        opp_plays = game_df[game_df["team"] == opp[0]] if opp else pd.DataFrame()

        fg = tp[tp["playType"].isin(FG_TYPES)]
        fga = len(fg)
        fg_txt = fg["playText"].fillna("").str.lower()
        fgm = (
            fg_txt.str.contains("makes")
            | (fg_txt.str.contains(" made ") & ~fg_txt.str.contains("missed"))
        ).sum()
        tpa = fg["playText"].fillna("").str.contains("three point", case=False, na=False).sum()
        tpm_mask = (fg_txt.str.contains("three point")) & (
            fg_txt.str.contains("makes")
            | (fg_txt.str.contains(" made ") & ~fg_txt.str.contains("missed"))
        )
        tpm = tpm_mask.sum()
        ft = tp[tp["playType"] == "MadeFreeThrow"]
        fta = len(ft)
        ft_txt = ft["playText"].fillna("").str.lower()
        ftm = (
            ft_txt.str.contains("makes")
            | (ft_txt.str.contains(" made ") & ~ft_txt.str.contains("missed"))
        ).sum()
        tov = len(tp[tp["playType"].str.contains("Turnover", na=False)])
        orb = len(tp[tp["playType"] == "Offensive Rebound"])
        drb = len(tp[tp["playType"] == "Defensive Rebound"])
        opp_drb = (
            len(opp_plays[opp_plays["playType"] == "Defensive Rebound"])
            if len(opp_plays)
            else 0
        )

        possessions = fga - orb + tov + 0.475 * fta
        efg = (fgm + 0.5 * tpm) / fga * 100 if fga else 0
        to_p = tov / possessions * 100 if possessions else 0
        orb_p = orb / (orb + opp_drb) * 100 if (orb + opp_drb) else 0
        ft_r = fta / fga * 100 if fga else 0
        tpa_r = tpa / fga * 100 if fga else 0
        n_per = game_df["period"].max()
        mins = 40 if n_per <= 2 else 40 + (n_per - 2) * 5
        tempo = possessions / (mins / 40)

        results[team] = {
            "FGA": fga, "FGM": int(fgm), "3PA": tpa, "3PM": int(tpm),
            "2PA": fga - tpa, "2PM": int(fgm - tpm),
            "FTA": fta, "FTM": int(ftm), "TOV": tov,
            "ORB": orb, "DRB": drb, "Opp_DRB": opp_drb,
            "Possessions": round(possessions, 1),
            "eFG%": round(efg, 1), "TO%": round(to_p, 1),
            "ORB%": round(orb_p, 1), "FT_Rate": round(ft_r, 1),
            "3PA_Rate": round(tpa_r, 1), "Tempo": round(tempo, 1),
        }
    return results


# ---------------------------------------------------------------------------
# Roster fetch
# ---------------------------------------------------------------------------
def fetch_game_roster(game_id, game_df, configuration, season):
    game_date = game_df["gameStartDate"].iloc[0]
    game_teams = game_df[game_df["team"].notna()]["team"].unique()

    with cbbd.ApiClient(configuration) as api_client:
        games_api = cbbd.GamesApi(api_client)
        all_players = []
        for team in game_teams:
            gp = games_api.get_game_players(
                start_date_range=game_date,
                end_date_range=game_date,
                team=team,
                season=season,
            )
            all_players.extend([p.to_dict() for p in gp])

    gp_df = pd.DataFrame(all_players)
    flat = []
    for _, row in gp_df.iterrows():
        team = row["team"]
        for player in row.get("players") or []:
            player["team"] = team
            flat.append(player)
    players_flat_df = pd.DataFrame(flat)

    starters_by_team = {}
    for team in players_flat_df["team"].unique():
        starters_by_team[team] = players_flat_df[
            (players_flat_df["team"] == team) & (players_flat_df["starter"] == True)
        ]["name"].tolist()

    return players_flat_df, starters_by_team


# ---------------------------------------------------------------------------
# Per-game pipeline
# ---------------------------------------------------------------------------
def process_single_game(game_id, game_df, configuration, season):
    teams = game_df[game_df["team"].notna()]["team"].unique()
    if len(teams) < 2:
        raise ValueError(f"Game {game_id}: found {len(teams)} teams, need 2")

    players_flat_df, starters_by_team = fetch_game_roster(
        game_id, game_df, configuration, season
    )

    lineup_df, _ = track_lineups_with_real_starters(game_df, starters_by_team)

    pbp_with_lineups = game_df.merge(
        lineup_df[
            ["play_id", "home_lineup", "away_lineup", "home_lineup_size", "away_lineup_size"]
        ],
        left_on="id",
        right_on="play_id",
        how="left",
    )

    # shots_df
    shots_df = pbp_with_lineups[pbp_with_lineups["shootingPlay"] == True].copy()
    for idx, row in shots_df.iterrows():
        si = row.get("shotInfo")
        if si and isinstance(si, dict):
            shots_df.at[idx, "shooter_name"] = si.get("shooter", {}).get("name")
            shots_df.at[idx, "shooter_id"] = si.get("shooter", {}).get("id")
            shots_df.at[idx, "made"] = si.get("made")
            shots_df.at[idx, "assisted"] = si.get("assisted")
            shots_df.at[idx, "assisted_by"] = si.get("assistedBy", {}).get("name")
            shots_df.at[idx, "shot_range"] = si.get("range")
            shots_df.at[idx, "x"] = si.get("location", {}).get("x")
            shots_df.at[idx, "y"] = si.get("location", {}).get("y")
    shots_df["distance"] = shots_df["playText"].str.extract(r"(\d+)-foot").astype(float)
    shots_df["is_three"] = shots_df["playText"].str.contains("three point", case=False, na=False)
    shots_df["home_lineup_key"] = shots_df["home_lineup"].apply(lineup_to_key)
    shots_df["away_lineup_key"] = shots_df["away_lineup"].apply(lineup_to_key)
    shots_df = shots_df.drop(
        columns=["home_lineup", "away_lineup", "shotInfo", "participants"], errors="ignore"
    )

    # players_df
    players_df = players_flat_df.copy()
    players_df["gameId"] = game_id
    for col in [
        "rebounds", "freeThrows", "threePointFieldGoals", "twoPointFieldGoals", "fieldGoals"
    ]:
        if col in players_df.columns:
            for idx, row in players_df.iterrows():
                if isinstance(row[col], dict):
                    for key, val in row[col].items():
                        players_df.at[idx, f"{col}_{key}"] = val
            players_df = players_df.drop(columns=[col])

    # lineup_stints_df
    lineup_stints_df = get_lineup_stints(
        pbp_with_lineups.sort_values("secondsRemaining", ascending=False)
    )
    lineup_stints_df["gameId"] = game_id
    lineup_stints_df["duration_seconds"] = (
        lineup_stints_df["start_seconds"] - lineup_stints_df["end_seconds"]
    )
    lineup_stints_df["home_pts_scored"] = (
        lineup_stints_df["end_home_score"] - lineup_stints_df["start_home_score"]
    )
    lineup_stints_df["away_pts_scored"] = (
        lineup_stints_df["end_away_score"] - lineup_stints_df["start_away_score"]
    )
    lineup_stints_df["home_plus_minus"] = (
        lineup_stints_df["home_pts_scored"] - lineup_stints_df["away_pts_scored"]
    )

    # pbp_flat
    pbp_flat = pbp_with_lineups.copy()
    pbp_flat["home_lineup_key"] = pbp_flat["home_lineup"].apply(lineup_to_key)
    pbp_flat["away_lineup_key"] = pbp_flat["away_lineup"].apply(lineup_to_key)
    pbp_flat = pbp_flat.drop(
        columns=["home_lineup", "away_lineup", "shotInfo", "participants"], errors="ignore"
    )

    # possessions
    possessions_df = track_possessions_v2(game_df)
    poss_enriched = classify_possessions(possessions_df, game_df)

    return {
        "possessions_df": possessions_df,
        "poss_enriched": poss_enriched,
        "shots_df": shots_df,
        "lineup_stints_df": lineup_stints_df,
        "players_df": players_df,
        "pbp_flat": pbp_flat,
    }


# ---------------------------------------------------------------------------
# Completeness check
# ---------------------------------------------------------------------------
def check_completeness(games_df, plays_df, failed_games):
    """
    Compare every game the API returned for the day against what was actually
    fetched and processed.

    Returns a dict with:
        expected_count   – games returned by get_games()
        pbp_fetched_count – games that had PBP data returned
        processed_count  – games that completed the full analysis pipeline
        failed_count     – games that failed at the analysis stage
        missing_pbp      – list of game IDs with no PBP data
        failed_analysis  – list of game IDs that errored during analysis
        is_complete      – True only when every expected game was fully processed
    """
    expected_ids = (
        set(games_df["id"].astype(str).tolist()) if not games_df.empty else set()
    )
    pbp_ids = (
        set(plays_df["gameId"].astype(str).tolist()) if not plays_df.empty else set()
    )
    failed_ids = {str(fg["gameId"]) for fg in failed_games}
    processed_ids = pbp_ids - failed_ids

    missing_pbp = expected_ids - pbp_ids
    failed_analysis = expected_ids & failed_ids
    is_complete = not missing_pbp and not failed_analysis

    report = {
        "expected_count": len(expected_ids),
        "pbp_fetched_count": len(pbp_ids),
        "processed_count": len(processed_ids),
        "failed_count": len(failed_ids),
        "missing_pbp": sorted(missing_pbp),
        "failed_analysis": sorted(failed_analysis),
        "is_complete": is_complete,
    }

    log.info("=== Completeness Check ===")
    log.info("  Expected games   : %d", report["expected_count"])
    log.info("  PBP fetched      : %d", report["pbp_fetched_count"])
    log.info("  Fully processed  : %d", report["processed_count"])
    log.info("  Failed           : %d", report["failed_count"])
    if missing_pbp:
        log.warning(
            "  Missing PBP (%d) : %s",
            len(missing_pbp),
            ", ".join(sorted(missing_pbp)),
        )
    if failed_analysis:
        log.warning(
            "  Failed analysis (%d): %s",
            len(failed_analysis),
            ", ".join(sorted(failed_analysis)),
        )
    if is_complete:
        log.info("  Status: COMPLETE — all %d games processed.", len(expected_ids))
    else:
        log.warning(
            "  Status: INCOMPLETE — %d/%d games fully processed.",
            len(processed_ids),
            len(expected_ids),
        )
    log.info("==========================")
    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(target_date: datetime, configuration, season: int):
    date_str = target_date.strftime("%Y%m%d")
    log.info("Fetching games for %s (season %d)", target_date.date(), season)

    with cbbd.ApiClient(configuration) as api_client:
        games_api = cbbd.GamesApi(api_client)
        plays_api = cbbd.PlaysApi(api_client)

        games = games_api.get_games(
            start_date_range=target_date,
            end_date_range=target_date,
            season=season,
        )
        games_df = pd.DataFrame([g.to_dict() for g in games])

    if games_df.empty:
        log.info("No games found for %s — nothing to do.", target_date.date())
        return None

    log.info("Found %d games", len(games_df))

    # Fetch PBP
    all_plays = []
    with cbbd.ApiClient(configuration) as api_client:
        plays_api = cbbd.PlaysApi(api_client)
        for _, game in games_df.iterrows():
            gid = game["id"]
            try:
                plays = plays_api.get_plays(game_id=gid)
                all_plays.extend([p.to_dict() for p in plays])
            except ApiException as e:
                log.warning("Game %s: PBP fetch failed (%s)", gid, e)
            time.sleep(0.5)

    plays_df = pd.DataFrame(all_plays)
    log.info("Collected %d plays across %d games", len(plays_df), plays_df["gameId"].nunique())

    # Per-game analysis
    unique_game_ids = plays_df["gameId"].unique()
    n_games = len(unique_game_ids)

    accum = {k: [] for k in
             ["possessions_df", "poss_enriched", "shots_df",
              "lineup_stints_df", "players_df", "pbp_flat"]}
    failed_games = []
    ff_rows = []

    for i, gid in enumerate(unique_game_ids):
        game_df = plays_df[plays_df["gameId"] == gid].copy()
        teams = game_df[game_df["team"].notna()]["team"].unique()
        label = f"[{i+1}/{n_games}] Game {gid}"
        try:
            log.info("%s: %s", label, " vs ".join(teams[:2]))
            result = process_single_game(gid, game_df, configuration, season)
            for key in accum:
                accum[key].append(result[key])

            ff = compute_four_factors(game_df)
            for team, stats in ff.items():
                stats["game_id"] = gid
                stats["team"] = team
                stats["opponent"] = [t for t in teams if t != team][0]
                ff_rows.append(stats)

            log.info(
                "%s: %d plays, %d possessions",
                label, len(result["pbp_flat"]), len(result["poss_enriched"]),
            )
        except Exception as e:
            log.error("%s FAILED: %s", label, e)
            failed_games.append({"gameId": gid, "error": str(e)})

        if i < n_games - 1:
            time.sleep(1.0)

    # Concatenate
    def concat(key):
        return (
            pd.concat(accum[key], ignore_index=True)
            if accum[key]
            else pd.DataFrame()
        )

    all_possessions_df   = concat("possessions_df")
    all_poss_enriched_df = concat("poss_enriched")
    all_shots_df         = concat("shots_df")
    all_lineup_stints_df = concat("lineup_stints_df")
    all_players_df       = concat("players_df")
    all_pbp_flat_df      = concat("pbp_flat")
    all_ff_df            = pd.DataFrame(ff_rows)

    # Save CSVs — one subdir per type, one file per day
    to_save = {
        "games":                games_df,
        "plays":                plays_df,
        "possessions":          all_possessions_df,
        "possessions_enriched": all_poss_enriched_df,
        "shots":                all_shots_df,
        "lineup_stints":        all_lineup_stints_df,
        "players":              all_players_df,
        "pbp_flat":             all_pbp_flat_df,
        "four_factors":         all_ff_df,
    }
    for name, df in to_save.items():
        if df is not None and len(df) > 0:
            subdir = os.path.join(OUTPUT_DIR, name)
            os.makedirs(subdir, exist_ok=True)
            fname = os.path.join(subdir, f"{date_str}_{season}.csv")
            df.to_csv(fname, index=False)
            log.info("  Saved %s -> %s (%d rows)", name, fname, len(df))

    log.info(
        "Done. Processed %d/%d games. Failed: %d",
        n_games - len(failed_games), n_games, len(failed_games),
    )
    if failed_games:
        for fg in failed_games:
            log.warning("  Failed game %s: %s", fg["gameId"], fg["error"])

    return check_completeness(games_df, plays_df, failed_games)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch yesterday's CBBD game data.")
    parser.add_argument(
        "--date",
        help="Target date YYYY-MM-DD (default: yesterday)",
        default=None,
    )
    args = parser.parse_args()

    api_key = os.environ.get("CBBD_API_KEY", "").strip()
    if not api_key:
        log.error("CBBD_API_KEY environment variable is not set.")
        sys.exit(1)

    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            log.error("Invalid date format '%s'. Use YYYY-MM-DD.", args.date)
            sys.exit(1)
    else:
        # Compute "yesterday" in Eastern Time so the correct calendar day is
        # used regardless of the server's local timezone.
        if _EASTERN is not None:
            now_et = datetime.now(tz=_EASTERN)
        else:
            # Fallback: assume server is already in EST/EDT, or close enough.
            now_et = datetime.now()
        yesterday_et = now_et - timedelta(days=1)
        target_date = yesterday_et.replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None
        )

    configuration = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=api_key,
    )

    report = run_pipeline(target_date, configuration, SEASON)

    # Exit with a non-zero code when the run was incomplete so that cron /
    # monitoring systems can detect the failure automatically.
    if report is not None and not report["is_complete"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
