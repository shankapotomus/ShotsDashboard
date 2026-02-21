"""
load_data.py — single source of truth for loading cbbd_data CSVs.

Usage (from any notebook one level up):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cbbd_data"))
    from load_data import games_df, ff_df, poss_enriched_df, \
                          plays_df, pbp_df, shots_df, players_df, lineups_df

Or from within cbbd_data/:
    from load_data import games_df, ff_df, ...

File structure expected:
    cbbd_data/
      games/              YYYYMMDD_SEASON.csv
      plays/              YYYYMMDD_SEASON.csv
      possessions/        YYYYMMDD_SEASON.csv
      possessions_enriched/ YYYYMMDD_SEASON.csv
      shots/              YYYYMMDD_SEASON.csv
      lineup_stints/      YYYYMMDD_SEASON.csv
      players/            YYYYMMDD_SEASON.csv
      pbp_flat/           YYYYMMDD_SEASON.csv
      four_factors/       YYYYMMDD_SEASON.csv

All DataFrames are:
  • deduplicated across any overlapping daily files
  • enriched with a `game_date` column (real date from games.startDate)
"""

import glob
import os

import pandas as pd

# ── locate the data directory relative to this file ───────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _concat(subdir: str) -> pd.DataFrame:
    """Glob all CSVs inside DATA_DIR/subdir/ and concatenate."""
    paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, "*.csv")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


# ── games — dedupe on id ───────────────────────────────────────────────────────
games_df = _concat("games")
games_df = games_df.drop_duplicates(subset=["id"])
games_df["game_date"] = pd.to_datetime(games_df["startDate"], utc=True).dt.date

# Lookup table: str game_id -> game_date
_game_dates = (
    games_df[["id", "game_date"]]
    .rename(columns={"id": "game_id"})
    .assign(game_id=lambda d: d["game_id"].astype(str))
    .drop_duplicates("game_id")
)


def _attach_date(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Left-join game_date onto df using key_col as the game id."""
    df = df.copy()
    df[key_col] = df[key_col].astype(str)
    return df.merge(_game_dates.rename(columns={"game_id": key_col}),
                    on=key_col, how="left")


# ── four_factors — dedupe on (game_id, team) ──────────────────────────────────
ff_df = _concat("four_factors")
ff_df = ff_df.drop_duplicates(subset=["game_id", "team"])
ff_df = _attach_date(ff_df, "game_id")

# ── possessions_enriched — dedupe on (gameId, possession_id, possession_team) ─
poss_enriched_df = _concat("possessions_enriched")
poss_enriched_df = poss_enriched_df.drop_duplicates(
    subset=["gameId", "possession_id", "possession_team"]
)
poss_enriched_df = _attach_date(poss_enriched_df, "gameId")

# ── possessions (raw) — dedupe on (play_id, gameId, possession_team) ──────────
poss_df = _concat("possessions")
poss_df = poss_df.drop_duplicates(subset=["play_id", "gameId", "possession_team"])
poss_df = _attach_date(poss_df, "gameId")

# ── plays — dedupe on id ───────────────────────────────────────────────────────
plays_df = _concat("plays")
plays_df = plays_df.drop_duplicates(subset=["id"])
plays_df = _attach_date(plays_df, "gameId")

# ── pbp_flat — dedupe on id ───────────────────────────────────────────────────
pbp_df = _concat("pbp_flat")
pbp_df = pbp_df.drop_duplicates(subset=["id"])
pbp_df = _attach_date(pbp_df, "gameId")

# ── shots — dedupe on id ──────────────────────────────────────────────────────
shots_df = _concat("shots")
shots_df = shots_df.drop_duplicates(subset=["id"])
shots_df = _attach_date(shots_df, "gameId")

# ── players — dedupe on (gameId, athleteId) ───────────────────────────────────
players_df = _concat("players")
players_df = players_df.drop_duplicates(subset=["gameId", "athleteId"])
players_df = _attach_date(players_df, "gameId")

# ── lineup_stints — dedupe on (gameId, home_lineup_key, away_lineup_key, start_seconds)
lineups_df = _concat("lineup_stints")
lineups_df = lineups_df.drop_duplicates(
    subset=["gameId", "home_lineup_key", "away_lineup_key", "start_seconds"]
)
lineups_df = _attach_date(lineups_df, "gameId")


if __name__ == "__main__":
    for name, df in [
        ("games",                games_df),
        ("four_factors",         ff_df),
        ("possessions_enriched", poss_enriched_df),
        ("possessions",          poss_df),
        ("plays",                plays_df),
        ("pbp_flat",             pbp_df),
        ("shots",                shots_df),
        ("players",              players_df),
        ("lineup_stints",        lineups_df),
    ]:
        nulls = df["game_date"].isna().sum() if "game_date" in df.columns else "n/a"
        print(f"{name:25s}  rows={len(df):>7,}  date_nulls={nulls}")
