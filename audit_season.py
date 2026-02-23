#!/usr/bin/env python3
"""
audit_season.py
---------------
Fetches every game ID for the current season from CBBD, then checks which
ones are missing from your local cbbd_data/ CSVs.

Prints:
  - Total games the API knows about (completed only)
  - How many you have locally
  - Which dates are missing / incomplete
  - A backfill command for each missing date

Usage:
    python audit_season.py
    python audit_season.py --season 2026       # explicit season
    python audit_season.py --through 2026-02-20  # only audit up to a date
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")
except ImportError:
    _EASTERN = None

import cbbd
import pandas as pd
from cbbd.rest import ApiException

SEASON      = 2026
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbbd_data")


def load_env():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def get_configuration():
    load_env()
    api_key = os.environ.get("CBBD_API_KEY", "").strip()
    if not api_key:
        print("ERROR: CBBD_API_KEY not set.")
        sys.exit(1)
    return cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=api_key,
    )


def fetch_all_season_games(configuration, season):
    """Pull every game for the season from the API in monthly chunks.

    The API has a ~3000 record cap per request, so fetching the full season
    in one call silently truncates results. Chunking by month avoids this.
    """
    from datetime import timedelta

    print(f"Fetching all games for season {season} from CBBD API (monthly chunks)...")

    # Season runs Nov of prior year through April of season year
    season_start = datetime(season - 1, 11, 1)
    season_end   = datetime(season, 4, 30, 23, 59, 59)

    all_games = []
    seen_ids  = set()

    current = season_start
    while current <= season_end:
        # Last day of current month
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1)
        else:
            next_month = datetime(current.year, current.month + 1, 1)
        month_end = next_month - timedelta(seconds=1)
        chunk_end = min(month_end, season_end)

        with cbbd.ApiClient(configuration) as api_client:
            games_api = cbbd.GamesApi(api_client)
            games = games_api.get_games(
                season=season,
                start_date_range=current,
                end_date_range=chunk_end,
            )

        new = 0
        for g in games:
            d = g.to_dict()
            if str(d["id"]) not in seen_ids:
                seen_ids.add(str(d["id"]))
                all_games.append(d)
                new += 1

        print(f"  {current.strftime('%Y-%m')}: {new} games fetched")
        current = next_month

    games_df = pd.DataFrame(all_games)
    print(f"  Total unique games fetched: {len(games_df)}")
    return games_df


def load_local_game_ids(output_dir, season):
    """Read all local games CSVs and return a set of game IDs we already have,
    plus a dict of date_str -> set(game_ids) for per-date reporting."""
    games_dir = os.path.join(output_dir, "games")
    local_by_date = defaultdict(set)

    if not os.path.isdir(games_dir):
        print(f"  WARNING: {games_dir} does not exist — no local data found.")
        return set(), local_by_date

    for fname in sorted(os.listdir(games_dir)):
        if not fname.endswith(f"_{season}.csv"):
            continue
        date_str = fname.replace(f"_{season}.csv", "")  # e.g. "20260220"
        fpath = os.path.join(games_dir, fname)
        try:
            df = pd.read_csv(fpath, usecols=["id"])
            ids = set(df["id"].astype(str).tolist())
            local_by_date[date_str] = ids
        except Exception as e:
            print(f"  WARNING: could not read {fname}: {e}")

    all_local_ids = set()
    for ids in local_by_date.values():
        all_local_ids |= ids

    return all_local_ids, local_by_date


def also_check_plays(output_dir, season, game_ids):
    """Cross-check that we also have play-by-play data for each game ID."""
    plays_dir = os.path.join(output_dir, "plays")
    local_play_game_ids = set()

    if not os.path.isdir(plays_dir):
        return local_play_game_ids

    for fname in sorted(os.listdir(plays_dir)):
        if not fname.endswith(f"_{season}.csv"):
            continue
        fpath = os.path.join(plays_dir, fname)
        try:
            df = pd.read_csv(fpath, usecols=["gameId"])
            local_play_game_ids |= set(df["gameId"].astype(str).tolist())
        except Exception as e:
            print(f"  WARNING: could not read plays/{fname}: {e}")

    return local_play_game_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument(
        "--through",
        help="Only audit games on or before this date (YYYY-MM-DD). "
             "Defaults to yesterday ET.",
        default=None,
    )
    args = parser.parse_args()

    # Determine cutoff date
    if args.through:
        cutoff = datetime.strptime(args.through, "%Y-%m-%d")
    else:
        if _EASTERN:
            from datetime import timedelta
            cutoff = (datetime.now(tz=_EASTERN) - timedelta(days=1)).replace(
                hour=23, minute=59, second=59, tzinfo=None
            )
        else:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=1)

    print(f"Auditing season {args.season} through {cutoff.date()}\n")

    configuration = get_configuration()

    # 1. Fetch all games from API
    all_games_df = fetch_all_season_games(configuration, args.season)

    # 2. Parse game dates and filter to completed games on or before cutoff
    all_games_df["start_dt"] = pd.to_datetime(
        all_games_df["startDate"], utc=True, errors="coerce"
    ).dt.tz_localize(None)

    # Only games that have already been played (start_date <= cutoff)
    completed = all_games_df[all_games_df["start_dt"] <= pd.Timestamp(cutoff)].copy()

    # Derive a local calendar date string (YYYYMMDD) from UTC start time
    # Use ET offset: subtract 5h (EST) — close enough for audit grouping
    from datetime import timedelta
    completed["et_date"] = (
        completed["start_dt"] - timedelta(hours=5)
    ).dt.strftime("%Y%m%d")

    print(f"  Completed games (on or before {cutoff.date()}): {len(completed)}\n")

    # 3. Load local game IDs
    local_ids, local_by_date = load_local_game_ids(OUTPUT_DIR, args.season)
    print(f"  Local game IDs found: {len(local_ids)}\n")

    # 4. Also check plays coverage
    local_play_ids = also_check_plays(OUTPUT_DIR, args.season, local_ids)

    # 5. Find missing games
    api_ids = set(completed["id"].astype(str).tolist())
    missing_ids = api_ids - local_ids
    missing_plays_only = local_ids - local_play_ids  # have games CSV but no plays

    # Group missing by ET date for backfill commands
    missing_by_date = defaultdict(list)
    for _, row in completed[completed["id"].astype(str).isin(missing_ids)].iterrows():
        missing_by_date[row["et_date"]].append(str(row["id"]))

    # 6. Build missing games dataframe and write to CSV
    missing_rows = []
    for _, row in completed[completed["id"].astype(str).isin(missing_ids)].iterrows():
        missing_rows.append({
            "game_id":      str(row["id"]),
            "et_date":      row["et_date"],
            "date_fmt":     datetime.strptime(row["et_date"], "%Y%m%d").strftime("%Y-%m-%d"),
            "away_team":    row.get("awayTeam", "?"),
            "home_team":    row.get("homeTeam", "?"),
            "status":       row.get("status", "?"),
            "missing_type": "games_and_plays",
        })

    for gid in sorted(missing_plays_only & api_ids):
        row = completed[completed["id"].astype(str) == gid]
        if not row.empty:
            r = row.iloc[0]
            missing_rows.append({
                "game_id":      gid,
                "et_date":      r["et_date"],
                "date_fmt":     datetime.strptime(r["et_date"], "%Y%m%d").strftime("%Y-%m-%d"),
                "away_team":    r.get("awayTeam", "?"),
                "home_team":    r.get("homeTeam", "?"),
                "status":       r.get("status", "?"),
                "missing_type": "plays_only",
            })

    missing_df = pd.DataFrame(missing_rows).sort_values(["et_date", "game_id"]).reset_index(drop=True)

    # Unique dates that need backfilling (games_and_plays only)
    backfill_dates = sorted(
        missing_df[missing_df["missing_type"] == "games_and_plays"]["date_fmt"].unique()
    )
    backfill_df = pd.DataFrame({
        "date":            backfill_dates,
        "backfill_command": [f"python daily_fetch.py --date {d}" for d in backfill_dates],
    })

    # Write CSVs — always next to this script, regardless of cwd
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    missing_csv  = os.path.join(script_dir, "audit_missing_games.csv")
    backfill_csv = os.path.join(script_dir, "audit_backfill_dates.csv")
    missing_df.to_csv(missing_csv, index=False)
    backfill_df.to_csv(backfill_csv, index=False)

    # 7. Print summary
    print("=" * 60)
    print(f"SEASON AUDIT — {args.season}")
    print("=" * 60)
    print(f"  API total (completed)  : {len(api_ids)}")
    print(f"  Local (games CSV)      : {len(local_ids & api_ids)}")
    print(f"  Local (plays CSV)      : {len(local_play_ids & api_ids)}")
    print(f"  Missing games          : {len(missing_ids)}")
    print(f"  Games w/o plays data   : {len(missing_plays_only & api_ids)}")
    print(f"  Dates needing backfill : {len(backfill_dates)}")
    print()
    print(f"  📄 Missing games detail  -> {missing_csv}")
    print(f"  📄 Backfill date list    -> {backfill_csv}")
    print()

    if not missing_ids and not (missing_plays_only & api_ids):
        print("✅ All completed games accounted for — nothing to backfill!")
    else:
        print(f"❌ {len(missing_ids)} missing games across {len(backfill_dates)} dates.")
        print("   Run the backfill commands in audit_backfill_dates.csv to catch up.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
