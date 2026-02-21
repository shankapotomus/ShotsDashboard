"""
consolidate_csvs.py
-------------------
One-time migration script to:
  1. Load all existing flat batch CSV files from cbbd_data/, dedupe across overlaps
  2. Split each data type by actual game date (from games.startDate)
  3. Write daily files into cbbd_data/{type}/YYYYMMDD_SEASON.csv
  4. Delete the old flat batch files

Run from the repo root:
    python cbbd_data/consolidate_csvs.py

After this runs, cbbd_data/ will have the structure:
    cbbd_data/
      games/              20251103_2026.csv  20251104_2026.csv  ...
      plays/              20251103_2026.csv  ...
      possessions/        ...
      possessions_enriched/ ...
      shots/              ...
      lineup_stints/      ...
      players/            ...
      pbp_flat/           ...
      four_factors/       ...

This matches what daily_fetch.py produces going forward.
"""

import glob
import os

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SEASON = 2026

# ---------------------------------------------------------------------------
# Dedup keys per CSV type
# ---------------------------------------------------------------------------
DEDUP_KEYS = {
    "games":                ["id"],
    "plays":                ["id"],
    "possessions":          ["play_id", "gameId", "possession_team"],
    "possessions_enriched": ["gameId", "possession_id", "possession_team"],
    "shots":                ["id"],
    "lineup_stints":        ["gameId", "home_lineup_key", "away_lineup_key", "start_seconds"],
    "players":              ["gameId", "athleteId"],
    "pbp_flat":             ["id"],
    "four_factors":         ["game_id", "team"],
}

# Column used to look up game date per row
GAME_ID_COL = {
    "games":                "id",
    "plays":                "gameId",
    "possessions":          "gameId",
    "possessions_enriched": "gameId",
    "shots":                "gameId",
    "lineup_stints":        "gameId",
    "players":              "gameId",
    "pbp_flat":             "gameId",
    "four_factors":         "game_id",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_flat_files(csv_type: str) -> list[str]:
    """Find all flat (non-subdir) CSV files for this type in DATA_DIR."""
    pattern = os.path.join(DATA_DIR, f"{csv_type}_*.csv")
    paths = sorted(glob.glob(pattern))
    # possessions raw: exclude enriched files that match the same glob
    if csv_type == "possessions":
        paths = [p for p in paths if "possessions_enriched" not in os.path.basename(p)]
    return paths


def load_all(csv_type: str) -> pd.DataFrame:
    """Load and dedup all flat batch/consolidated files for a given type."""
    paths = _find_flat_files(csv_type)
    if not paths:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    dedup_keys = [k for k in DEDUP_KEYS[csv_type] if k in df.columns]
    if dedup_keys:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_keys)
        print(f"  [{csv_type}] {len(paths)} file(s), {before:,} -> {len(df):,} rows after dedup")
    else:
        print(f"  [{csv_type}] {len(paths)} file(s), {len(df):,} rows (no dedup keys found)")

    return df


def build_date_lookup(games_df: pd.DataFrame) -> dict:
    """Return dict of str(game_id) -> 'YYYYMMDD'."""
    lookup = {}
    for _, row in games_df.iterrows():
        date_str = str(row["startDate"])[:10].replace("-", "")
        lookup[str(row["id"])] = date_str
    return lookup


def split_and_write(csv_type: str, df: pd.DataFrame, date_lookup: dict):
    """Split df by game date and write one CSV per day into the type subdir."""
    gid_col = GAME_ID_COL[csv_type]
    if gid_col not in df.columns:
        print(f"  [{csv_type}] WARNING: no '{gid_col}' column, skipping")
        return 0

    df = df.copy()
    df["_date"] = df[gid_col].astype(str).map(date_lookup)

    missing = df["_date"].isna().sum()
    if missing:
        print(f"  [{csv_type}] WARNING: {missing} rows have no matching game date, dropping")
    df = df.dropna(subset=["_date"])

    subdir = os.path.join(DATA_DIR, csv_type)
    os.makedirs(subdir, exist_ok=True)

    files_written = 0
    for date, group in df.groupby("_date"):
        group = group.drop(columns=["_date"])
        fname = os.path.join(subdir, f"{date}_{SEASON}.csv")
        group.to_csv(fname, index=False)
        files_written += 1

    print(f"  [{csv_type}] Written {files_written} daily file(s) -> cbbd_data/{csv_type}/")
    return files_written


def delete_flat_files(csv_type: str):
    """Delete all flat files for this type from DATA_DIR (not subdirs)."""
    paths = _find_flat_files(csv_type)
    deleted = []
    for p in paths:
        os.remove(p)
        deleted.append(os.path.basename(p))
    if deleted:
        print(f"  [{csv_type}] Deleted {len(deleted)} old flat file(s): {', '.join(deleted)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Working directory: {DATA_DIR}\n")

    # Games first — needed to build the date lookup
    print("=== games ===")
    games_df = load_all("games")
    if games_df.empty:
        print("ERROR: no games files found — cannot build date lookup. Aborting.")
        return

    date_lookup = build_date_lookup(games_df)
    print(f"  Date lookup: {len(date_lookup)} games across {len(set(date_lookup.values()))} dates")
    split_and_write("games", games_df, date_lookup)
    delete_flat_files("games")
    print()

    # All other types
    for csv_type in DEDUP_KEYS:
        if csv_type == "games":
            continue
        print(f"=== {csv_type} ===")
        df = load_all(csv_type)
        if df.empty:
            print(f"  [{csv_type}] No flat files found, skipping\n")
            continue
        split_and_write(csv_type, df, date_lookup)
        delete_flat_files(csv_type)
        print()

    print("Migration complete.")
    print("cbbd_data/ now has one subdir per type, one file per game date.")
    print("daily_fetch.py will continue writing in this format going forward.")


if __name__ == "__main__":
    main()
