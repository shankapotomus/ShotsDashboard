"""
consolidate_csvs.py
-------------------
One-time script to:
  1. Load all existing batch CSV files, dedupe across overlapping batches
  2. Split each data type into one file per game date
     (matching the format daily_fetch.py uses going forward)
  3. Delete the old batch files

Run from the repo root:
    python cbbd_data/consolidate_csvs.py

After this runs, cbbd_data/ will contain one file per type per day
(e.g. possessions_enriched_20251103_2026.csv), just like daily_fetch.py
produces. load_data.py dedupes automatically so any future overlaps
between new daily files are handled safely.
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

# Column used to join each type to games for date lookup
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

def load_all(csv_type: str) -> pd.DataFrame:
    """Load and dedup all batch files for a given csv_type."""
    pattern = os.path.join(DATA_DIR, f"{csv_type}_*.csv")
    paths = sorted(glob.glob(pattern))

    # possessions raw: exclude enriched files that match the same glob
    if csv_type == "possessions":
        paths = [p for p in paths if "possessions_enriched" not in os.path.basename(p)]

    if not paths:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

    dedup_keys = DEDUP_KEYS[csv_type]
    existing_keys = [k for k in dedup_keys if k in df.columns]
    if existing_keys:
        before = len(df)
        df = df.drop_duplicates(subset=existing_keys)
        print(f"  [{csv_type}] {len(paths)} files, {before:,} -> {len(df):,} rows after dedup")
    else:
        print(f"  [{csv_type}] {len(paths)} files, {len(df):,} rows (no dedup keys found)")

    return df


def build_game_date_lookup(games_df: pd.DataFrame) -> dict:
    """Return dict of str(game_id) -> 'YYYYMMDD' from the games DataFrame."""
    lookup = {}
    for _, row in games_df.iterrows():
        date_str = str(row["startDate"])[:10].replace("-", "")  # '20251103'
        lookup[str(row["id"])] = date_str
    return lookup


def split_and_write(csv_type: str, df: pd.DataFrame, date_lookup: dict):
    """Split df by game date and write one CSV per day."""
    gid_col = GAME_ID_COL[csv_type]

    if gid_col not in df.columns:
        print(f"  [{csv_type}] WARNING: no '{gid_col}' column, skipping split")
        return 0

    df = df.copy()
    df["_date"] = df[gid_col].astype(str).map(date_lookup)

    missing = df["_date"].isna().sum()
    if missing:
        print(f"  [{csv_type}] WARNING: {missing} rows have no matching game date, will be dropped")

    df = df.dropna(subset=["_date"])

    files_written = 0
    for date, group in df.groupby("_date"):
        group = group.drop(columns=["_date"])
        fname = os.path.join(DATA_DIR, f"{csv_type}_{date}_{SEASON}.csv")
        group.to_csv(fname, index=False)
        files_written += 1

    print(f"  [{csv_type}] Written {files_written} daily files")
    return files_written


def delete_batch_files(csv_type: str, daily_file_dates: set):
    """Delete files whose name contains two 8-digit date blocks (old batch files)."""
    pattern = os.path.join(DATA_DIR, f"{csv_type}_*.csv")
    paths = sorted(glob.glob(pattern))
    if csv_type == "possessions":
        paths = [p for p in paths if "possessions_enriched" not in os.path.basename(p)]

    deleted = []
    for p in paths:
        name = os.path.basename(p)
        parts = [s for s in name.replace(".csv", "").split("_") if s.isdigit() and len(s) == 8]
        if len(parts) >= 2:  # two date blocks = batch file
            os.remove(p)
            deleted.append(name)

    if deleted:
        print(f"  [{csv_type}] Deleted {len(deleted)} batch file(s): {', '.join(deleted)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Working directory: {DATA_DIR}\n")

    # Step 1: load games first to build the date lookup
    print("=== Loading games ===")
    games_df = load_all("games")
    if games_df.empty:
        print("ERROR: no games files found — cannot build date lookup. Aborting.")
        return

    date_lookup = build_game_date_lookup(games_df)
    print(f"  Date lookup built: {len(date_lookup)} games across {len(set(date_lookup.values()))} dates\n")

    # Step 2: write games daily files and delete batch
    print("=== Splitting games ===")
    split_and_write("games", games_df, date_lookup)
    delete_batch_files("games", set())
    print()

    # Step 3: process all other types
    for csv_type in DEDUP_KEYS:
        if csv_type == "games":
            continue
        print(f"=== {csv_type} ===")
        df = load_all(csv_type)
        if df.empty:
            print(f"  [{csv_type}] No data, skipping\n")
            continue
        split_and_write(csv_type, df, date_lookup)
        delete_batch_files(csv_type, set())
        print()

    print("Done.")
    print("cbbd_data/ now has one file per type per game date.")
    print("daily_fetch.py will continue appending in the same format going forward.")


if __name__ == "__main__":
    main()
