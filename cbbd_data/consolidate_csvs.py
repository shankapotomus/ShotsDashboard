"""
consolidate_csvs.py
-------------------
One-time script to consolidate overlapping batch CSVs into a single
season file per data type, then delete the old batch files.

Run from the repo root:
    python cbbd_data/consolidate_csvs.py

After this runs, cbbd_data/ will contain one consolidated file per type
covering the full season so far (e.g. possessions_enriched_20251103_20260218_2026.csv),
plus any daily files already written by daily_fetch.py.  Going forward,
daily_fetch.py writes one file per day and load_data.py dedupes automatically.
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
    "games":                  ["id"],
    "plays":                  ["id"],
    "possessions":            ["play_id", "gameId", "possession_team"],
    "possessions_enriched":   ["gameId", "possession_id", "possession_team"],
    "shots":                  ["id"],
    "lineup_stints":          ["gameId", "home_lineup_key", "away_lineup_key", "start_seconds"],
    "players":                ["gameId", "athleteId"],
    "pbp_flat":               ["id"],
    "four_factors":           ["game_id", "team"],
}

# ---------------------------------------------------------------------------
# Batch file patterns to consolidate (exclude any already-daily files)
# A "batch" file has two dates in the name: name_YYYYMMDD_YYYYMMDD_SEASON.csv
# A "daily" file has one date:             name_YYYYMMDD_SEASON.csv
# ---------------------------------------------------------------------------

def _is_batch(path: str) -> bool:
    """Return True if filename contains two 8-digit date blocks (batch file)."""
    name = os.path.basename(path)
    parts = name.split("_")
    date_parts = [p for p in parts if p.isdigit() and len(p) == 8]
    return len(date_parts) >= 2


def load_and_consolidate(csv_type: str, dedup_keys: list) -> pd.DataFrame | None:
    pattern = os.path.join(DATA_DIR, f"{csv_type}_*.csv")
    all_paths = sorted(glob.glob(pattern))

    # For possessions (raw), skip enriched files that also match
    if csv_type == "possessions":
        all_paths = [p for p in all_paths if "possessions_enriched" not in os.path.basename(p)]

    if not all_paths:
        print(f"  [{csv_type}] No files found, skipping.")
        return None

    frames = [pd.read_csv(p) for p in all_paths]
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)

    # Only dedup on keys that actually exist in the data
    existing_keys = [k for k in dedup_keys if k in combined.columns]
    if existing_keys:
        combined = combined.drop_duplicates(subset=existing_keys)

    after = len(combined)
    print(f"  [{csv_type}] {len(all_paths)} files, {before:,} rows -> {after:,} after dedup ({before - after:,} dupes removed)")
    return combined


def date_range_from_files(csv_type: str) -> tuple[str, str]:
    """Extract earliest start date and latest end date across all batch files."""
    pattern = os.path.join(DATA_DIR, f"{csv_type}_*.csv")
    all_paths = sorted(glob.glob(pattern))
    if csv_type == "possessions":
        all_paths = [p for p in all_paths if "possessions_enriched" not in os.path.basename(p)]

    all_dates = []
    for p in all_paths:
        name = os.path.basename(p)
        parts = name.split("_")
        dates = [p for p in parts if p.isdigit() and len(p) == 8]
        all_dates.extend(dates)

    if not all_dates:
        return "00000000", "99999999"
    return min(all_dates), max(all_dates)


def delete_old_files(csv_type: str, keep_path: str):
    """Delete all files for this type except the newly written consolidated one."""
    pattern = os.path.join(DATA_DIR, f"{csv_type}_*.csv")
    all_paths = sorted(glob.glob(pattern))
    if csv_type == "possessions":
        all_paths = [p for p in all_paths if "possessions_enriched" not in os.path.basename(p)]

    deleted = []
    for p in all_paths:
        if os.path.abspath(p) != os.path.abspath(keep_path):
            os.remove(p)
            deleted.append(os.path.basename(p))

    if deleted:
        print(f"  [{csv_type}] Deleted {len(deleted)} old file(s): {', '.join(deleted)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Consolidating CSVs in: {DATA_DIR}\n")

    for csv_type, dedup_keys in DEDUP_KEYS.items():
        start_date, end_date = date_range_from_files(csv_type)
        out_name = f"{csv_type}_{start_date}_{end_date}_{SEASON}.csv"
        out_path = os.path.join(DATA_DIR, out_name)

        df = load_and_consolidate(csv_type, dedup_keys)
        if df is None or df.empty:
            continue

        df.to_csv(out_path, index=False)
        print(f"  [{csv_type}] Written -> {out_name}")

        delete_old_files(csv_type, out_path)
        print()

    print("Done. cbbd_data/ now has one consolidated file per type.")
    print("Going forward, daily_fetch.py will append daily files (YYYYMMDD naming).")
    print("load_data.py dedupes automatically so overlaps won't cause issues.")


if __name__ == "__main__":
    main()
