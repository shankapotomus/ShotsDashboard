#!/usr/bin/env bash
# run_daily_fetch.sh
# ------------------
# Shell wrapper for daily_fetch.py, intended to be called by cron.
#
# Cron setup (run at 6 AM every day):
#   crontab -e
#   0 6 * * * /path/to/ShotsDashboard/run_daily_fetch.sh
#
# The script logs output to logs/daily_fetch_YYYYMMDD.log.

set -euo pipefail

# --- Resolve paths -----------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_fetch_$(date +%Y%m%d).log"

# --- Environment -------------------------------------------------------------
# Load API key from a .env file if present (key=value format, no export needed)
ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${CBBD_API_KEY:-}" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S')  ERROR  CBBD_API_KEY is not set." | tee -a "$LOG_FILE"
    exit 1
fi

# --- Optional: activate a virtual environment --------------------------------
# Uncomment and adjust the path below if you use a venv:
# source "$SCRIPT_DIR/.venv/bin/activate"

# --- Run ---------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S')  INFO   Starting daily fetch" | tee -a "$LOG_FILE"

python "$SCRIPT_DIR/daily_fetch.py" "$@" 2>&1 | tee -a "$LOG_FILE"

echo "$(date '+%Y-%m-%d %H:%M:%S')  INFO   Finished daily fetch" | tee -a "$LOG_FILE"
