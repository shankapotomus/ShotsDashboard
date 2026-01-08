# ShotsDashboard

College Basketball Analytics Pipeline using the CollegeBasketballData.com API.

## Overview

A data pipeline that collects play-by-play data from the CBBD API, transforms it into meaningful analytics, stores it in DuckDB, and visualizes it with Streamlit.

```
API (Raw Data) → Transform (DuckDB Queries) → Store (DuckDB) → Visualize (Streamlit)
```

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW API DATA                                │
├─────────────────────────────────────────────────────────────────────┤
│  get_teams()     get_team_roster()     get_games()     get_plays()  │
│       │                 │                   │               │       │
│       ▼                 ▼                   ▼               ▼       │
│   TeamInfo         TeamRoster           GameInfo        PlayInfo    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TRANSFORMATION LAYER                           │
├─────────────────────────────────────────────────────────────────────┤
│  • Derive lineups from on_floor[]                                   │
│  • Identify starting lineups (first play of each game)              │
│  • Aggregate box scores from plays                                  │
│  • Calculate shooting stats from shot_info                          │
│  • Compute offensive/defensive ratings                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DUCKDB STORAGE                              │
├─────────────────────────────────────────────────────────────────────┤
│  Dimension Tables          │  Fact Tables                           │
│  ─────────────────         │  ───────────                           │
│  dim_teams                 │  fact_plays                            │
│  dim_players               │  fact_games                            │
│  dim_lineups (derived)     │  fact_box_scores (derived)             │
│                            │  fact_starting_lineups (derived)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STREAMLIT DASHBOARD                            │
├─────────────────────────────────────────────────────────────────────┤
│  • Shot charts                                                      │
│  • Lineup analysis                                                  │
│  • Player/team performance                                          │
│  • Game flow visualization                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Sources

### Raw API Endpoints (4 total)

| Endpoint | Model | Purpose |
|----------|-------|---------|
| `get_teams()` | TeamInfo | Team reference data (id, name, conference, venue) |
| `get_team_roster(season)` | TeamRoster | Player info (id, name, height, position) |
| `get_games(season)` | GameInfo | Game metadata (date, teams, scores, venue) |
| `get_plays_by_*()` | PlayInfo | Play-by-play - the source of truth |

### PlayInfo Structure (Core Data Model)

```
PlayInfo
├── id, game_id, season
├── period, clock, seconds_remaining
├── play_type, play_text
├── scoring_play, shooting_play, score_value
├── team_id, team, opponent_id, opponent
├── home_score, away_score, home_win_probability
│
├── participants[]
│   ├── id (player_id)
│   └── name
│
├── on_floor[] ◄─── Links plays to lineups (5 players per team)
│   ├── id (player_id)
│   ├── name
│   └── team
│
└── shot_info ◄──── Shot chart data
    ├── made (bool)
    ├── range
    ├── assisted, assisted_by
    ├── shooter {id, name}
    └── location {x, y} ◄─── Court coordinates
```

### Data Availability by Season

| Data Type | Available Since | Notes |
|-----------|-----------------|-------|
| Play-by-play | 2005-06 | Core data |
| Substitutions/Lineups | 2023-24 | `on_floor[]` in PlayInfo |
| Shot distribution | 2013-14 | `shot_info.location` |
| Box scores/season stats | 2002-03 | Can derive from PBP |

**Current Focus:** 2025-26 season (full data available)

## Key Insight: Derive Everything from Play-by-Play

With detailed PBP data, we can derive:

| Derived Data | How |
|--------------|-----|
| Box scores | Aggregate plays by player + game |
| Shooting stats | Filter `shooting_play=True`, calc makes/attempts |
| Lineup combinations | Extract unique `on_floor[]` groupings |
| Starting lineups | First play of each game → `on_floor[]` |
| Offensive/defensive rating | Points per possession from plays |
| Assist networks | `shot_info.assisted_by` relationships |

## Prerequisites

- Python 3.7+
- API key from [CollegeBasketballData.com](https://collegebasketballdata.com)
- Required packages:
  ```bash
  pip install cbbd pandas duckdb streamlit
  ```

## Project Status

🚧 **Planning Phase** - Defining data architecture and pipeline design.

## API Reference

- [CBBD Python SDK](https://github.com/CFBD/cbbd-python)
- [API Documentation](https://api.collegebasketballdata.com/docs)
