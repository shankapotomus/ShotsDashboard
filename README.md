# ShotsDashboard

College Basketball Analytics Pipeline using the CollegeBasketballData.com API.

## Overview

A data pipeline that collects play-by-play data from the CBBD API, transforms it into meaningful analytics, stores it in DuckDB, and visualizes it with Streamlit.

```
API (Raw Data) → Transform (DuckDB Queries) → Store (DuckDB) → Visualize (Streamlit)
```

**Current Focus:** 2025-26 season

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
│  (365 teams)     (~15 players)        (~30 games)    (~7000 plays)  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TRANSFORMATION LAYER                           │
├─────────────────────────────────────────────────────────────────────┤
│  • Derive lineups from substitution plays (see Key Findings)        │
│  • Identify starting lineups from first plays                       │
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

---

## Milestone 1: Raw Data Layer (VALIDATED)

### Endpoints Tested

| Endpoint | Model | Records | Status |
|----------|-------|---------|--------|
| `TeamsApi.get_teams(season=2026)` | TeamInfo | 365 teams | ✓ |
| `TeamsApi.get_team_roster(season, team)` | TeamRoster | ~15 players/team | ✓ |
| `GamesApi.get_games(season, team)` | GameInfo | ~30 games/team | ✓ |
| `PlaysApi.get_plays_by_team(season, team)` | PlayInfo | ~7000 plays/team | ✓ |

### Validated Data Models

#### TeamInfo (15 columns)
```
id, sourceId, school, mascot, abbreviation, displayName, shortDisplayName,
primaryColor, secondaryColor, conference, conferenceId,
currentVenueId, currentVenue, currentCity, currentState
```

#### TeamRoster.players (12 columns)
```
id, sourceId, name, firstName, lastName, jersey, position,
height, weight, dateOfBirth, startSeason, endSeason,
hometown: {city, state, country, latitude, longitude, countyFips}  ← NESTED
```

#### GameInfo (31 columns)
```
id, sourceId, season, seasonLabel, seasonType, startDate, startTimeTbd,
neutralSite, conferenceGame, gameType, status, attendance, excitement,
homeTeamId, homeTeam, homeConferenceId, homeConference,
homePoints, homePeriodPoints[], homeWinner,                          ← NESTED ARRAY
awayTeamId, awayTeam, awayConferenceId, awayConference,
awayPoints, awayPeriodPoints[], awayWinner,                          ← NESTED ARRAY
venueId, venue, city, state
```

#### PlayInfo (29 columns)
```
id, sourceId, gameId, gameSourceId, gameStartDate,
season, seasonType, gameType,
playType, period, clock, secondsRemaining, wallclock,
homeScore, awayScore, homeWinProbability,
scoringPlay, shootingPlay, scoreValue, playText,
isHomeTeam, teamId, team, conference,
opponentId, opponent, opponentConference,

participants: [{name, id}]                                           ← NESTED ARRAY

shotInfo: {                                                          ← NESTED OBJECT
  shooter: {name, id},
  made: bool,
  range: "three_pointer" | "two_pointer" | ...,
  assisted: bool,
  assistedBy: {name, id},
  location: {x, y}                                                   ← SHOT COORDINATES
}
```

---

## Key Findings

### `onFloor[]` NOT in API Response

The SDK documentation shows `on_floor[]` as a field in PlayInfo, but **it is not returned by the API**.

| Expected | Reality |
|----------|---------|
| `onFloor[]` with 10 players per play | **NOT PRESENT** in actual response |

### Lineup Derivation Strategy

Since `onFloor[]` is missing, we derive lineups from substitution plays:

```
playType: "Substitution"
playText: "Patrick Ngongba II subbing out for Duke"
playText: "Maliq Brown subbing in for Duke"
participants: [{name: "...", id: 216}]
```

**Algorithm:**
```python
lineup = set(starters)  # Need to identify from first plays
for play in plays_ordered_by_time:
    if "subbing out" in play.playText:
        lineup.remove(play.participants[0]['id'])
    elif "subbing in" in play.playText:
        lineup.add(play.participants[0]['id'])
    play.current_lineup = lineup.copy()
```

### Shot Coordinates Confirmed

`shotInfo.location` contains court coordinates for shot charts:

```python
shotInfo: {
    'shooter': {'name': 'Isaiah Evans', 'id': 211},
    'made': False,
    'range': 'three_pointer',
    'assisted': False,
    'location': {'x': 263.2, 'y': 415}  # ← COURT COORDINATES
}
```

---

## Data Availability by Season

| Data Type | Available Since | Notes |
|-----------|-----------------|-------|
| Play-by-play | 2005-06 | Core data |
| Substitutions/Lineups | 2023-24 | Via substitution plays |
| Shot coordinates | 2013-14 | `shotInfo.location` |
| Box scores/season stats | 2002-03 | Can derive from PBP |

---

## Raw Tables for DuckDB

| Table | Source | Est. Rows (full season) |
|-------|--------|-------------------------|
| `raw_teams` | get_teams() | 365 |
| `raw_players` | get_team_roster() | ~5,500 |
| `raw_games` | get_games() | ~5,500 |
| `raw_plays` | get_plays_by_team() | ~2.5M |

---

## Derived Analytics (Milestone 2)

With play-by-play data, we can derive:

| Derived Data | Method |
|--------------|--------|
| Box scores | Aggregate plays by player + game |
| Shooting stats | Filter `shootingPlay=True`, calc makes/attempts |
| Lineup combinations | Track substitutions to build lineup state |
| Starting lineups | First 5 players on court at period 1 start |
| Offensive/defensive rating | Points per possession |
| Assist networks | `shotInfo.assistedBy` relationships |

---

## Prerequisites

- Python 3.7+
- API key from [CollegeBasketballData.com](https://collegebasketballdata.com)
- Required packages:
  ```bash
  pip install cbbd pandas duckdb streamlit
  ```

## Project Status

- [x] **Milestone 1:** Raw Data Layer - Validated
- [ ] **Milestone 2:** Data Transformations & Metrics
- [ ] **Milestone 3:** DuckDB Schema Design
- [ ] **Milestone 4:** Streamlit Dashboard
- [ ] **Milestone 5:** Build & Test Pipeline

## Files

| File | Purpose |
|------|---------|
| `test_api_endpoints.ipynb` | Validate raw API endpoints |
| `test_api_endpoints.py` | CLI version of endpoint tests |
| `cbbd_playbyplay_2025_26.ipynb` | Original data collection notebook |

## API Reference

- [CBBD Python SDK](https://github.com/CFBD/cbbd-python)
- [API Documentation](https://api.collegebasketballdata.com/docs)
