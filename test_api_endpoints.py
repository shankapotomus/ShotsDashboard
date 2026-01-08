"""
Test script for CBBD API endpoints
Validates the 4 raw data sources for the ShotsDashboard pipeline

Usage:
    python test_api_endpoints.py

Requires:
    - CBBD_API_KEY environment variable, or
    - Will prompt for API key
"""

import os
import sys

# Check for required packages
try:
    import cbbd
    import pandas as pd
    from cbbd.rest import ApiException
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install cbbd pandas")
    sys.exit(1)


def get_api_key():
    """Get API key from environment or prompt user."""
    api_key = os.environ.get('CBBD_API_KEY')
    if not api_key:
        import getpass
        print("CBBD_API_KEY not found in environment.")
        api_key = getpass.getpass("Enter your API key: ")
    return api_key


def create_client(api_key):
    """Create configured API client."""
    configuration = cbbd.Configuration(
        host="https://api.collegebasketballdata.com",
        access_token=api_key
    )
    return configuration


def test_get_teams(configuration, season=2026):
    """Test 1: get_teams() - Team reference data."""
    print("\n" + "=" * 60)
    print("TEST 1: TeamsApi.get_teams()")
    print("=" * 60)

    with cbbd.ApiClient(configuration) as api_client:
        teams_api = cbbd.TeamsApi(api_client)

        try:
            teams = teams_api.get_teams(season=season)
            df = pd.DataFrame([t.to_dict() for t in teams])

            print(f"✓ SUCCESS: Retrieved {len(df)} teams")
            print(f"\nColumns ({len(df.columns)}):")
            print(f"  {list(df.columns)}")
            print(f"\nSample (first 3):")
            print(df[['id', 'school', 'conference', 'mascot']].head(3).to_string(index=False))

            return True, df

        except ApiException as e:
            print(f"✗ FAILED: {e}")
            return False, None


def test_get_roster(configuration, season=2026, team="Duke"):
    """Test 2: get_team_roster() - Player reference data."""
    print("\n" + "=" * 60)
    print(f"TEST 2: TeamsApi.get_team_roster(season={season}, team='{team}')")
    print("=" * 60)

    with cbbd.ApiClient(configuration) as api_client:
        teams_api = cbbd.TeamsApi(api_client)

        try:
            rosters = teams_api.get_team_roster(season=season, team=team)

            if rosters:
                roster = rosters[0]  # First team's roster
                roster_dict = roster.to_dict()

                print(f"✓ SUCCESS: Retrieved roster for {roster_dict.get('team', team)}")
                print(f"\nRoster keys: {list(roster_dict.keys())}")

                players = roster_dict.get('players', [])
                if players:
                    players_df = pd.DataFrame(players)
                    print(f"\nPlayers ({len(players_df)}):")
                    print(f"  Player columns: {list(players_df.columns)}")
                    print(f"\nSample (first 3):")
                    display_cols = [c for c in ['name', 'position', 'height', 'jersey'] if c in players_df.columns]
                    print(players_df[display_cols].head(3).to_string(index=False))

                return True, roster_dict
            else:
                print(f"✗ No roster found for {team}")
                return False, None

        except ApiException as e:
            print(f"✗ FAILED: {e}")
            return False, None


def test_get_games(configuration, season=2026, team="Duke"):
    """Test 3: get_games() - Game metadata."""
    print("\n" + "=" * 60)
    print(f"TEST 3: GamesApi.get_games(season={season}, team='{team}')")
    print("=" * 60)

    with cbbd.ApiClient(configuration) as api_client:
        games_api = cbbd.GamesApi(api_client)

        try:
            games = games_api.get_games(season=season, team=team)
            df = pd.DataFrame([g.to_dict() for g in games])

            print(f"✓ SUCCESS: Retrieved {len(df)} games")
            print(f"\nColumns ({len(df.columns)}):")
            print(f"  {list(df.columns)}")
            print(f"\nSample (first 3):")
            display_cols = ['id', 'start_date', 'home_team', 'away_team', 'home_points', 'away_points']
            display_cols = [c for c in display_cols if c in df.columns]
            print(df[display_cols].head(3).to_string(index=False))

            return True, df

        except ApiException as e:
            print(f"✗ FAILED: {e}")
            return False, None


def test_get_plays(configuration, season=2026, team="Duke"):
    """Test 4: get_plays_by_team() - Play-by-play data (CRITICAL)."""
    print("\n" + "=" * 60)
    print(f"TEST 4: PlaysApi.get_plays_by_team(season={season}, team='{team}')")
    print("=" * 60)

    with cbbd.ApiClient(configuration) as api_client:
        plays_api = cbbd.PlaysApi(api_client)

        try:
            # Get just shooting plays to limit data
            plays = plays_api.get_plays_by_team(
                season=season,
                team=team,
                shooting_plays_only=True  # Limit to save API calls
            )
            df = pd.DataFrame([p.to_dict() for p in plays])

            print(f"✓ SUCCESS: Retrieved {len(df)} shooting plays")
            print(f"\nColumns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                print(f"  {i+1:2}. {col}")

            # CRITICAL: Check for on_floor
            print("\n" + "-" * 40)
            print("CRITICAL FIELD CHECKS:")
            print("-" * 40)

            has_on_floor = 'on_floor' in df.columns or 'onFloor' in df.columns
            on_floor_col = 'on_floor' if 'on_floor' in df.columns else 'onFloor' if 'onFloor' in df.columns else None

            if has_on_floor:
                print(f"✓ on_floor: FOUND (column: '{on_floor_col}')")
                # Check if it has data
                sample = df[on_floor_col].dropna().head(1)
                if len(sample) > 0:
                    print(f"  Sample value: {sample.iloc[0]}")
            else:
                print("✗ on_floor: NOT FOUND - Lineup tracking may not work!")

            has_shot_info = 'shot_info' in df.columns or 'shotInfo' in df.columns
            shot_col = 'shot_info' if 'shot_info' in df.columns else 'shotInfo' if 'shotInfo' in df.columns else None

            if has_shot_info:
                print(f"✓ shot_info: FOUND (column: '{shot_col}')")
                sample = df[shot_col].dropna().head(1)
                if len(sample) > 0:
                    print(f"  Sample value: {sample.iloc[0]}")
            else:
                print("✗ shot_info: NOT FOUND")

            has_participants = 'participants' in df.columns
            if has_participants:
                print(f"✓ participants: FOUND")
                sample = df['participants'].dropna().head(1)
                if len(sample) > 0:
                    print(f"  Sample value: {sample.iloc[0]}")
            else:
                print("✗ participants: NOT FOUND")

            return True, df

        except ApiException as e:
            print(f"✗ FAILED: {e}")
            return False, None


def main():
    """Run all tests."""
    print("=" * 60)
    print("CBBD API ENDPOINT TESTS")
    print("ShotsDashboard Pipeline - Raw Data Validation")
    print("=" * 60)

    api_key = get_api_key()
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)

    configuration = create_client(api_key)

    season = 2026
    team = "Duke"  # Test with one team to minimize API calls

    print(f"\nTest parameters: season={season}, team='{team}'")
    print("(Using minimal queries to preserve API quota)")

    results = {}

    # Run tests
    results['teams'] = test_get_teams(configuration, season)
    results['roster'] = test_get_roster(configuration, season, team)
    results['games'] = test_get_games(configuration, season, team)
    results['plays'] = test_get_plays(configuration, season, team)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, (passed, _) in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Raw data layer is ready.")
    else:
        print("Some tests failed. Check output above for details.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
