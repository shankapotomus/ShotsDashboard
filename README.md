# ShotsDashboard

College Basketball Play-by-Play Data Collection and Analysis Tool using the CollegeBasketballData.com API for the 2025/26 season.

## Overview

ShotsDashboard is a comprehensive Jupyter notebook that demonstrates how to collect, process, and analyze college basketball play-by-play data from the CollegeBasketballData.com (CBBD) API. This tool enables you to gather detailed game data including shooting locations, player actions, and game events for statistical analysis, shot charts, and advanced metrics.

## Features

- **Multiple Data Collection Methods**:
  - Collect play-by-play data for specific games
  - Retrieve all games within a date range
  - Get entire season data for specific teams
  - Filter for shooting plays only (ideal for shot charts)

- **Comprehensive Play-by-Play Data**:
  - All game events (shots, rebounds, turnovers, fouls, substitutions)
  - Shot location coordinates (X, Y) for spatial analysis
  - Play timestamps and game context
  - Team and player information

- **Data Processing & Export**:
  - Automatic conversion to pandas DataFrames
  - CSV export functionality
  - Shot location extraction
  - Data quality checks and summary statistics

- **Advanced Analysis Ready**:
  - Shot chart creation
  - Shooting efficiency calculations
  - Advanced metrics (offensive rating, pace, etc.)
  - Lineup performance analysis
  - Player development tracking

## Prerequisites

- **Python 3.7+**
- **API Key**: Register for a free API key at [CollegeBasketballData.com](https://collegebasketballdata.com)
- **Required Python Packages**:
  ```bash
  pip install cbbd pandas numpy
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/shankapotomus/ShotsDashboard.git
   cd ShotsDashboard
   ```

2. Install required packages:
   ```bash
   pip install cbbd pandas numpy
   ```

3. Get your API key from [CollegeBasketballData.com](https://collegebasketballdata.com)

## Usage

### Quick Start

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook cbbd_playbyplay_2025_26.ipynb
   ```

2. Follow the notebook cells sequentially:
   - **Cell 1-2**: Import libraries and set up working directory
   - **Cell 3-4**: Configure API authentication with your API key
   - **Cell 5-6**: Set the season parameter (2026 for 2025/26 season)
   - **Cell 7-16**: Choose one of four data collection methods
   - **Cell 17-20**: Extract shot locations and export to CSV

### Data Collection Methods

The notebook provides four flexible methods to collect data:

#### Method 1: Specific Game Data
Retrieve play-by-play for games on a specific date:
```python
# Prompts for date, collects all games on that date
# Output: games_df, plays_df
```

#### Method 2: Date Range Collection
Get all plays for games within a date range:
```python
start_date = '2025-11-04'
end_date = '2025-11-07'
# Output: plays_date_df
```

#### Method 3: Team Season Data
Collect entire season for a specific team:
```python
team_name = 'Duke'
# Output: team_plays_df
```

#### Method 4: Shooting Plays Only
Filter for shooting plays (for shot charts):
```python
team_name = 'Duke'
shooting_plays_only = True
# Output: shooting_df
```

### API Authentication

The notebook uses secure API key entry:
```python
import getpass
api_key = getpass.getpass("API Key: ")
```

Alternatively, use environment variables:
```python
api_key = os.environ.get('CBBD_API_KEY')
```

## Output Files

Data is automatically saved to the `cbbd_data/` directory:

- **games_{date}_{season}.csv**: Game metadata and scores
- **plays_{date}_{season}.csv**: Complete play-by-play data
- **shot_locations_{date}_{season}.csv**: Extracted shot coordinates and context

### Output Data Structure

**Games DataFrame** includes:
- Game IDs, teams, scores
- Date, season, game type
- Conference information

**Plays DataFrame** includes:
- Play type, period, clock
- Team and opponent
- Score before/after play
- Player participants
- Shot information (for shooting plays)

**Shot Locations DataFrame** includes:
- X, Y coordinates
- Shot type and distance
- Made/missed indicator
- Game context (period, clock, score)

## Example Analyses

The notebook includes several analysis examples:

### Shooting Statistics
```python
# Automatic calculation of:
# - Field Goal %
# - Three-Point %
# - Shots with location coordinates
```

### Shot Location Extraction
```python
# Extracts X, Y coordinates from shooting plays
# Ready for matplotlib/seaborn visualization
```

## Season Timing

- NCAA basketball season: November through April
- Use `season=2026` for the 2025/26 academic year
- Season parameter uses the **end year** of the academic year

## Tips and Best Practices

### Rate Limiting
- Add delays between bulk requests: `time.sleep(0.5)`
- Monitor API rate limits
- Batch process multiple teams efficiently

### Data Storage
- Save data incrementally to avoid data loss
- Include timestamps in filenames for versioning
- Consider parquet format for large datasets

### Error Handling
- All API calls include try-except blocks
- Logs exceptions for debugging
- Gracefully handles missing data

## Next Steps

After collecting data, you can:
1. **Create shot charts** using matplotlib/seaborn
2. **Calculate advanced metrics** (offensive rating, pace, effective FG%)
3. **Analyze lineup performance** and player combinations
4. **Build predictive models** for game outcomes
5. **Track player development** throughout the season

For shot chart creation, see the [CBBD blog post](https://blog.collegefootballdata.com/talking-tech-generating-shot-charts-using-the-basketball-api/)

## API Reference

This project uses the [CollegeBasketballData.com API](https://api.collegebasketballdata.com):
- **Authentication**: Bearer token
- **Available Endpoints**: Games, Plays, Teams, Players
- **Documentation**: [API Docs](https://api.collegebasketballdata.com/docs)

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or analyses
- Submit pull requests

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Data provided by [CollegeBasketballData.com](https://collegebasketballdata.com)
- Built with the CBBD Python SDK

---

**Happy analyzing!** 🏀