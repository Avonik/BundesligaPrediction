# Combined Bundesliga Prediction Pipeline
import requests
import pandas as pd
import numpy as np
import os
import joblib # For saving/loading the model
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
# Add near other sklearn imports
from sklearn.metrics import brier_score_loss
# Import the new model
import lightgbm as lgb

# --- Configuration ---
# File Paths
BUNDESLIGA_RAW_CSV = "bundesliga_matches_combined_v2.csv"
BUNDESLIGA_FEATURES_CSV = "bundesliga_features_elo_v2.csv"
ELO_HISTORY_CSV = "elo_history_detailed_v2.csv"
MODEL_FILE = "bundesliga_model_v2.joblib"
PREDICTIONS_CSV = "bundesliga_predictions_v2.csv"

# Data Fetching
API_BASE_URL = "https://api.openligadb.de/getmatchdata/bl1/{season}/{matchday}"
# Define the seasons you want to fetch HISTORICAL data for.
# Example: range(2010, 2024) fetches seasons 2010/11 up to 2023/24
HISTORICAL_SEASONS = range(2010, 2024)
# Define the season(s) for UPCOMING matches (usually the current or next one)
# Example: Fetches 2024/2025. The script will automatically only predict matches without scores.
UPCOMING_SEASONS = range(2024, 2025)
MAX_MATCHDAY = 34

# Feature Calculation Constants
INITIAL_ELO = 1500
ELO_K_FACTOR = 20
HOME_ADVANTAGE_ELO = 65 # Points added to home team's ELO for expected result calculation
FORM_WINDOW = 5 # Number of games for form calculation
H2H_WINDOW = 3 # Number of games for head-to-head calculation
PROMOTION_RESET_ELO = 1450 # ELO for teams absent long term (closer to avg)

# Model Training
# Define features to be used by the model. MUST match columns generated in calculate_features
# Adjusted based on elocal.py output. Add more if you engineer them.
FEATURES = [
    # Standard Elo
    'pre_match_elo_home', 'pre_match_elo_away', 'pre_match_elo_diff',
    # Home/Away Elo
    'pre_match_elo_home_H', # Home team's Home Elo
    'pre_match_elo_away_A', # Away team's Away Elo
    'elo_home_H_vs_elo_away_A_diff', # Difference between Home Elo (Home) and Away Elo (Away)
    # Weighted Form (Overall) - Example names, adjust if needed
    'form_points_weighted_home', 'form_points_weighted_away', 'form_points_weighted_diff',
    'form_gd_weighted_home', 'form_gd_weighted_away', 'form_gd_weighted_diff',
    # Home/Away Specific Form (Weighted) - Example names
    'form_points_weighted_home_H', # Home team's weighted points form in home games
    'form_points_weighted_away_A', # Away team's weighted points form in away games
    'form_gd_weighted_home_H', # Home team's weighted GD form in home games
    'form_gd_weighted_away_A', # Away team's weighted GD form in away games
    # H2H Points/GD (Venue Specific) - Example names
    'h2h_points_venue', # Points for home team in recent H2H matches at this venue
    'h2h_gd_venue',     # Goal diff for home team in recent H2H matches at this venue
    # Season PPG (as before)
    'ppg_season_home', 'ppg_season_away', 'ppg_season_diff'
    # Add other previous features if desired (e.g., non-weighted form, overall H2H)
]

TARGET_VARIABLE = 'match_outcome' # 1: Home Win, 0: Draw, -1: Away Win
FORCE_RETRAIN = True # Set to True to always retrain the model, even if a file exists

# Elo K Factor configuration - Now supporting variable K
ELO_K_FACTOR_INITIAL = 30 # Higher K early in season
ELO_K_FACTOR_STABLE = 15  # Lower K later in season
ELO_K_STABILITY_MATCHDAY = 6 # Matchday after which K factor reduces

# Form weighting configuration
FORM_WEIGHTING_DECAY = 0.85 # Weight multiplier for each older game (e.g., 1, 0.85, 0.85^2, ...)

# --- Helper Functions ---

def get_season_from_date(date):
    """Determines the football season string (e.g., '2023/2024') from a datetime object."""
    year = date.year
    month = date.month
    if month >= 7: # Assuming season starts around July/August
        return f"{year}/{year + 1}"
    else:
        return f"{year - 1}/{year}"

def expected_result(elo_a, elo_b):
    """Calculates the expected score (probability of winning) for player A."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

# --- Step 1: Data Fetching ---

def get_match_data_api(season, matchday):
    """Fetches match data for a specific season and matchday from OpenLigaDB API."""
    url = API_BASE_URL.format(season=season, matchday=matchday)
    try:
        response = requests.get(url, timeout=15) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for Season {season}, Matchday {matchday}: {e}")
        return None

def parse_match_data(data, season_year):
    """Parses the JSON data from the API into a list of dictionaries."""
    match_list = []
    if not data:
        return [] # Return empty list if data is None or empty

    for match in data:
        # Determine actual season string based on date if possible
        match_dt = pd.to_datetime(match.get("matchDateTime"), errors='coerce')
        season_str = get_season_from_date(match_dt) if pd.notna(match_dt) else f"{season_year}/{season_year+1}"

        # Handle potential missing matchResults or different structures
        score_home, score_away = None, None
        if match.get("matchResults"):
            # Find the final result (usually typeID 2) or take the first one
            final_result = next((r for r in match["matchResults"] if r.get("resultTypeID") == 2), match["matchResults"][0])
            score_home = final_result.get("pointsTeam1")
            score_away = final_result.get("pointsTeam2")

        match_info = {
            "api_season": season_year, # The year used in the API call
            "matchday": match["group"]["groupOrderID"], # Use OrderID for numerical matchday
            "group_name": match["group"]["groupName"], # Keep the original name too
            "date": match.get("matchDateTime"),
            "team_home": match["team1"]["teamName"],
            "team_away": match["team2"]["teamName"],
            "score_home": score_home,
            "score_away": score_away,
            "match_id": match.get("matchID") # Useful identifier
        }
        match_list.append(match_info)
    return match_list

def fetch_all_data(seasons_to_fetch, output_csv):
    """Fetches data for multiple seasons and saves to CSV."""
    print(f"Fetching data for seasons: {list(seasons_to_fetch)}...")
    all_matches_data = []
    for season in seasons_to_fetch:
        print(f"--- Fetching Season: {season} ---")
        season_has_data = False
        for matchday in tqdm(range(1, MAX_MATCHDAY + 1), desc=f"Season {season} Matchdays"):
            data = get_match_data_api(season, matchday)
            if data:
                parsed_data = parse_match_data(data, season)
                all_matches_data.extend(parsed_data)
                season_has_data = True
        if not season_has_data:
             print(f"Warning: No data found for season {season}.")

    if not all_matches_data:
        print("No data fetched. Exiting feature calculation.")
        return None

    df = pd.DataFrame(all_matches_data)
    print(f"Fetched {len(df)} matches.")

    # Basic Cleaning before saving raw data
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df["score_home"] = pd.to_numeric(df["score_home"], errors='coerce')
    df["score_away"] = pd.to_numeric(df["score_away"], errors='coerce')
    df.dropna(subset=["date", "team_home", "team_away"], inplace=True) # Drop rows missing essential info
    df.sort_values(by="date", inplace=True) # Sort chronologically before saving
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Raw match data saved to {output_csv}")
    return df

# --- Step 2: Feature Calculation (including ELO) ---
# Using the functions from elocal.py (with minor adjustments)

# Feature calculation helper functions (as provided in elocal.py, slightly adapted)
def calculate_points_form(team, df_history, current_match_date, date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=FORM_WINDOW):
    team_matches = df_history[((df_history[home_col] == team) | (df_history[away_col] == team)) & (df_history[date_col] < current_match_date)].sort_values(by=date_col, ascending=False).head(num_games)
    if len(team_matches) == 0: return np.nan # No history yet
    if len(team_matches) < min(num_games, 1) : return np.nan # Avoid calculating with too few games early on

    points = 0
    valid_game_count = 0
    for _, row in team_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]):
            continue # Skip matches without scores in history
        valid_game_count += 1
        if row[home_col] == team:
            if row[score_h_col] > row[score_a_col]: points += 3
            elif row[score_h_col] == row[score_a_col]: points += 1
        else: # Team was away
            if row[score_a_col] > row[score_h_col]: points += 3
            elif row[score_a_col] == row[score_h_col]: points += 1

    return points / valid_game_count if valid_game_count > 0 else 0 # Avoid division by zero

def calculate_gd_form(team, df_history, current_match_date, date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=FORM_WINDOW):
    team_matches = df_history[((df_history[home_col] == team) | (df_history[away_col] == team)) & (df_history[date_col] < current_match_date)].sort_values(by=date_col, ascending=False).head(num_games)
    if len(team_matches) < min(num_games, 1): return np.nan
    goal_differences = []
    for _, row in team_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]): continue
        if row[home_col] == team:
            goal_differences.append(row[score_h_col] - row[score_a_col])
        else:
            goal_differences.append(row[score_a_col] - row[score_h_col])
    return np.mean(goal_differences) if goal_differences else 0

def calculate_goals_scored_form(team, df_history, current_match_date, date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=FORM_WINDOW):
    team_matches = df_history[((df_history[home_col] == team) | (df_history[away_col] == team)) & (df_history[date_col] < current_match_date)].sort_values(by=date_col, ascending=False).head(num_games)
    if len(team_matches) < min(num_games, 1): return np.nan
    goals = []
    for _, row in team_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]): continue
        if row[home_col] == team: goals.append(row[score_h_col])
        else: goals.append(row[score_a_col])
    return np.mean(goals) if goals else 0

def calculate_goals_conceded_form(team, df_history, current_match_date, date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=FORM_WINDOW):
    team_matches = df_history[((df_history[home_col] == team) | (df_history[away_col] == team)) & (df_history[date_col] < current_match_date)].sort_values(by=date_col, ascending=False).head(num_games)
    if len(team_matches) < min(num_games, 1): return np.nan
    goals = []
    for _, row in team_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]): continue
        if row[home_col] == team: goals.append(row[score_a_col])
        else: goals.append(row[score_h_col])
    return np.mean(goals) if goals else 0

def calculate_h2h_goal_diff(home_team, away_team, df_history, current_match_date, date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=H2H_WINDOW):
    h2h_matches = df_history[
        (((df_history[home_col] == home_team) & (df_history[away_col] == away_team)) |
         ((df_history[home_col] == away_team) & (df_history[away_col] == home_team))) &
        (df_history[date_col] < current_match_date)
    ].sort_values(by=date_col, ascending=False).head(num_games)
    if len(h2h_matches) == 0: return 0 # Return 0 if no prior H2H, treating it as neutral
    goal_diffs = []
    for _, row in h2h_matches.iterrows():
         if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]): continue
         if row[home_col] == home_team: # Perspective of the current home team
            goal_diffs.append(row[score_h_col] - row[score_a_col])
         else: # current home_team was the away team in this historical match
            goal_diffs.append(row[score_a_col] - row[score_h_col]) # Flip the difference
    return np.mean(goal_diffs) if goal_diffs else 0

def calculate_points_per_game_season(team, df_history, current_season, current_match_date, season_col='season', date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away'):
    season_matches = df_history[
        ((df_history[home_col] == team) | (df_history[away_col] == team)) &
        (df_history[season_col] == current_season) &
        (df_history[date_col] < current_match_date)
    ]
    if len(season_matches) == 0: return 0 # No matches played yet in season

    points = 0
    valid_matches_count = 0
    for _, row in season_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]): continue
        valid_matches_count += 1
        if row[home_col] == team:
            if row[score_h_col] > row[score_a_col]: points += 3
            elif row[score_h_col] == row[score_a_col]: points += 1
        else: # Team was away
            if row[score_a_col] > row[score_h_col]: points += 3
            elif row[score_a_col] == row[score_h_col]: points += 1
    return points / valid_matches_count if valid_matches_count > 0 else 0


# --- Feature Calculation Helpers (Modified/New) ---

def get_form_weights(num_games, decay_factor):
    """Generates decaying weights for form calculation."""
    weights = [decay_factor**i for i in range(num_games)]
    return np.array(weights) / sum(weights) # Normalize weights

def calculate_weighted_points_form(team, df_history, current_match_date, venue='all', date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=FORM_WINDOW, decay=FORM_WEIGHTING_DECAY):
    """Calculates weighted average points per game from the last num_games matches."""
    team_matches = df_history[
        ((df_history[home_col] == team) | (df_history[away_col] == team)) &
        (df_history[date_col] < current_match_date)
    ].sort_values(by=date_col, ascending=False)

    # Filter by venue if specified
    if venue == 'home':
        team_matches = team_matches[team_matches[home_col] == team]
    elif venue == 'away':
        team_matches = team_matches[team_matches[away_col] == team]

    team_matches = team_matches.head(num_games)

    if len(team_matches) == 0: return np.nan
    if len(team_matches) < min(num_games, 2): return np.nan # Need at least a couple of games

    points = []
    for _, row in team_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]):
            points.append(np.nan) # Keep placeholder for weight alignment
            continue
        if row[home_col] == team:
            if row[score_h_col] > row[score_a_col]: points.append(3)
            elif row[score_h_col] == row[score_a_col]: points.append(1)
            else: points.append(0)
        else: # Team was away
            if row[score_a_col] > row[score_h_col]: points.append(3)
            elif row[score_a_col] == row[score_h_col]: points.append(1)
            else: points.append(0)

    points = np.array(points)
    valid_indices = ~np.isnan(points)
    if not np.any(valid_indices): return 0 # No valid games found

    # Apply weights only to valid points
    weights = get_form_weights(len(points), decay)
    weighted_avg = np.average(points[valid_indices], weights=weights[valid_indices])

    # Scale weighted average back to approx points per game (optional, helps interpretability)
    # Approximation: A simple average team gets 1 point per game.
    # Weighted average might be different. Scaling can help.
    # Let's return the direct weighted average for now.
    return weighted_avg


def calculate_weighted_gd_form(team, df_history, current_match_date, venue='all', date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=FORM_WINDOW, decay=FORM_WEIGHTING_DECAY):
    """Calculates weighted average goal difference from the last num_games matches."""
    team_matches = df_history[
        ((df_history[home_col] == team) | (df_history[away_col] == team)) &
        (df_history[date_col] < current_match_date)
    ].sort_values(by=date_col, ascending=False)

    if venue == 'home':
        team_matches = team_matches[team_matches[home_col] == team]
    elif venue == 'away':
        team_matches = team_matches[team_matches[away_col] == team]

    team_matches = team_matches.head(num_games)

    if len(team_matches) == 0: return np.nan
    if len(team_matches) < min(num_games, 2): return np.nan

    goal_diffs = []
    for _, row in team_matches.iterrows():
        if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]):
            goal_diffs.append(np.nan)
            continue
        if row[home_col] == team:
            goal_diffs.append(row[score_h_col] - row[score_a_col])
        else:
            goal_diffs.append(row[score_a_col] - row[score_h_col])

    goal_diffs = np.array(goal_diffs)
    valid_indices = ~np.isnan(goal_diffs)
    if not np.any(valid_indices): return 0

    weights = get_form_weights(len(goal_diffs), decay)
    weighted_avg = np.average(goal_diffs[valid_indices], weights=weights[valid_indices])
    return weighted_avg


def calculate_h2h_metrics(home_team, away_team, df_history, current_match_date, venue='all', date_col='date', home_col='team_home', away_col='team_away', score_h_col='score_home', score_a_col='score_away', num_games=H2H_WINDOW):
    """Calculates average goal difference and points for the home_team from last N H2H matches."""
    # Base filter for H2H matches before the current date
    h2h_matches_base = df_history[
        (((df_history[home_col] == home_team) & (df_history[away_col] == away_team)) |
         ((df_history[home_col] == away_team) & (df_history[away_col] == home_team))) &
        (df_history[date_col] < current_match_date)
    ].sort_values(by=date_col, ascending=False)

    # Apply venue filter
    if venue == 'specific': # Only matches where home_team was home
        h2h_matches = h2h_matches_base[h2h_matches_base[home_col] == home_team]
    elif venue == 'all':
        h2h_matches = h2h_matches_base
    else: # Should not happen, but default to all
        h2h_matches = h2h_matches_base

    h2h_matches = h2h_matches.head(num_games)

    if len(h2h_matches) == 0:
        return 0.0, 0.0 # Return neutral GD and points if no history

    goal_diffs = []
    points_list = []
    valid_matches = 0
    for _, row in h2h_matches.iterrows():
         if pd.isna(row[score_h_col]) or pd.isna(row[score_a_col]):
            continue
         valid_matches += 1
         # Always calculate from the perspective of the *current* home_team
         if row[home_col] == home_team:
            gd = row[score_h_col] - row[score_a_col]
            if gd > 0: pts = 3
            elif gd == 0: pts = 1
            else: pts = 0
            goal_diffs.append(gd)
            points_list.append(pts)
         else: # current home_team was the away team in this historical match
            gd = row[score_a_col] - row[score_h_col] # Flip GD
            if gd > 0: pts = 3
            elif gd == 0: pts = 1
            else: pts = 0
            goal_diffs.append(gd)
            points_list.append(pts)

    if valid_matches == 0:
        return 0.0, 0.0

    avg_gd = np.mean(goal_diffs) if goal_diffs else 0.0
    avg_points = np.mean(points_list) if points_list else 0.0

    return avg_gd, avg_points


# --- Main Feature Calculation Function (Heavily Modified) ---

def calculate_features(df_raw):
    """Calculates ELO ratings and other features chronologically."""
    print("Calculating features (ELO, Form, H2H, etc.) - Enhanced Version...")
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df.dropna(subset=["date", "team_home", "team_away", "matchday"], inplace=True) # Ensure matchday exists
    # Convert matchday to numeric, coercing errors
    df["matchday"] = pd.to_numeric(df["matchday"], errors='coerce')
    df.dropna(subset=["matchday"], inplace=True) # Drop if matchday couldn't be converted
    df["matchday"] = df["matchday"].astype(int)

    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['season'] = df['date'].apply(get_season_from_date)

    # --- Initialize ELO Ratings (Overall, Home, Away) ---
    print("Initializing ELO ratings (Overall, Home, Away)...")
    all_teams = set(df["team_home"]).union(set(df["team_away"]))
    team_elo = {team: INITIAL_ELO for team in all_teams}
    team_elo_home = {team: INITIAL_ELO for team in all_teams}
    team_elo_away = {team: INITIAL_ELO for team in all_teams}
    last_seen_season = {team: "" for team in all_teams}

    # --- Prepare columns for features ---
    feature_cols_to_add = [
        # Overall Elo
        'pre_match_elo_home', 'pre_match_elo_away',
        # Home/Away Elo
        'pre_match_elo_home_H', 'pre_match_elo_away_A',
        # Weighted Form (Overall)
        'form_points_weighted_home', 'form_points_weighted_away',
        'form_gd_weighted_home', 'form_gd_weighted_away',
        # Weighted Form (Home/Away Specific)
        'form_points_weighted_home_H', 'form_points_weighted_away_A', # Home team home form, Away team away form
        'form_gd_weighted_home_H', 'form_gd_weighted_away_A',
        # H2H (Venue Specific)
        'h2h_gd_venue', 'h2h_points_venue',
        # Season PPG (as before)
        'ppg_season_home', 'ppg_season_away',
        # Target
        TARGET_VARIABLE
    ]
    for col in feature_cols_to_add:
        df[col] = np.nan # Initialize columns

    elo_history = []

    print(f"Processing {len(df)} matches for feature calculation...")
    # --- Iterate through matches chronologically ---
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Calculating Features"):
        home_team, away_team = row["team_home"], row["team_away"]
        score_home, score_away = row["score_home"], row["score_away"]
        match_date = row["date"]
        current_season = row["season"]
        matchday = row["matchday"]

        # --- Promotion/Relegation ELO Adjustment ---
        current_year = match_date.year
        for team in [home_team, away_team]:
            if team not in team_elo: # Initialize new teams if encountered
                team_elo[team] = PROMOTION_RESET_ELO
                team_elo_home[team] = PROMOTION_RESET_ELO
                team_elo_away[team] = PROMOTION_RESET_ELO
                last_seen_season[team] = ""
                print(f"Team '{team}' not seen before, initializing ELOs to {PROMOTION_RESET_ELO}")

            last_s = last_seen_season[team]
            if last_s and not (str(current_year) in last_s or str(current_year-1) in last_s):
                 team_elo[team] = (team_elo[team] + PROMOTION_RESET_ELO) / 2
                 team_elo_home[team] = (team_elo_home[team] + PROMOTION_RESET_ELO) / 2
                 team_elo_away[team] = (team_elo_away[team] + PROMOTION_RESET_ELO) / 2

        # --- Store Pre-Match ELOs ---
        df.loc[index, 'pre_match_elo_home'] = team_elo[home_team]
        df.loc[index, 'pre_match_elo_away'] = team_elo[away_team]
        df.loc[index, 'pre_match_elo_home_H'] = team_elo_home[home_team]
        df.loc[index, 'pre_match_elo_away_A'] = team_elo_away[away_team]

        # Store ELO history (using overall ELO for simplicity here)
        elo_history.append({"date": match_date, "team": home_team, "elo": team_elo[home_team], "season": current_season})
        elo_history.append({"date": match_date, "team": away_team, "elo": team_elo[away_team], "season": current_season})

        # --- Calculate Other Pre-Match Features ---
        df_history_for_calc = df.iloc[:index]

        # Weighted Form (Overall)
        df.loc[index, 'form_points_weighted_home'] = calculate_weighted_points_form(home_team, df_history_for_calc, match_date, venue='all')
        df.loc[index, 'form_points_weighted_away'] = calculate_weighted_points_form(away_team, df_history_for_calc, match_date, venue='all')
        df.loc[index, 'form_gd_weighted_home'] = calculate_weighted_gd_form(home_team, df_history_for_calc, match_date, venue='all')
        df.loc[index, 'form_gd_weighted_away'] = calculate_weighted_gd_form(away_team, df_history_for_calc, match_date, venue='all')

        # Weighted Form (Home/Away Specific)
        df.loc[index, 'form_points_weighted_home_H'] = calculate_weighted_points_form(home_team, df_history_for_calc, match_date, venue='home')
        df.loc[index, 'form_points_weighted_away_A'] = calculate_weighted_points_form(away_team, df_history_for_calc, match_date, venue='away')
        df.loc[index, 'form_gd_weighted_home_H'] = calculate_weighted_gd_form(home_team, df_history_for_calc, match_date, venue='home')
        df.loc[index, 'form_gd_weighted_away_A'] = calculate_weighted_gd_form(away_team, df_history_for_calc, match_date, venue='away')

        # H2H (Venue Specific)
        h2h_gd_v, h2h_pts_v = calculate_h2h_metrics(home_team, away_team, df_history_for_calc, match_date, venue='specific')
        df.loc[index, 'h2h_gd_venue'] = h2h_gd_v
        df.loc[index, 'h2h_points_venue'] = h2h_pts_v

        # Season PPG (Original logic)
        df.loc[index, 'ppg_season_home'] = calculate_points_per_game_season(home_team, df_history_for_calc, current_season, match_date)
        df.loc[index, 'ppg_season_away'] = calculate_points_per_game_season(away_team, df_history_for_calc, current_season, match_date)


        # --- Update ELO & Define Target (If result known) ---
        if pd.notna(score_home) and pd.notna(score_away):
            score_diff = score_home - score_away
            # Define target variable
            if score_diff > 0:
                df.loc[index, TARGET_VARIABLE] = 1; actual_home_score_prob = 1.0
            elif score_diff < 0:
                df.loc[index, TARGET_VARIABLE] = -1; actual_home_score_prob = 0.0
            else:
                df.loc[index, TARGET_VARIABLE] = 0; actual_home_score_prob = 0.5

            # --- ELO Update Logic ---
            # 1. Determine K-Factor (Variable K)
            k_factor = ELO_K_FACTOR_INITIAL if matchday <= ELO_K_STABILITY_MATCHDAY else ELO_K_FACTOR_STABLE

            # 2. Weighting by Goal Difference (Example: simple multiplier)
            # Avoid extreme multipliers; cap the effect
            gd_weight = 1 + 0.1 * np.log(1 + abs(score_diff)) # Log scale to dampen effect
            gd_weight = min(gd_weight, 1.5) # Cap the multiplier

            # 3. Calculate ELO Updates (for Overall, Home, Away)
            # Overall Elo Update (using overall Elo + standard home advantage for expected calc)
            elo_home_adj = team_elo[home_team] + HOME_ADVANTAGE_ELO
            expected_home_overall = expected_result(elo_home_adj, team_elo[away_team])
            update_val_overall = k_factor * gd_weight * (actual_home_score_prob - expected_home_overall)
            team_elo[home_team] += update_val_overall
            team_elo[away_team] -= update_val_overall

            # Home Elo Update (only update Home team's Home Elo, based on Overall Elo expectation)
            # Weaker update as it only reflects one venue type
            update_val_home = update_val_overall * 0.5 # Smaller update for venue-specific Elo
            team_elo_home[home_team] += update_val_home

            # Away Elo Update (only update Away team's Away Elo, based on Overall Elo expectation)
            update_val_away = -update_val_overall * 0.5 # Smaller update for venue-specific Elo
            team_elo_away[away_team] += update_val_away # Note: update is negative of overall update


            # Update last seen season
            last_seen_season[home_team] = current_season
            last_seen_season[away_team] = current_season
        else:
            df.loc[index, TARGET_VARIABLE] = np.nan
            # Still update last seen season for future match rows
            last_seen_season[home_team] = current_season
            last_seen_season[away_team] = current_season

    # --- Post-Loop Calculations (Difference Features) ---
    print("Calculating difference features...")
    df['pre_match_elo_diff'] = df['pre_match_elo_home'] - df['pre_match_elo_away']
    df['elo_home_H_vs_elo_away_A_diff'] = df['pre_match_elo_home_H'] - df['pre_match_elo_away_A']
    df['form_points_weighted_diff'] = df['form_points_weighted_home'] - df['form_points_weighted_away']
    df['form_gd_weighted_diff'] = df['form_gd_weighted_home'] - df['form_gd_weighted_away']
    df['ppg_season_diff'] = df['ppg_season_home'] - df['ppg_season_away']
    # Add more diffs if needed based on FEATURES list

    # --- Handle Missing Values (NaNs) ---
    print("Handling missing values in features...")
    # Define fill values (adapt as needed for new features)
    # Calculate medians/means *before* splitting train/test if possible
    # Using neutral values (0 for diffs, 1 for points) for simplicity here
    nan_fill_values = {
        # Elo diffs
        'pre_match_elo_diff': 0, 'elo_home_H_vs_elo_away_A_diff': 0,
        # Weighted Form diffs
        'form_points_weighted_diff': 0.0, 'form_gd_weighted_diff': 0.0,
        # Weighted Form absolutes (Assume neutral form ~1pt/game if unknown)
        'form_points_weighted_home': 1.0, 'form_points_weighted_away': 1.0,
        'form_gd_weighted_home': 0.0, 'form_gd_weighted_away': 0.0,
        'form_points_weighted_home_H': 1.0, 'form_points_weighted_away_A': 1.0,
        'form_gd_weighted_home_H': 0.0, 'form_gd_weighted_away_A': 0.0,
        # H2H (Assume neutral if no history)
        'h2h_gd_venue': 0.0, 'h2h_points_venue': 1.0, # Assume ~1 H2H point/game if unknown
        # Season PPG
        'ppg_season_home': 1.0, 'ppg_season_away': 1.0, 'ppg_season_diff': 0.0,
    }
    # Fill NaNs only for columns intended to be used as features
    for col in FEATURES:
        if col in nan_fill_values and col in df.columns:
            if df[col].isnull().any():
                df[col].fillna(value=nan_fill_values[col], inplace=True)

    # Drop rows if essential base ELO is missing (shouldn't happen)
    df.dropna(subset=['pre_match_elo_home', 'pre_match_elo_away',
                       'pre_match_elo_home_H', 'pre_match_elo_away_A'], inplace=True)

    # --- Save Results ---
    print("Saving feature-engineered data...")
    df.to_csv(BUNDESLIGA_FEATURES_CSV, index=False, encoding='utf-8')
    print(f"Feature data saved to {BUNDESLIGA_FEATURES_CSV}")

    # Save ELO history
    if elo_history:
        elo_df = pd.DataFrame(elo_history)
        elo_df.to_csv(ELO_HISTORY_CSV, index=False, encoding='utf-8')
        print(f"ELO history saved to {ELO_HISTORY_CSV}")

    return df

# --- Step 3 & 4: Model Training ---

# --- Step 3 & 4: Model Training & Evaluation (Modified) ---

def train_model(df_features_historical, features_list, target_col, model_output_path, force_retrain=False, test_split_ratio=0.2):
    """
    Trains the prediction model (LightGBM) using a portion of historical data
    and evaluates on a held-out test portion. Includes class weighting and Brier score.
    """
    print("--- Model Training & Evaluation (LightGBM) ---")

    # --- Chronological Train/Test Split ---
    print(f"Splitting historical data chronologically (Test Ratio: {test_split_ratio:.0%})...")
    df_historical_sorted = df_features_historical.sort_values(by='date').reset_index(drop=True)
    split_index = int(len(df_historical_sorted) * (1 - test_split_ratio))

    if split_index == 0 or split_index == len(df_historical_sorted):
        print("Warning: Test split ratio results in empty train or test set. Adjust ratio or data size.")
        train_df = df_historical_sorted
        test_df = pd.DataFrame()
    else:
        train_df = df_historical_sorted.iloc[:split_index].copy()
        test_df = df_historical_sorted.iloc[split_index:].copy()

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # --- Model Training ---
    if os.path.exists(model_output_path) and not force_retrain:
        print(f"Loading existing model from {model_output_path}...")
        # Ensure model is loaded correctly (might need specific loader if not joblib, but joblib often works for LGBM)
        try:
            model = joblib.load(model_output_path)
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            force_retrain = True # Force retraining if loading fails

    if not os.path.exists(model_output_path) or force_retrain:
        print("Preparing training data...")
        X_train = train_df[features_list]
        y_train = train_df[target_col].astype(int)

        if X_train.isnull().sum().sum() > 0:
            print("Warning: NaNs found in training features. Imputing with median.")
            # Simple imputation for safety, replace with better strategy if needed
            X_train = X_train.fillna(X_train.median())

        print(f"Training LightGBM Classifier on {len(X_train)} samples...")
        # Initialize LightGBM Classifier
        model = lgb.LGBMClassifier(
            objective='multiclass', # for multi-class classification (H, D, A)
            num_class=3,            # Number of classes
            metric='multi_logloss', # Evaluation metric during training
            class_weight='balanced',# Address class imbalance
            random_state=42,
            n_jobs=-1,
            # Add other LGBM parameters here if desired (e.g., n_estimators, learning_rate, num_leaves)
            # Consider hyperparameter tuning separately
            n_estimators=200,      # Example: Increased estimators
            learning_rate=0.05,    # Example: Adjusted learning rate
            num_leaves=31          # Default value
        )

        model.fit(X_train, y_train)
        print("Training complete.")

        print(f"Saving trained model to {model_output_path}...")
        joblib.dump(model, model_output_path) # joblib should work for saving/loading basic LGBM models
        print("Model saved.")

    # --- Model Evaluation on Test Set ---
    if not test_df.empty:
        print("\n--- Evaluating model on the reserved test set ---")
        X_test = test_df[features_list]
        y_test = test_df[target_col].astype(int)

        if X_test.isnull().sum().sum() > 0:
            print("Warning: NaNs found in test features. Imputing with train median.")
             # Impute using *training* data stats to avoid leakage
            if 'X_train' in locals(): # Check if X_train exists from training phase
                 X_test = X_test.fillna(X_train.median())
            else: # Fallback if model was loaded and X_train not available
                 X_test = X_test.fillna(X_test.median()) # Less ideal

        print("Making predictions on the test set...")
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        print("\nTest Set Performance Metrics:")
        print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")

        # Calculate Brier Score Loss (lower is better)
        # Need to convert y_test to one-hot encoding for multiclass Brier score
        y_test_one_hot = pd.get_dummies(y_test, prefix='outcome').reindex(columns=model.classes_, fill_value=0)
        # Ensure columns match proba order (important!)
        y_test_one_hot = y_test_one_hot[[c for c in model.classes_ if c in y_test_one_hot.columns]] # handle missing classes if any

        if y_test_one_hot.shape[1] == y_test_proba.shape[1]: # Check dimensions match
           brier = brier_score_loss(y_test_one_hot.values.ravel(), y_test_proba.ravel())
           # Note: sklearn's brier_score_loss averages over all samples and classes
           print(f"Brier Score Loss: {brier:.4f}")
        else:
           print("Could not calculate Brier score due to mismatch in classes between y_test and predictions.")


        print("\nClassification Report (Test Set):")
        target_names_map = {-1: 'Away Win (-1)', 0: 'Draw (0)', 1: 'Home Win (1)'}
        present_classes = sorted(list(set(y_test) | set(y_test_pred)))
        target_names_ordered = [target_names_map[cls] for cls in present_classes if cls in target_names_map]
        print(classification_report(y_test, y_test_pred, labels=present_classes, target_names=target_names_ordered, zero_division=0))
    else:
        print("\nSkipping evaluation as no test set was generated.")

    return model

# --- Step 5: Prediction ---

def predict_upcoming_matches(model, df_features_predict, features_list, output_csv):
    """Makes predictions on future matches and saves them."""
    print("--- Predicting Upcoming Matches ---")
    if df_features_predict.empty:
        print("No future matches found to predict.")
        return

    print(f"Preparing {len(df_features_predict)} upcoming matches for prediction...")
    X_predict = df_features_predict[features_list]

    # Check for NaNs in prediction features (should have been handled)
    if X_predict.isnull().sum().sum() > 0:
        print("Warning: NaNs found in features for prediction. Check feature calculation/handling.")
        # Optional: Impute using stats from training data if a scaler/imputer was saved
        # Or use the same fill strategy as before
        # X_predict = X_predict.fillna(X_predict.median()) # Less ideal than using training medians

    # Optional: Scale prediction features using the *loaded* scaler fitted on training data
    # scaler = joblib.load('scaler.joblib') # Load the scaler saved during training
    # X_predict_scaled = scaler.transform(X_predict)
    # Use X_predict directly if no scaling was done during training

    print("Making predictions...")
    predictions_num = model.predict(X_predict)
    probabilities = model.predict_proba(X_predict)

    print("Processing prediction results...")
    # Map numerical predictions back to labels
    outcome_map = {1: 'Home Win', 0: 'Draw', -1: 'Away Win'}
    predictions_label = [outcome_map[pred] for pred in predictions_num]

    # Get probability class order (important!)
    prob_classes = model.classes_ # This gives the order, e.g., [-1, 0, 1]

    # Create a DataFrame for results
    results_df = df_features_predict[['date', 'matchday', 'team_home', 'team_away']].copy()
    results_df['predicted_outcome'] = predictions_label

    # Add probabilities for each class correctly aligned
    for i, class_label in enumerate(prob_classes):
        class_name = outcome_map[class_label]
        results_df[f'prob_{class_name}'] = probabilities[:, i] # Assign probability for this class

    # Calculate Fair Odds (1 / probability)
    for class_label in prob_classes:
        class_name = outcome_map[class_label]
        prob_col = f'prob_{class_name}'
        odds_col = f'fair_odds_{class_name}'
        # Avoid division by zero if probability is 0
        results_df[odds_col] = results_df[prob_col].apply(lambda p: round(1/p, 2) if p > 0 else None)


    print(f"Saving predictions to {output_csv}...")
    results_df.sort_values(by=['date', 'matchday'], inplace=True)
    results_df.to_csv(output_csv, index=False, encoding='utf-8', float_format='%.4f')
    print("Predictions saved.")

    # --- Print summary to console ---
    print("\n--- Prediction Summary ---")
    for index, row in results_df.head(15).iterrows(): # Print first few predictions
         print(f"{row['date'].strftime('%Y-%m-%d')}: {row['team_home']} vs {row['team_away']}")
         print(f"  Prediction: {row['predicted_outcome']}")
         prob_str = ", ".join([f"{outcome_map[cls]}: {row[f'prob_{outcome_map[cls]}']:.2%}" for cls in prob_classes])
         print(f"  Probabilities: {prob_str}")
         odds_str = ", ".join([f"{outcome_map[cls]}: {row[f'fair_odds_{outcome_map[cls]}']}" for cls in prob_classes if row[f'fair_odds_{outcome_map[cls]}'] is not None])
         print(f"  Fair Odds: {odds_str}\n")



# --- Main Execution Logic ---
if __name__ == "__main__":
    print("Starting Bundesliga Prediction Pipeline...")
    start_time = datetime.now()

    # --- Step 1: Get Raw Data ---
    df_raw = None
    if os.path.exists(BUNDESLIGA_RAW_CSV):
        print(f"Found existing raw data file: {BUNDESLIGA_RAW_CSV}")
        # Optional: Check if it needs updating (e.g., based on date modified or if future season requested)
        # For simplicity, we'll just load it. Re-run fetch manually by deleting the file if needed.
        try:
            df_raw = pd.read_csv(BUNDESLIGA_RAW_CSV)
            # Ensure date column is parsed correctly after loading
            df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
            print("Loaded raw data from CSV.")
        except Exception as e:
            print(f"Error loading raw data CSV: {e}. Attempting to fetch from API.")
            df_raw = None # Reset df_raw so fetch is triggered

    if df_raw is None:
         # Combine historical and upcoming seasons for a single fetch operation
        all_seasons = list(HISTORICAL_SEASONS) + list(UPCOMING_SEASONS)
        # Remove duplicates and sort
        all_seasons = sorted(list(set(all_seasons)))
        df_raw = fetch_all_data(all_seasons, BUNDESLIGA_RAW_CSV)


    # --- Step 2: Calculate Features ---
    df_features = None
    if df_raw is not None: # Proceed only if raw data is available
        if os.path.exists(BUNDESLIGA_FEATURES_CSV):
            print(f"Found existing feature file: {BUNDESLIGA_FEATURES_CSV}")
             # Optional: Add logic to check if features need recalculation (e.g., new raw data added)
             # For now, just load if exists. Delete the file to force recalculation.
            try:
                df_features = pd.read_csv(BUNDESLIGA_FEATURES_CSV)
                # Ensure date column is parsed correctly after loading
                df_features['date'] = pd.to_datetime(df_features['date'], errors='coerce')
                print("Loaded features from CSV.")
            except Exception as e:
                 print(f"Error loading features CSV: {e}. Attempting to recalculate.")
                 df_features = None # Reset df_features

        if df_features is None:
            df_features = calculate_features(df_raw)
    else:
        print("Skipping feature calculation because raw data could not be loaded or fetched.")


    # --- Step 3 & 4: Prepare Data and Train Model ---
    model = None
    df_predict = pd.DataFrame() # Initialize empty df for prediction data

    if df_features is not None: # Proceed only if features are available
        print("Separating historical (training) and future (prediction) data...")
        # Future matches are those where the target variable is NaN (score was unknown)
        # Or filter by date compared to now, but NaN score is more direct from the data processing
        df_train = df_features[df_features[TARGET_VARIABLE].notna()].copy()
        df_predict = df_features[df_features[TARGET_VARIABLE].isna()].copy()

        print(f"Found {len(df_train)} matches for training.")
        print(f"Found {len(df_predict)} matches for prediction.")

        if not df_train.empty:
            # Train the model using only the historical data
             model = train_model(df_train, FEATURES, TARGET_VARIABLE, MODEL_FILE, force_retrain=FORCE_RETRAIN)
        else:
            print("No historical data with outcomes found. Cannot train model.")
    else:
        print("Skipping model training because features could not be loaded or calculated.")

    # --- Step 5: Predict Upcoming Matches ---
    if model is not None and not df_predict.empty:
        predict_upcoming_matches(model, df_predict, FEATURES, PREDICTIONS_CSV)
    elif model is None:
         print("Skipping prediction because the model was not loaded or trained.")
    elif df_predict.empty:
         print("Skipping prediction because no future matches were found in the feature data.")

    # --- Finish ---
    end_time = datetime.now()
    print(f"\nPipeline finished in: {end_time - start_time}")