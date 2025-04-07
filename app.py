import pandas as pd
from flask import Flask, render_template_string, url_for
import os
import datetime
from collections import defaultdict
import math # For checking NaN

app = Flask(__name__)

# --- Configuration ---
CSV_FILE = 'bundesliga_predictions_v2.csv'
LOGO_FOLDER = 'static/img'
# Define a mapping if team names in CSV differ from logo filenames
TEAM_LOGO_MAP = {}
# --- End Configuration ---



def format_team_name_for_logo(team_name):
    import pandas as pd
    """Attempts to find the correct logo filename relative to the static/img folder."""
    if not isinstance(team_name, str):
        return None
    if team_name in TEAM_LOGO_MAP:
        # Ensure the mapped name becomes a relative path
        return os.path.join('img', TEAM_LOGO_MAP[team_name]).replace("\\", "/")

    # Default: Try common variations
    potential_filenames = [
        f"{team_name}.png", f"{team_name.lower()}.png",
        f"{team_name.replace(' ', '_')}.png", f"{team_name.lower().replace(' ', '_')}.png",
        f"{team_name}.jpg", f"{team_name.lower()}.jpg",
        f"{team_name}.svg", f"{team_name.lower()}.svg",
        f"{team_name}.webp", f"{team_name.lower()}.webp", # Added webp
    ]
    logo_dir = os.path.join(app.static_folder, 'img') # Absolute path for checking existence

    for filename in potential_filenames:
        full_path = os.path.join(logo_dir, filename)
        if os.path.exists(full_path):
            # Return the relative path needed by url_for('static', filename=...)
            return os.path.join('img', filename).replace("\\", "/")

    # print(f"Warning: Logo not found for team '{team_name}'. Checked in '{logo_dir}'.") # Less verbose logging
    return None

def load_predictions(csv_path):
    """Loads, processes, and groups prediction data by matchday."""
    try:

        df = pd.read_csv(csv_path)

        # --- Essential Columns ---
        prob_cols = ['prob_Home Win', 'prob_Draw', 'prob_Away Win']
        odds_cols = ['fair_odds_Home Win', 'fair_odds_Draw', 'fair_odds_Away Win']
        # **** Crucially requires 'matchday' column ****
        required_cols = ['date', 'matchday', 'team_home', 'team_away'] + prob_cols + odds_cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns in CSV: {', '.join(missing_cols)}")
            return {}, [] # Return empty data and empty matchday list

        # --- Data Cleaning and Type Conversion ---
        # Convert probabilities and odds first
        for col in prob_cols + odds_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert matchday - try numeric first, then string
        try:
            df['matchday'] = pd.to_numeric(df['matchday'], errors='coerce').fillna(-1).astype(int) # Use -1 for failed conversions
        except Exception:
            df['matchday'] = df['matchday'].astype(str) # Fallback to string if numeric fails

        # Drop rows with invalid essential data AFTER potential conversions
        df.dropna(subset=['date', 'matchday', 'team_home', 'team_away'] + prob_cols, inplace=True)
        df = df[df['matchday'] != -1] # Remove rows where matchday conversion failed if it was numeric
        # Fill missing odds with NaN
        df[odds_cols] = df[odds_cols].fillna(float('nan'))


        # --- Date/Time Formatting ---
        try:
            df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['parsed_date'], inplace=True)
            # Simple date format, time is separate if exists
            df['formatted_date'] = df['parsed_date'].dt.strftime('%a, %b %d') # e.g., Sat, Apr 27

            if 'time' in df.columns:
                 df['formatted_time'] = df['time'].astype(str) # Assuming time is already formatted ok
            else:
                 df['formatted_time'] = None

            # Sort primarily by matchday, then by date/time within matchday
            sort_cols = ['matchday', 'parsed_date']
            if 'formatted_time' in df.columns:
                # Attempt to parse time for sorting if possible, fallback otherwise
                try:
                    df['sort_time'] = pd.to_datetime(df['formatted_time'], format='%H:%M', errors='coerce').dt.time
                    sort_cols.append('sort_time')
                except Exception:
                    pass # If time format is inconsistent, just sort by date

            df.sort_values(by=sort_cols, inplace=True, na_position='last')

        except Exception as e:
            print(f"Warning: Could not fully process/sort date/time columns: {e}.")
            df['formatted_date'] = df['date'].astype(str) # Fallback
            df['formatted_time'] = None
            df.sort_values(by=['matchday'], inplace=True) # Sort only by matchday


        # --- Process and Group Data ---
        games_by_matchday = defaultdict(list)
        processed_records = []

        for record in df.to_dict('records'):
            # Probability processing
            ph, pdr, pa = record.get(prob_cols[0], 0), record.get(prob_cols[1], 0), record.get(prob_cols[2], 0)
            prob_sum = ph + pdr + pa
            if prob_sum > 0 and abs(prob_sum - 1.0) > 0.01:
                ph, pdr, pa = ph/prob_sum, pd/prob_sum, pa/prob_sum
            elif prob_sum <= 0:
                ph, pdr, pa = 1/3, 1/3, 1/3

            record[prob_cols[0]], record[prob_cols[1]], record[prob_cols[2]] = ph, pdr, pa

            # Determine predicted outcome based on highest probability
            if ph > pa and ph > pdr: record['predicted_winner'] = 'home'
            elif pa > ph and pa > pdr: record['predicted_winner'] = 'away'
            else: record['predicted_winner'] = 'draw' # Includes draw > home/away

            # Format odds
            for col in odds_cols:
                record[col] = f"{record[col]:.2f}" if not math.isnan(record[col]) else "N/A"

            # Get Logo Paths
            record['home_logo_path'] = format_team_name_for_logo(record['team_home'])
            record['away_logo_path'] = format_team_name_for_logo(record['team_away'])

            # Clean up unnecessary fields before adding
            record.pop('parsed_date', None)
            record.pop('sort_time', None)

            processed_records.append(record)
            games_by_matchday[record['matchday']].append(record)

        # Get sorted list of unique matchdays
        # Try numeric sort first, then string sort
        try:
            matchdays = sorted(list(games_by_matchday.keys()), key=int)
        except ValueError:
            matchdays = sorted(list(games_by_matchday.keys()))


        # Convert defaultdict to regular dict for Jinja2
        return dict(games_by_matchday), matchdays

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return {}, []
    except Exception as e:
        print(f"Error reading or processing CSV: {e}")
        import traceback
        traceback.print_exc()
        return {}, []

# --- Updated HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bundesliga Predictions - Matchday View</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f7f6; /* Light background */
            --card-bg-color: #ffffff; /* White card */
            --text-primary: #2c3e50; /* Dark blue-gray text */
            --text-secondary: #5a6a7a; /* Lighter gray text */
            --text-muted: #8a9aa9;
            --border-color: #e1e8ed; /* Light border */
            --shadow-color: rgba(44, 62, 80, 0.1);
            --highlight-color: #3498db; /* Blue for highlights */

            --home-color: #3498db;
            --draw-color: #95a5a6;
            --away-color: #e74c3c;

            --font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-family);
            margin: 0;
            padding: 0;
            line-height: 1.6;
            font-size: 16px;
        }

        .container {
            max-width: 900px; /* Wider layout */
            margin: 0 auto;
            padding: 30px 20px;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 5px;
            color: var(--text-primary);
        }

        header p {
            font-size: 1.1em;
            color: var(--text-secondary);
            font-weight: 300;
        }

        .controls {
            margin-bottom: 30px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .controls label {
            margin-right: 10px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .controls select {
            padding: 10px 15px;
            font-family: var(--font-family);
            font-size: 1em;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background-color: var(--card-bg-color);
            color: var(--text-primary);
            min-width: 150px;
            cursor: pointer;
            transition: border-color 0.2s ease;
        }
        .controls select:focus {
             border-color: var(--highlight-color);
             outline: none;
             box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        }

        .matchday-group {
            display: none; /* Initially hide all groups */
            animation: fadeIn 0.5s ease; /* Fade in animation */
        }
        .matchday-group.active {
            display: block; /* Show the active group */
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .game-entry {
            background-color: var(--card-bg-color);
            border-radius: 8px;
            margin-bottom: 20px;
            padding: 20px 25px;
            box-shadow: 0 3px 8px var(--shadow-color);
            border: 1px solid var(--border-color);
            transition: box-shadow 0.3s ease;
            overflow: hidden;
        }
         .game-entry:hover {
             box-shadow: 0 6px 12px var(--shadow-color);
         }

        .game-meta {
            font-size: 0.9em;
            color: var(--text-muted);
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px dashed var(--border-color);
            padding-bottom: 10px;
        }
        .game-meta span { font-weight: 500; } /* Make date/time slightly bolder */

        .match-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
        }

        .team-display {
            flex: 1;
            display: flex;
            align-items: center;
            font-size: 1.2em;
            font-weight: 600;
        }
        .team-display.home { justify-content: flex-start; text-align: left; }
        .team-display.away { justify-content: flex-end; text-align: right; }

        .team-display img {
            width: 35px; /* Smaller logos inline */
            height: 35px;
            object-fit: contain;
            margin: 0 12px; /* Space around logo */
            background-color: #eee; /* Light background for transparent logos */
            border-radius: 50%;
            border: 1px solid var(--border-color);
        }
         .team-display .logo-placeholder {
             width: 35px; height: 35px; border-radius: 50%;
             background-color: #e1e8ed; color: var(--text-muted);
             display: inline-flex; align-items: center; justify-content: center;
             font-size: 0.8em; font-weight: bold; margin: 0 12px;
         }

        .vs-separator {
            font-size: 1em;
            font-weight: 400;
            color: var(--text-muted);
            padding: 0 20px;
        }

        /* Highlight predicted winner */
        .team-display.predicted-winner {
             font-weight: 700;
             /* Optional: color the text slightly */
             /* color: var(--highlight-color); */
        }
        .vs-separator.predicted-draw {
            font-weight: 700;
            color: var(--text-secondary);
        }

        .stats-row {
             display: flex;
             justify-content: space-between;
             align-items: center;
             background-color: #f8fafa; /* Slightly different background for stats */
             margin: 15px -25px -20px -25px; /* Extend to card edges */
             padding: 15px 25px;
             border-top: 1px solid var(--border-color);
        }

        .probabilities {
             display: flex;
             align-items: center;
             flex-grow: 1; /* Take available space */
        }
        .prob-bar {
            height: 12px; /* Slimmer bar */
            display: flex;
            flex-grow: 1;
            border-radius: 6px;
            overflow: hidden;
            margin: 0 15px;
            background-color: #e1e8ed; /* Bar background */
        }
        .prob-segment {
            height: 100%;
            transition: width 0.5s ease; /* Smooth transition (though width set directly) */
            /* Gradient removed for cleaner look */
        }
        .prob-segment.home { background-color: var(--home-color); }
        .prob-segment.draw { background-color: var(--draw-color); }
        .prob-segment.away { background-color: var(--away-color); }

        .prob-value {
            font-size: 1.1em;
            font-weight: 600;
            min-width: 45px;
            text-align: center;
        }
        .prob-value.home { color: var(--home-color); }
        .prob-value.away { color: var(--away-color); }

        .odds {
            font-size: 0.9em;
            color: var(--text-secondary);
            text-align: right;
            white-space: nowrap; /* Prevent wrapping */
            margin-left: 20px; /* Space between probs and odds */
        }
         .odds span {
             margin-left: 10px; /* Space between odds labels */
         }
         .odds strong {
              font-weight: 600;
              color: var(--text-primary); /* Make odds numbers clearer */
         }

         /* Responsive */
         @media (max-width: 768px) {
             .container { max-width: 100%; padding: 20px 15px; }
             h1 { font-size: 1.8em; }
             header p { font-size: 1em; }
             .team-display { font-size: 1.1em; }
             .vs-separator { padding: 0 10px; }
             .stats-row { flex-direction: column; align-items: stretch; text-align: center; padding: 15px;}
             .probabilities { margin-bottom: 10px; }
             .odds { margin-left: 0; text-align: center; margin-top: 10px; }
             .odds span { margin: 0 8px; }
         }
         @media (max-width: 480px) {
             body { font-size: 14px; }
             h1 { font-size: 1.6em; }
             .controls { flex-direction: column; align-items: stretch; }
             .controls label { margin-bottom: 5px; text-align: center; margin-right: 0;}
             .controls select { width: 100%; }
             .game-entry { padding: 15px; }
             .team-display { font-size: 1em; flex-direction: column; } /* Stack logo and name */
             .team-display img, .team-display .logo-placeholder { margin: 0 0 5px 0; width: 30px; height: 30px; }
             .vs-separator { padding: 10px 0; } /* Give VS more vertical space */
             .match-row { flex-direction: column; } /* Stack teams vertically */
             .team-display.home, .team-display.away { justify-content: center; text-align: center; margin-bottom: 5px;}
             .stats-row { margin: 15px -15px -15px -15px; padding: 10px 15px;}
             .prob-bar { margin: 0 10px; }
             .prob-value { font-size: 1em; }
             .odds span { display: inline-block; margin: 0 5px; } /* Keep odds inline if possible */
         }

    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Bundesliga Predictions</h1>
            <p>Select a matchday to view predictions and fair odds</p>
        </header>

        {% if not matchdays %}
            <p style="text-align:center; color: var(--text-secondary);">Could not load prediction data or find matchdays. Please check the CSV file ('{{ csv_filename }}') and ensure it contains a 'matchday' column.</p>
        {% else %}
            <div class="controls">
                <label for="matchday-select">Select Matchday:</label>
                <select id="matchday-select">
                    {% for md in matchdays %}
                        <option value="{{ md }}" {% if md == latest_matchday %}selected{% endif %}>
                            Matchday {{ md }}
                        </option>
                    {% endfor %}
                </select>
            </div>

            {% for md in matchdays %}
                <div class="matchday-group" id="matchday-{{ md }}" {% if md == latest_matchday %}style="display: block;"{% endif %}>
                     {# Optional: Title for each group if needed #}
                    {% for game in games_by_matchday[md] %}
                        <div class="game-entry">
                            <div class="game-meta">
                                <span>{{ game.formatted_date }}</span>
                                {% if game.formatted_time %}
                                    <span>{{ game.formatted_time }}</span>
                                {% endif %}
                            </div>

                            <div class="match-row">
                                <div class="team-display home {{ 'predicted-winner' if game.predicted_winner == 'home' else '' }}">
                                    {% set logo_path = game.home_logo_path %}
                                    {% if logo_path and url_for('static', filename=logo_path) %}
                                        <img src="{{ url_for('static', filename=logo_path) }}" alt="{{ game.team_home }} Logo">
                                    {% else %}
                                        <span class="logo-placeholder">?</span>
                                    {% endif %}
                                    <span>{{ game.team_home }}</span>
                                </div>

                                <div class="vs-separator {{ 'predicted-draw' if game.predicted_winner == 'draw' else '' }}">VS</div>

                                <div class="team-display away {{ 'predicted-winner' if game.predicted_winner == 'away' else '' }}">
                                    <span>{{ game.team_away }}</span>
                                    {% set logo_path = game.away_logo_path %}
                                    {% if logo_path and url_for('static', filename=logo_path) %}
                                        <img src="{{ url_for('static', filename=logo_path) }}" alt="{{ game.team_away }} Logo">
                                    {% else %}
                                        <span class="logo-placeholder">?</span>
                                    {% endif %}
                                </div>
                            </div>

                            <div class="stats-row">
                                <div class="probabilities">
                                    <span class="prob-value home">{{ "%.0f%%" % (game['prob_Home Win'] * 100) }}</span>
                                    <div class="prob-bar">
                                        <div class="prob-segment home" style="width: {{ game['prob_Home Win'] * 100 }}%;"></div>
                                        <div class="prob-segment draw" style="width: {{ game['prob_Draw'] * 100 }}%;"></div>
                                        <div class="prob-segment away" style="width: {{ game['prob_Away Win'] * 100 }}%;"></div>
                                    </div>
                                    <span class="prob-value away">{{ "%.0f%%" % (game['prob_Away Win'] * 100) }}</span>
                                </div>
                                <div class="odds">
                                    Odds:
                                    <span>H: <strong>{{ game['fair_odds_Home Win'] }}</strong></span>
                                    <span>D: <strong>{{ game['fair_odds_Draw'] }}</strong></span>
                                    <span>A: <strong>{{ game['fair_odds_Away Win'] }}</strong></span>
                                </div>
                            </div>
                        </div>
                    {% endfor %}
                </div>
            {% endfor %}
        {% endif %}
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const selectElement = document.getElementById('matchday-select');
            const groups = document.querySelectorAll('.matchday-group');
            const latestMatchday = selectElement.value; // Get the initially selected value

            function showMatchday(matchdayValue) {
                groups.forEach(group => {
                    if (group.id === 'matchday-' + matchdayValue) {
                        group.style.display = 'block'; // Show selected group
                        group.classList.add('active'); // Mark as active (for potential CSS)
                    } else {
                        group.style.display = 'none'; // Hide others
                        group.classList.remove('active');
                    }
                });
            }

            // Initial setup: ensure only the selected one is visible
            // (redundant due to inline style, but good practice)
             showMatchday(latestMatchday);


            // Event listener for dropdown change
            selectElement.addEventListener('change', function() {
                showMatchday(this.value);
            });
        });
    </script>

</body>
</html>
"""

@app.route('/')
def index():
    games_grouped_by_matchday, matchdays_list = load_predictions(CSV_FILE)

    # Determine the latest matchday to show by default (last one in the sorted list)
    latest_md = matchdays_list[-1] if matchdays_list else None

    # Ensure the static/img folder exists
    logo_dir = os.path.join(app.static_folder, 'img')
    if not os.path.exists(logo_dir):
         print(f"Warning: Logo directory '{logo_dir}' not found. Creating it.")
         os.makedirs(logo_dir)

    return render_template_string(
        HTML_TEMPLATE,
        games_by_matchday=games_grouped_by_matchday,
        matchdays=matchdays_list,
        latest_matchday=latest_md,
        csv_filename=CSV_FILE
    )

if __name__ == '__main__':
    app.run(debug=True)