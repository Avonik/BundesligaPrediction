# Bundesliga Match Outcome Predictor


This project provides a Python pipeline to fetch German Bundesliga match data, calculate team Elo ratings and other statistical features, train a machine learning model, evaluate its performance, and predict the outcomes (Home Win, Draw, Away Win) of future matches.

The pipeline processes data chronologically, ensuring that predictions and evaluations realistically simulate forecasting future events based only on past information. It includes mechanisms to cache downloaded data and calculated features to avoid redundant computations on subsequent runs.

## Features

* **Data Fetching:** Retrieves historical and upcoming match data (scores, teams, dates) for specified Bundesliga seasons from the free [OpenLigaDB API](https://www.openligadb.de/).
* **Elo Rating Calculation:** Implements a chronological Elo rating system for teams, updating ratings after each match.
* **Feature Engineering:** Calculates relevant features for modeling, including:
    * Pre-match Elo ratings (home, away, difference)
    * Team form based on recent matches (points, goal difference)
    * Head-to-Head (H2H) statistics between teams
    * Points Per Game (PPG) within the current season
    * and more
* **Caching:** Checks for existing data/feature files to prevent unnecessary downloads and recalculations.
* **Model Training:** Trains a model using scikit-learn.
* **Prediction:** Predicts the outcome and associated probabilities for future matches.
* **Persistence:** Saves the fetched raw data, calculated features, Elo history, the trained model, and the final predictions.


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Avonik/BundesligaPrediction.git
    cd BundesligaPrediction
    ```
2.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configuration:**
    * Open the main Python script (all_in_onV2.py)
    * Modify the variables in the `--- Configuration ---` section at the top of the script as needed:
        * `HISTORICAL_SEASONS`, `UPCOMING_SEASONS`: Define the years for data fetching.
        * `BUNDESLIGA_RAW_CSV`, `BUNDESLIGA_FEATURES_CSV`, etc.: Adjust output filenames if desired.
        * `ELO_K_FACTOR`, `HOME_ADVANTAGE_ELO`, `FORM_WINDOW`, etc.: Fine-tune feature calculation parameters.
        * `FORCE_RETRAIN`: Set to `True` to force model retraining even if a saved model file exists.
2.  **Run the pipeline:**
    ```bash
    python all_in_onV2.py
    ```
3.  **Monitor Output:** The script will print progress updates to the console, indicating steps like data fetching, feature calculation, model training, evaluation, and prediction.
4.  **Check Results:** Once the script finishes, check the generated files in the project directory.
5.  opional: run app.py to visualise the results from the resulting csv on a local website
    ```bash
    python app.py
    ```

## Output Files

The script generates the following files:

* `bundesliga_matches_combined_v2.csv`: Raw match data fetched from the API, sorted chronologically.
* `bundesliga_features_elo_v2.csv`: The raw data enriched with calculated Elo ratings and other engineered features. Includes past (training/test) and future (prediction) matches.
* `elo_history_detailed_v2.csv`: A log of team Elo ratings before each match over time.
* `bundesliga_model_v2.joblib`: The serialized (saved) trained RandomForest model object.
* `bundesliga_predictions_v2.csv`: Predictions for upcoming matches, including predicted outcome, probabilities for Home Win/Draw/Away Win, and calculated fair odds.


## Note

* The project was commented and re-formatted by Google Gemini 2.5 pro for easier use.
* requirements.txt might be outdated.

