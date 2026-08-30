# NBA Playoff Prediction Model

This project uses machine learning to predict NBA playoff game and series outcomes based on team statistics from past seasons. It trains a logistic regression model on historical playoff matchups and uses current season stats to simulate game and series probabilities.

---

##  How It Works

The model:

1. Collects NBA team statistics using the `nba_api`
2. Builds training data from historical playoff seasons
3. Converts team stats into matchup differences
4. Trains a Logistic Regression model to predict win probability
5. Simulates best-of-7 playoff series using Monte Carlo simulation

---

## Metrics Used

The model uses advanced team and opponent stats:

- Offensive Rating
- Defensive Rating
- Net Rating
- Pace
- Opponent FG%
- Opponent 3PT%
- Opponent Rebounds
- Opponent Assists
- Opponent Turnovers

## Setup
1. Clone the repo
2. Install Dependencies
```
pip install -r requirements.txt
```

## Usage
Run the scripts in this order:
1. **Fetch training data** - pulls historical playoff stats and builds the training dataset
```
python FetchData.py
```
2.  **Train the model** - trains the logistic regression model on the fetched data and saves it
```
python predictionModel.py
```
3. **Prediict a series** - loads the trained model and simulates a matchup
```
python predictSeries.py
```
You'll be prompted for two team names and a season (e.g 2025-26).

## Example output
```
Enter Team A: 76ers
Enter Team B: Lakers
Enter season (example 2025-26): 2025-26

Fetching stats...
76ers vs Lakers

Stat Differences:

OFF_RATING           -2.70
DEF_RATING           -1.10
NET_RATING           -1.60
PACE                 1.17
OPP_FG_PCT           -0.02
OPP_FG3_PCT          0.00
OPP_REB              354.00
OPP_AST              10.00
OPP_TOV              67.00

Single Game Win Probability:
76ers: 49.2%
Lakers: 50.8%

Series Win Probability:
76ers: 49.1%
Lakers: 50.9%
```
