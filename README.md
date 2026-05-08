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
