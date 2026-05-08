import joblib
import numpy as np

from FetchData import (
    FEATURES,
    get_team_id,
    get_current_team_stats
)


def simulate_series(
        team_a: str,
        team_b: str,
        prob_a: float,
        simulations: int = 10000
):

    a_wins_series = 0

    for _ in range(simulations):

        a_wins = 0
        b_wins = 0

        while a_wins < 4 and b_wins < 4:

            if np.random.random() < prob_a:
                a_wins += 1
            else:
                b_wins += 1

        if a_wins == 4:
            a_wins_series += 1

    prob_series_a = a_wins_series / simulations

    print(f"\nSeries Win Probability:")
    print(f"{team_a}: {prob_series_a:.1%}")
    print(f"{team_b}: {(1 - prob_series_a):.1%}")


def predict_series(
        team_a_name: str,
        team_b_name: str,
        season: str = '2025-26'
):

    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

    print(f"\nFetching stats...")
    print(f"{team_a_name} vs {team_b_name}\n")

    stats_a = get_current_team_stats(
        get_team_id(team_a_name),
        season
    )

    stats_b = get_current_team_stats(
        get_team_id(team_b_name),
        season
    )

    diff = (
        stats_a - stats_b
    ).values.reshape(1, -1)

    print("Stat Differences:\n")

    for feature, value in zip(FEATURES, diff.flatten()):
        print(f"{feature:<20} {value:.2f}")

    diff_scaled = scaler.transform(diff)

    prob_a = model.predict_proba(
        diff_scaled
    )[0][1]

    # Prevent unrealistic certainty
    prob_a = max(0.05, min(0.95, prob_a))

    prob_b = 1 - prob_a

    print(f"\nSingle Game Win Probability:")
    print(f"{team_a_name}: {prob_a:.1%}")
    print(f"{team_b_name}: {prob_b:.1%}")

    simulate_series(
        team_a_name,
        team_b_name,
        prob_a
    )


if __name__ == "__main__":

    team_a = input(
        "Enter Team A: "
    )

    team_b = input(
        "Enter Team B: "
    )

    season = input(
        "Enter season (example 2025-26): "
    ).strip()

    predict_series(
        team_a,
        team_b,
        season
    )