import time
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguedashteamstats
)

SEASONS = [
    '2013-14',
    '2014-15',
    '2015-16',
    '2016-17',
    '2017-18',
    '2018-19',
    '2019-20',
    '2020-21',
    '2021-22',
    '2022-23',
    '2023-24'
]

FEATURES = [
    'OFF_RATING',
    'DEF_RATING',
    'NET_RATING',
    'PACE',
    'OPP_FG_PCT',
    'OPP_FG3_PCT',
    'OPP_REB',
    'OPP_AST',
    'OPP_TOV',
]


def print_all_teams():
    all_teams = teams.get_teams()

    for team in sorted(all_teams, key=lambda t: t['full_name']):
        print(f"{team['full_name']} — {team['abbreviation']}")


def get_team_id(team_name: str) -> str:
    all_teams = teams.get_teams()

    match = [
        t for t in all_teams
        if team_name.lower() in t['full_name'].lower()
        or team_name.upper() == t['abbreviation']
    ]

    if not match:
        raise ValueError(f"No team found for '{team_name}'")

    return match[0]['id']


def get_current_team_stats(
        team_id: str,
        season: str = '2025-26'
) -> pd.Series:
    """
    Fetch current season regular season stats.
    """

    adv = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Advanced'
    ).get_data_frames()[0]

    time.sleep(1)

    opp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star='Regular Season',
        measure_type_detailed_defense='Opponent'
    ).get_data_frames()[0]

    time.sleep(1)

    adv_team = adv[adv['TEAM_ID'] == int(team_id)]
    opp_team = opp[opp['TEAM_ID'] == int(team_id)]

    if adv_team.empty or opp_team.empty:
        raise ValueError("No current stats found.")

    adv_stats = adv_team[
        [
            'OFF_RATING',
            'DEF_RATING',
            'NET_RATING',
            'PACE'
        ]
    ].iloc[0]

    opp_stats = opp_team[
        [
            'OPP_FG_PCT',
            'OPP_FG3_PCT',
            'OPP_REB',
            'OPP_AST',
            'OPP_TOV'
        ]
    ].iloc[0]

    return pd.concat([adv_stats, opp_stats])


def fetch_training_data() -> pd.DataFrame:
    """
    Create matchup-based playoff training data.
    """

    all_matchups = []

    for season in SEASONS:

        print(f"\nFetching {season} playoff data...")

        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Playoffs',
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]

        time.sleep(1)

        opp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Playoffs',
            measure_type_detailed_defense='Opponent'
        ).get_data_frames()[0]

        time.sleep(1)

        merged = adv[
            [
                'TEAM_ID',
                'TEAM_NAME',
                'OFF_RATING',
                'DEF_RATING',
                'NET_RATING',
                'PACE'
            ]
        ].merge(
            opp[
                [
                    'TEAM_ID',
                    'OPP_FG_PCT',
                    'OPP_FG3_PCT',
                    'OPP_REB',
                    'OPP_AST',
                    'OPP_TOV'
                ]
            ],
            on='TEAM_ID'
        )

        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable='Playoffs'
        ).get_data_frames()[0]

        time.sleep(1)

        win_rate = gf.groupby('TEAM_ID').apply(
            lambda x: (x['WL'] == 'W').mean()
        ).reset_index()

        win_rate.columns = ['TEAM_ID', 'WIN_RATE']

        merged = merged.merge(win_rate, on='TEAM_ID')

        teams_data = merged.to_dict('records')

        # Create pairwise matchups
        for i in range(len(teams_data)):
            for j in range(i + 1, len(teams_data)):

                team_a = teams_data[i]
                team_b = teams_data[j]

                matchup = {}

                for feature in FEATURES:
                    matchup[feature] = (
                        team_a[feature] - team_b[feature]
                    )

                matchup['RESULT'] = int(
                    team_a['WIN_RATE'] > team_b['WIN_RATE']
                )

                all_matchups.append(matchup)

                # Reverse matchup
                reverse = {}

                for feature in FEATURES:
                    reverse[feature] = (
                        team_b[feature] - team_a[feature]
                    )

                reverse['RESULT'] = int(
                    team_b['WIN_RATE'] > team_a['WIN_RATE']
                )

                all_matchups.append(reverse)

    df = pd.DataFrame(all_matchups)

    df.to_csv(
        'data/playoff_training_data.csv',
        index=False
    )

    print(f"\nSaved {len(df)} matchup rows.")

    return df


if __name__ == "__main__":

    print("What would you like to do?")
    print("1. Fetch training data")
    print("2. Print all teams")

    choice = input("\nEnter choice (1/2): ").strip()

    if choice == '1':
        fetch_training_data()

    elif choice == '2':
        print_all_teams()

    else:
        print("Invalid choice.")