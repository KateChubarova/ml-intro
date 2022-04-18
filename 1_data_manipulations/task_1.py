import sqlite3
import pandas as pd
import os
import calendar

# 1
pd.set_option('display.max_column', 100)
db = sqlite3.connect(os.environ.get("DB_PATH") or 'database.sqlite')
player_data = pd.read_sql("SELECT * FROM Player;", db)

# 2
players_180_190 = len(player_data[(player_data['height'] >= 180) & (player_data['height'] <= 190)])
size = player_data[(player_data['height'] >= 180) & (player_data['height'] <= 190)].sum()

# 3
player_data["birthday"] = pd.to_datetime(player_data["birthday"])
player_data['year'] = player_data['birthday'].apply(lambda birthday: birthday.year)
players_1980 = len(player_data[player_data["year"] == 1980])
highest_players = player_data.sort_values(by=['weight', 'player_name'], ascending=[False, True])['player_name'].head(
    10).tolist()
players_1980_1990 = player_data[(player_data['year'] >= 1980) & (player_data['year'] <= 1990)] \
    .groupby('year').year.count()
years_born_players = list(zip(players_1980_1990.index, players_1980_1990))

# 4
player_data['is_adriano'] = player_data.player_name.map(lambda player_name: player_name.startswith("Adriano"))
adrianos = player_data[player_data['is_adriano'] == True]
player_data['dayofweek'] = player_data['birthday'].apply(lambda birthday: birthday.dayofweek)
days = player_data.dayofweek.value_counts()
id = player_data.dayofweek.value_counts().idxmin()
dow_with_min_players_born = calendar.day_name[id]

# 5
league_data = pd.read_sql("SELECT * FROM  League;", db)
match_data = pd.read_sql("SELECT * FROM Match;", db)
league_data = league_data.rename(columns={'id': 'league_id'})
lm_data = league_data.join(match_data.set_index('league_id'), on='league_id', how='left', lsuffix='left',
                           rsuffix='right')
data = lm_data.groupby(['name']).league_id.agg([len]).sort_values(by=['len', 'name'], ascending=[False, True])

# 8
player_match_data = match_data.loc[:, 'home_player_1':'away_player_11']
counts_data = player_match_data.apply(pd.Series.value_counts)
counts_data['sum'] = counts_data.sum(axis=1)
counts_data.index.name = 'player_api_id'
plm_data = player_data.join(counts_data, on='player_api_id', how='left', lsuffix='left',
                            rsuffix='right')

# 11
team_data = pd.read_sql("SELECT * FROM Team;", db)
league_match_data = league_data.join(match_data.set_index('league_id'), on='league_id', how='left', lsuffix='left',
                                     rsuffix='right')
sorted_match_data = league_match_data[
    (league_match_data['season'] == '2008/2009') & (league_match_data['name'] == 'Germany 1. Bundesliga')]
sorted_match_data = sorted_match_data.groupby(['home_team_api_id']).home_team_api_id.agg([len])
sorted_match_data = sorted_match_data.rename(columns={'home_team_api_id': 'team_api_id'})
team_match_data = team_data.join(sorted_match_data, on=['team_api_id'], how='left')
row = team_match_data.loc[team_match_data['team_long_name'] == 'Borussia Dortmund']
borussia_bundesliga_2008_2009_matches = int(row.iloc[0].len)

# 12
# Find a team having the most matches (both home and away!)
# in the Germany 1. Bundesliga in 2008/2009 season. Return number of matches.
team_match_data = team_match_data.sort_values(by='len')

# 13
# Count total number of Arsenal matches (both home and away!)
# in the 2015/2016 season which they have won.
# Note: Winning a game means scoring more goals than an opponent.
game_match_data = match_data[
    (match_data['home_team_goal'] > match_data['away_team_goal']) & (match_data['season'] == '2015/2016')]

changed_team_data = team_data.rename(columns={'team_api_id': 'team_id'})
game_match_data = game_match_data.rename(columns={'home_team_api_id': 'team_id'})
gm_team_data = changed_team_data.join(game_match_data.set_index('team_id'), on='team_id', how='left', lsuffix='left',
                                      rsuffix='right')
gm_team_data = gm_team_data[gm_team_data['team_long_name'] == 'Arsenal']

num = len(gm_team_data['home_team_goal'])

game_match_data = match_data[
    (match_data['home_team_goal'] < match_data['away_team_goal']) & (match_data['season'] == '2015/2016')]

changed_team_data = team_data.rename(columns={'team_api_id': 'team_id'})
game_match_data = game_match_data.rename(columns={'away_team_api_id': 'team_id'})
gm_team_data = changed_team_data.join(game_match_data.set_index('team_id'), on='team_id', how='left', lsuffix='left',
                                      rsuffix='right')
gm_team_data = gm_team_data[gm_team_data['team_long_name'] == 'Arsenal']

num = len(gm_team_data['home_team_goal']) + num

# 9
player_attributes_data = pd.read_sql("SELECT * FROM Player_Attributes;", db)
player_attributes_data = player_attributes_data.drop(
    columns=['id', 'player_fifa_api_id', 'player_api_id', 'date', 'preferred_foot', 'attacking_work_rate',
             'defensive_work_rate'])
corr_matrix = player_attributes_data.corr(method='pearson').abs().unstack().sort_values(ascending=False)
corr_matrix = corr_matrix[corr_matrix < 1]
matrix = corr_matrix.reset_index()

result = matrix[0:10:2].to_numpy()
answer = []
for i in result:
    answer.append((i[0], i[1]))

