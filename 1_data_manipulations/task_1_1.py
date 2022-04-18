import sqlite3
import pandas as pd
import os
import numpy as np

# 1
pd.set_option('display.max_column', 100)
db = sqlite3.connect(os.environ.get("DB_PATH") or 'database.sqlite')

# 10
# team
player_data = pd.read_sql("SELECT * FROM Player;", db)
player_attributes_data = pd.read_sql("SELECT * FROM Player_Attributes;", db)
player_attributes_data["date"] = pd.to_datetime(player_attributes_data["date"])
data = player_attributes_data.groupby('player_api_id')['date'].max().reset_index()
merged = pd.merge(data, player_attributes_data, on=['player_api_id', 'date']).sort_values('player_api_id')

joined = player_data.join(merged.set_index('player_api_id'), on=['player_api_id'], how='left',
                          lsuffix='left', rsuffix='right')
joined = joined.drop(
    columns=['idright', 'date', 'preferred_foot',
             'attacking_work_rate',
             'defensive_work_rate', 'player_fifa_api_idleft', 'player_fifa_api_idright', 'idleft', 'birthday', 'height',
             'weight'])

joined = joined.set_index('player_api_id')
neymar = joined[joined['player_name'] == 'Neymar'].drop(columns=['player_name']).iloc[0]
joined = joined.drop(columns=['player_name'])

print(joined)
joined['euclidean'] = 0


def euclidean(row):
    row.euclidean = np.sqrt(np.sum([(a - b) * (a - b) for a, b in zip(row, neymar)]))
    return row


joined = joined.apply(euclidean, axis='columns')

joined = joined.join(player_data.set_index('player_api_id'), on=['player_api_id'], how='left',
                     lsuffix='left', rsuffix='right')

joined = joined.sort_values(['euclidean', 'player_name'], ascending=[True, True])

answer = joined['player_name'][1:6].tolist()

print(answer)
