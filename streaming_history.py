# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:47:08 2023

@author: Philip
"""

import pylab as pl
import pandas as pd
import glob as glob

json_files = glob.glob('C://Users//phili//Downloads//my_spotify_data_all//MyData//endsong_*.json')

#data = pd.read_json('C://Users//phili//Downloads//my_spotify_data_all//MyData//endsong_1.json')
#
#mask = data['ts'] < '2019-04-03T15:39:00Z'
#
#df = data[mask]

hold = []

for i in range(len(json_files)):
    df_json = pd.read_json(json_files[i])
    #mask = (df_json['ts'] > '2019-04-03T08:00:00Z') & (df_json['ts'] < '2019-04-04T01:00:00Z')
    mask = df_json['ts'] < '2019-04-03T15:39:00Z'
    df_out = df_json[mask]
    hold.append(df_out)

df_all = pd.concat([hold[0],hold[1],hold[2],hold[3],hold[4],hold[5],
                    hold[6],hold[7]],ignore_index=True)

df_all = df_all.drop(df_all[df_all.ms_played == 0].index)
# perhaps remove ms_played for other small values, e.g. <=1000
df_all = df_all.drop(df_all[df_all.skipped == True].index)
df_all = df_all.drop_duplicates(subset=['master_metadata_track_name','ts'])

df_all = df_all.sort(columns='ts')# sort(columns='ts')
    # find any repeats with small ms_played
#Air Fàir An Là # u in front of string for special characters