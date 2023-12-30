# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:47:08 2023

@author: Philip
"""

import pylab as pl
import pandas as pd
import glob as glob
from datetime import date, timedelta

json_files = glob.glob('C://Users//phili//Documents//spotify_stream_history_dec23//StreamingHistory*.json')

old_files = glob.glob('C://Users//phili//Documents//spotify_stream_history_jun23//Streaming_History_Audio*')

df_json = pd.read_json(old_files[0])
old_col_names = df_json.keys()

#data = pd.read_json('C://Users//phili//Downloads//my_spotify_data_all//MyData//endsong_1.json')
#
#mask = data['ts'] < '2019-04-03T15:39:00Z'
#
#df = data[mask]

hold = []

for i in range(len(json_files)):
    df_json = pd.read_json(json_files[i])
    #mask = (df_json['ts'] > '2019-04-03T08:00:00Z') & (df_json['ts'] < '2019-04-04T01:00:00Z')
    df_json.endTime = pd.to_datetime(df_json['endTime'])
    #df_json.endTime = df_json['endTime'].dt.date#strftime('%Y-%m-%d %H:%M')
    mask = (pl.datetime64(date(2023,6,6)) < df_json['endTime']) &\
                        (pl.datetime64(date(2023,7,10)) > df_json['endTime'])
    df_out = df_json[mask]
    hold.append(df_out)

df_all = pd.concat([hold[0],hold[1],hold[2]],ignore_index=True)

#df_all = df_all.drop(df_all[df_all.ms_played == 0].index)
# perhaps remove ms_played for other small values, e.g. <=1000
df_all = df_all.drop(df_all[df_all.msPlayed<=31000].index)
#df_all = df_all.drop(df_all[df_all.skipped == True].index)
#df_all = df_all.drop_duplicates(subset=['master_metadata_track_name','ts'])
# remove podcasts
#df_all = df_all.drop(df_all[df_all.episode_name.notnull()].index)

# remove null entries of track name, album name, artist
df_all = df_all.drop(df_all[(df_all.artistName.isnull()) &
                            (df_all.trackName.isnull())].index)

#df_all = df_all.sort(columns='ts')# sort(columns='ts')
    # find any repeats with small ms_played
#Air Fàir An Là # u in front of string for special characters

df_all = df_all.sort_values(by=['artistName','trackName'])

A = df_all.groupby(by=['trackName',
                       'msPlayed']).aggregate({'trackName':'first',
                                               'msPlayed':'max'})

df_dd = df_all.drop_duplicates(subset=['artistName',
                                       'trackName']).reset_index()

inds_to_drop = []
#for i in range(len(df_all)):
#    abm_nm = df_all.iloc[i].master_metadata_album_album_name
#    sng_nm = df_all.iloc[i].master_metadata_track_name
#    
#    max_ms = A[(A.master_metadata_album_album_name==abm_nm) & 
#                        (A.master_metadata_track_name==sng_nm)].ms_played.max()
#    
#    if df_all.iloc[i].ms_played < max_ms/2:
#        inds_to_drop.append(df_all.iloc[i].name)

for i in range(len(df_dd)):
    sng_nm = df_dd.iloc[i].trackName
    #abm_nm = df_dd.iloc[i].master_metadata_album_album_name
    ast_nm = df_dd.iloc[i].artistName
    
    X = df_all[df_all.trackName==sng_nm] 
               # (df_all.master_metadata_album_artist_name==ast_nm)]
    
    if len(X) > 1 and len(X[X.msPlayed<X.msPlayed.max()/2]) > 0:
        inds_to_drop.append(X[X.msPlayed<X.msPlayed.max()/2].index)

inds_to_drop = [item for sublist in inds_to_drop for item in sublist]

B = df_all.drop(inds_to_drop)
#B.to_csv('C://Users//phili//Documents//spotify_stream_history_jun23//spotify_all_history.csv',
#         encoding='utf_32')

C = B[['artistName','trackName']]
C = C.rename(columns={'artistName':'Artist',
                      'trackName':'Track Title'})

ms_list = B.msPlayed.tolist()
C['Duration'] = [str(timedelta(seconds=round((i/1000.)))) for i in ms_list]

ts_list = B.endTime.tolist()
C['Date Scrobbled'] = [str(i).replace('T',' ').replace('Z','')[:-3] for i in ts_list]

C['Album Artist'] = C.Artist

C = C[['Date Scrobbled','Artist','Track Title','Duration']]
C = C.rename(columns={'Date Scrobbled':old_col_names[0],
                      'Artist':old_col_names[8],
                      'Track Title':old_col_names[7],
                      'Duration':old_col_names[3]})

# columns not in C DataFrame
not_used = list(set(old_col_names) - set(C.keys()))

for i in range(len(not_used)):
    C[not_used[i]] = pl.nan

C['username'] = 'the_black_wizard'

C.to_json('C://Users//phili/Documents/spotify_stream_history_dec23/Streaming_History_Audio_2023_0.json',
          lines=True,orient='records',force_ascii=False)
#C.to_csv('C://Users//phili//Documents//spotify_stream_history_jun23//spotify_all_history_v3.csv',
 #        encoding='utf_32_be',index=False,header=False)