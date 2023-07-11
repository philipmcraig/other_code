# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:47:08 2023

@author: Philip
"""

import pylab as pl
import pandas as pd
import glob as glob
import datetime

json_files = glob.glob('C://Users//phili//Documents//spotify_stream_history_jun23//Streaming_History_Audio_*.json')

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

#df_all = df_all.drop(df_all[df_all.ms_played == 0].index)
# perhaps remove ms_played for other small values, e.g. <=1000
df_all = df_all.drop(df_all[df_all.ms_played<=31000].index)
df_all = df_all.drop(df_all[df_all.skipped == True].index)
df_all = df_all.drop_duplicates(subset=['master_metadata_track_name','ts'])
# remove podcasts
df_all = df_all.drop(df_all[df_all.episode_name.notnull()].index)

# remove null entries of track name, album name, artist
df_all = df_all.drop(df_all[(df_all.master_metadata_album_artist_name.isnull()) &
                            (df_all.master_metadata_album_album_name.isnull()) &
                            (df_all.master_metadata_track_name.isnull())].index)

#df_all = df_all.sort(columns='ts')# sort(columns='ts')
    # find any repeats with small ms_played
#Air Fàir An Là # u in front of string for special characters

df_all = df_all.sort(columns=['master_metadata_album_artist_name',
                              'master_metadata_album_album_name',
                              'master_metadata_track_name'])

A = df_all.groupby(by=['master_metadata_track_name',
                       'master_metadata_album_album_name',
                       'ms_played']).aggregate({'master_metadata_track_name':'first',
                                               'master_metadata_album_album_name':'first',
                                               'ms_played':'max'})

df_dd = df_all.drop_duplicates(subset=['master_metadata_album_artist_name',
                                       'master_metadata_album_album_name',
                                       'master_metadata_track_name']).reset_index()

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
    sng_nm = df_dd.iloc[i].master_metadata_track_name
    abm_nm = df_dd.iloc[i].master_metadata_album_album_name
    ast_nm = df_dd.iloc[i].master_metadata_album_artist_name
    
    X = df_all[(df_all.master_metadata_track_name==sng_nm) & 
                (df_all.master_metadata_album_album_name==abm_nm) & 
                (df_all.master_metadata_album_artist_name==ast_nm)]
    
    if len(X) > 1 and len(X[X.ms_played<X.ms_played.max()/2]) > 0:
        inds_to_drop.append(X[X.ms_played<X.ms_played.max()/2].index)

inds_to_drop = [item for sublist in inds_to_drop for item in sublist]

B = df_all.drop(inds_to_drop)
#B.to_csv('C://Users//phili//Documents//spotify_stream_history_jun23//spotify_all_history.csv',
#         encoding='utf_32')

C = B[['master_metadata_album_artist_name','master_metadata_album_album_name',
       'master_metadata_track_name']]
C = C.rename(columns={'master_metadata_album_artist_name':'Artist',
                      'master_metadata_album_album_name':'Album',
                      'master_metadata_track_name':'Track Title'})

ms_list = B.ms_played.tolist()
C['Duration'] = [str(timedelta(seconds=round((i/1000.)))) for i in ms_list]

ts_list = B.ts.tolist()
C['Date Scrobbled'] = [str(i).replace('T',' ').replace('Z','')[:-3] for i in ts_list]

C['Album Artist'] = C.Artist

C = C[['Artist','Album','Track Title','Duration','Album Artist','Date Scrobbled']]

C.to_csv('C://Users//phili//Documents//spotify_stream_history_jun23//spotify_all_history_v2.csv',
         encoding='utf_32',index=False,header=False)