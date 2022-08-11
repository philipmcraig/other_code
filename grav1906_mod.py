# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:55:15 2018

@author: qx911590
"""

import pylab as pl
import glob
import pandas as pd

homedir = '/home/users/qx911590/'
filenames = glob.glob(homedir+'weatherrescue/temp/*') # all filenames in dir
filenames = pl.asarray(filenames) # list into array
filenames = pl.sort(filenames) # get filenames in correct order
#filenames = pl.delete(filenames,330)
filenames = filenames[55:]

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book
         
SH = 9 # Sumburgh Head
N = [10,11,24,25,26,27,28] # Stornoway,Malin,Wick,Nairn,Aberdeen,Leith,Shields
S = [12,16,15,13,14,18,17,29,32,34,30,31,33,35,22,23] 
# Blacksod, Donaghadee, Birr, Valencia, Roches, Holyhead, Spurn, Nottingham,
# Clacton, Bath, London, Portland, Dungeness
SJ = [20,21] # Scilly, Jersey

for name in range(filenames.size): # loop over filenames
    df = pd.read_csv(filenames[name],header=None,names=names)
    # open as pandas dataframe because numpy/genfromtxt don't like '#N/A'
    #year = filenames[name][44:48]

    logs = pl.array(df) # dataframe into array
    b1 = pl.where(logs=='  '); logs[b1] = pl.float32('nan')
    b2 = pl.where(logs==' '); logs[b2] = pl.float32('nan')
    
    logs[SH,1:] = logs[SH,1:].astype(float)
    logs[SH,[1,3]] = logs[SH,[1,3]] - 0.04
        
    logs[N,1:] = logs[N,1:].astype(float)
    for i in range(len(N)):
        logs[N[i],[1,3]] = logs[N[i],[1,3]] - 0.03
    
    logs[S,1:] = logs[S,1:].astype(float)
    for i in range(len(S)):
        logs[S[i],[1,3]] = logs[S[i],[1,3]] - 0.02
    
    logs[SJ,1:] = logs[SJ,1:].astype(float)
    for i in range(len(SJ)):
        logs[SJ[i],[1,3]] = logs[SJ[i],[1,3]] - 0.01
    
    #logs[b1] = ' '; logs[b2] = ' '
    
    dwr = pd.DataFrame(logs)
    dwr = dwr.replace(pl.NaN,' ')
    
    #day = "{0:0=2d}".format(day)
    dwr.to_csv(filenames[name],header=None,index=None)