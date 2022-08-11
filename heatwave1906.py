# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:58:11 2018

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
import matplotlib.dates as mdates
import re

pl.close('all')

homedir = '/home/users/qx911590/'
filenames = glob.glob(homedir+'weatherrescue/csv_fixed/1906/*') # all filenames in dir
filenames = pl.asarray(filenames) # list into array
filenames = pl.sort(filenames)

aug15 = pl.where(filenames=='/home/users/qx911590/weatherrescue/csv_fixed/1906/dwr_1906_08_15.csv')
sep15 = pl.where(filenames=='/home/users/qx911590/weatherrescue/csv_fixed/1906/dwr_1906_09_15.csv')
aug15 = aug15[0][0]; sep15 = sep15[0][0]

filenames = filenames[aug15:sep15+1]
dates = pl.zeros([filenames.size],dtype='S5')
D = pl.zeros([filenames.size],dtype='S6')
for i in range(dates.size):
    d = filenames[i][59:-4]
    dates[i] = re.sub('_','/',d)
    month = datetime.date(1906, int(dates[i][:2]), 1).strftime('%b')
    day = datetime.date(1906, 1, int(dates[i][-2:])).strftime('%d')
    D[i] = month +  ' ' + day

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book

tmax = pl.zeros([filenames.size,27]); tmin = pl.zeros([filenames.size,27])
tdry = pl.zeros([filenames.size,27]); twet = pl.zeros([filenames.size,27])

for name in range(filenames.size): # loop over filenames
    df = pd.read_csv(filenames[name],header=None,names=names)
    # open as pandas dataframe because numpy/genfromtxt don't like '#N/A'

    logs = pl.array(df)
    space = pl.where(logs==' ')
    logs[space] = pl.float32('nan')
    
    tmax[name,:] = logs[9:36,6].astype(pl.float32)
    tmin[name,:] = logs[9:36,7].astype(pl.float32)
    tdry[name,:] = logs[9:36,4].astype(pl.float32)
    twet[name,:] = logs[9:36,5].astype(pl.float32)

# convert from Fahrenheit to Celsius:
tmax = (tmax-32)*(5./9.); tmin = (tmin-32)*(5./9.)
tdry = (tdry-32)*(5./9.); twet = (twet-32)*(5./9.)

for stat in range(27):

    #stat = 25 # station index
    
    fig, ax = pl.subplots(1,2,figsize=(9,4.5))
    
    ax1 = pl.subplot(121)
    pl.plot(tmax[:,stat],label='T$_{max}$')
    pl.plot(tmin[:,stat],label='T$_{min}$')
    pl.title('(a) T$_{max}$, T$_{min}$')
    pl.ylim(0,35); pl.xlim(0,31)
    pl.ylabel('$^\circ$C',fontsize=16)
    pl.legend(fontsize=16); pl.grid(axis='y',linestyle='--',linewidth=0.5)
    ax1.set_xticks([0,5,10,15,20,25,30])
    ax1.set_xticklabels(D[[0,5,10,15,20,25,30]])
    
    ax2 = pl.subplot(122)
    pl.plot(tdry[:,stat],label='T$_{dry}$')
    pl.plot(twet[:,stat],label='T$_{wet}$')
    pl.title('(a) T$_{dry}$, T$_{wet}$')
    pl.ylim(0,35); pl.xlim(0,31)
    pl.ylabel('$^\circ$C',fontsize=16)
    pl.legend(fontsize=16); pl.grid(axis='y',linestyle='--',linewidth=0.5)
    ax2.set_xticks([0,5,10,15,20,25,30])
    ax2.set_xticklabels(D[[0,5,10,15,20,25,30]])
    
    pl.suptitle(logs[9:36,0][stat],y=0.98)
    
    pl.subplots_adjust(top=0.92,bottom=0.08,left=0.08,right=0.96)
    
    #pl.savefig(homedir+'weatherrescue/heatwave1906/temps_hw_1906_'+logs[9:36,0][stat]+'.png')