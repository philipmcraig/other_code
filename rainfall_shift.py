#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:08:53 2018

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
from scipy.stats import pearsonr

pl.close('all')
resdir = '/home/users/qx911590/weatherrescue/'

rainstations = pd.read_csv(resdir+'uk_rain_1903.csv',header=None,names=None,
                                                                   delimiter=',')
#pl.genfromtxt(resdir+'uk_rain_1903.dat')
rainstations = pl.array(rainstations)

#pl.plot(rainstations[:,1:])

dwrfiles = glob.glob(resdir+'csv_fixed/1903/*')
dwrfiles = pl.asarray(dwrfiles) # list into array
dwrfiles = pl.sort(dwrfiles) # get filenames in correct order

logs = pl.zeros([len(dwrfiles),58,9],dtype='object')
names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book

for name in range(len(dwrfiles)):
    df = pd.read_csv(dwrfiles[name],names=names,header=None)
    logs[name] = pl.array(df)

#for st in range(rainstations.shape[0]):
#    station = rainstations[st,0]
#    if station == 'Loughborough' or station == 'Nottingham':
#        continue
    
stat_rw = pl.where(logs[0]=='Oxford ')
rain = logs[:,stat_rw[0][0],-1]

#for i in range(len(dwrfiles)):
#    if rain[i] == '  ' or rain[i] == ' ':
#        rain[i] = pl.float64('nan')
#    else:
#        rain[i] = float(rain[i])

a = pl.where(rain==' '); rain[a[0]] = pl.float64('nan')
b = pl.where(rain=='  '); rain[b[0]] = pl.float64('nan')
rain = rain.astype(float)
r2 = pl.zeros_like(rain); r2[:-1] = rain[1:]; rain[-1] = pl.float64('nan')

pl.figure(figsize=(10,8))
pl.plot(rainstations[23,1:-1],label='Met Office')
pl.plot(r2*25.4,label='Weather Rescue')
#pl.plot(rainstations[st,1:]-rain*25.4)
pl.xlim(0,365); pl.ylim(0,40)
pl.ylabel('mm',fontsize=16)
pl.grid(axis='y')
#pl.title('Met Office minus Weather Rescue: '+station,fontsize=18)
pl.legend()
pl.title('Oxford 1903')
pl.tight_layout()
    #pl.savefig(resdir+'raincomp_plots/mo_minus_wr_1903_'+station+'.png')

#print pearsonr(rainstations[2,1:],rain*25.4)