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

dwrfiles = glob.glob(resdir+'csv_fixed/1905/*')
dwrfiles = pl.asarray(dwrfiles) # list into array
dwrfiles = pl.sort(dwrfiles) # get filenames in correct order

logs = pl.zeros([len(dwrfiles),58,9],dtype='object')
names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book

for name in range(len(dwrfiles)):
    df = pd.read_csv(dwrfiles[name],names=names,header=None)
    logs[name] = pl.array(df)

pres = pl.zeros([27,len(dwrfiles)*2])


for st in range(9,36):
    yeve = logs[:,st,1] # yesterday evening pressure
    tmor = logs[:,st,3] # this morning pressure
    #pr_temp = logs[:,st,3]#.astype(float)
    a1 = pl.where(yeve==' '); yeve[a1[0]] = pl.float64('nan')
    b1 = pl.where(yeve=='  '); yeve[b1[0]] = pl.float64('nan')
    a2 = pl.where(tmor==' '); tmor[a2[0]] = pl.float64('nan')
    b2 = pl.where(tmor=='  '); tmor[b2[0]] = pl.float64('nan')
    yeve = yeve.astype(float); tmor = tmor.astype(float)
    pres[st-9,::2] = yeve*33.864
    pres[st-9,1::2] = tmor*33.864
#    for i in range(len(dwrfiles)):
#        if pres[i] == '  ' or pres[i] == ' ':
#            pres[i] = pl.float64('nan')
#        else:
#            pres[i] = float(pres[i])

#pl.figure(figsize=(10,8))
fig,ax = pl.subplots(1,1,figsize=(10,8))
ax.plot(pres[:,117:179].T)
pl.xlim(0,60); pl.ylim(940,1040); pl.ylabel('hPa',fontsize=16)
pl.xticks([0,10,20,30,40,50,61],
               ['28th Feb 8PM','5th Mar 8PM','10th Mar 8PM','16th Mar 8PM',
                    '21th Mar 8PM','26th Mar 8PM','31st Mar 8PM'])
pl.legend(logs[0,9:36,0],loc=4,ncol=2)
pl.grid(axis='y')
pl.title('March 1905 8AM & 8PM pressure')

pl.tight_layout()
#pl.savefig(resdir+'pres_8am_8pm_1905_03.png')