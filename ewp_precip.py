# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:01:25 2019

@author: qx911590
"""

import pylab as pl
import pandas as pd

pl.close('all')
homedir = '/home/users/qx911590/'

df = pd.read_csv(homedir+'ewp_precip.csv',header=None)

data = pl.array(df)

years = data[:-1,0]

precip = data[:-1,1:-1]
prf = precip[129:150].flatten()

fig, ax = pl.subplots(figsize=(12,6))
ax1 = pl.subplot(111)
ax1.plot(prf,lw=2)
pl.xlim(0,prf.size); pl.ylim(0,250)
pl.rcParams['xtick.top'] = True
ax1.tick_params(direction='in')
pl.xticks(pl.arange(0,prf.size,12))
ax1.set_xticklabels(years[129:150].astype(int),rotation=45,fontsize=12)
pl.ylabel('mm',fontsize=15)
pl.yticks(fontsize=12)
pl.grid(axis='y',ls='--')


pl.tight_layout()
pl.show()