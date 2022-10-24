# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:49:43 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import pandas as pd
from scipy import stats

pl.close('all')

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

inds = ['nao','scand','ea','eawr']
x = pl.linspace(1,68,68)
titles = ['(a) NAO','(b) SCA','(c) EA','(d) EAWR']
years = pl.linspace(1950,2017,68)

fig, ax = pl.subplots(2,2,figsize=(10,7))

for i in range(len(inds)):
    tele = pd.read_csv(indecis+inds[i]+'_index.tim',header=5,delim_whitespace=True)
    tele = pl.asarray(tele)
    
    tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
    tele_sns = pl.zeros([tele.shape[0],4])
    tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
    tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
    tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
    tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
    tele_sns[0,0] = pl.float32('nan')
    
    #TS = pl.zeros([tele_sns.size])
    #TS[0::4] = tele_sns[:,0]; TS[1::4] = tele_sns[:,1]
    #TS[2::4] = tele_sns[:,2]; TS[3::4] = tele_sns[:,3]
    
    axx = pl.subplot(2,2,i+1)
    axx.plot(years,tele_sns[:,3],color='k',lw=2)
    pl.xlim(years[0],years[-1])
    pl.ylim(-2.5,2.5)
    axx.grid(axis='y',color='grey',ls='--')
    
    m,b,r,p,s = stats.linregress(years,tele_sns[:,3])
    #print p
    
    axx.plot(years,m*years+b,ls='--',color='r')
    
    axx.annotate(titles[i]+'\n slope = '+str(round(m,3))+'\n $p$ = '+format(p,".2e"),#str(round(p,3)),
                 (1952,1.5),bbox={'facecolor':'w'},fontsize=10)
    axx.tick_params(axis='both',direction='in')
    axx.xaxis.set_ticks_position('both')
    axx.set_xticks(years[::10])
    
    if i in (1,3):
        axx.tick_params(labelleft=False)
    
    if i in (0,1):
        axx.tick_params(labelbottom=False)

    if i in (2,3):
        pl.xticks(fontsize=12)

pl.tight_layout()

#pl.savefig(indecis+'/figures/telecons_son.png',dpi=350)