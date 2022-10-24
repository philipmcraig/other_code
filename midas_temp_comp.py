# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:20:44 2022

@author: qx911590
"""

from __future__ import division
import pylab as pl
import glob
import pandas as pd

def MDS_func(filepath):
    """
    """
    df = pd.read_csv(filepath,skiprows=90)
    df = df.head(df.shape[0] -1)
    mds_arr = pl.asarray(df)
    mds_dates = pl.asarray([x[:10] for x in mds_arr[:,0]])
    Tx_mds = mds_arr[:,8]
    Tn_mds = mds_arr[:,9]
    
    return Tx_mds, Tn_mds, mds_dates
    

def SEF_maxmin(seffiles,ind,mds_dates):
    """
    """
    df = pd.read_csv(seffiles[ind],skiprows=13,delimiter='	')
    sef_arr = pl.asarray(df)
    
    dwr_dates = pl.asarray([str(x[0])+'-'+"{:02d}".format(x[1])+'-'\
                                    +"{:02d}".format(x[2]) for x in sef_arr[:]])

    temp = sef_arr[:,-2]
    temp_ext = pl.zeros([mds_dates.size]); temp_ext[:] = float('nan')
    
    for i in range(mds_dates.size):
        if mds_dates[i] in dwr_dates:
            F = pl.where(dwr_dates==mds_dates[i])[0][0]
            temp_ext[i] = temp[F]
    
    return temp_ext

def IRE_func(iredir):
    """
    """
    df = pd.read_csv(iredir+'NUIGalway_1851-1965.csv',skiprows=0)
    ire_arr = pl.asarray(df)
    yr_inds = pl.where(ire_arr[:,0]==1863)[0]
    ire_dates = pl.asarray([str(int(x[0]))+'-'+"{:02d}".format(int(x[1]))+'-'+\
            "{:02d}".format(int(x[2])) for x in ire_arr[yr_inds[0]:yr_inds[-1]+1]])
    Tx_ire = ire_arr[yr_inds[0]:yr_inds[-1]+1,-2]
    Tn_ire = ire_arr[yr_inds[0]:yr_inds[-1]+1,-1]
    
    return Tx_ire, Tn_ire, ire_dates

def SEF_tdry(seffiles,ire_dates):
    """
    """
    df = pd.read_csv(seffiles[0],skiprows=13,delimiter='	')
    sef_arr = pl.asarray(df)
    dwr_dates = pl.asarray([str(x[0])+'-'+"{:02d}".format(x[1])+'-'\
                                 +"{:02d}".format(x[2]) for x in sef_arr[:]])
    temp = sef_arr[:,-2]
    temp_ext = pl.zeros([ire_dates.size]); temp_ext[:] = float('nan')
    for i in range(mds_dates.size):
        if ire_dates[i] in dwr_dates:
            F = pl.where(dwr_dates==ire_dates[i])[0][0]
            temp_ext[i] = temp[F]
    
    return temp_ext

pl.close('all')

homedir = '/home/users/qx911590/'
midasdir = homedir + 'weatherrescue/midasfiles/'
sefdir = homedir + 'weatherrescue/seffiles/'
iredir = homedir + 'weatherrescue/ILMMT/'

year = '1875'
counties = ['merseyside','oxfordshire']
mds_locs = ['bidston','oxford']
sef_locs = ['LIVERPOOL','OXFORD','GALWAY']

fig, ax = pl.subplots(3,1,figsize=(16,8))

for i in range(len(mds_locs)):

    midasfiles = glob.glob(midasdir+\
                            'midas-open_uk-daily-temperature-obs_dv-202107_*'+\
                                                mds_locs[i]+'*'+year+'.csv')
    
    seffiles = glob.glob(sefdir+'*'+sef_locs[i]+'*')

    Tx_mds, Tn_mds, mds_dates = MDS_func(midasfiles[0])
    Tx_dwr = SEF_maxmin(seffiles,1,mds_dates)
    Tn_dwr = SEF_maxmin(seffiles,0,mds_dates)

    axx = pl.subplot(3,1,i+1)
    axx.plot(Tx_dwr-Tx_mds,marker='o',lw=0.5,label='Tmax')
    axx.plot(Tn_dwr-Tn_mds,marker='o',lw=0.5,label='Tmin')
    axx.grid(axis='y',ls=':',color='grey')
    pl.xlim(0,364)
    pl.xticks([0,31,59,90,120,151,181,212,243,273,304,334,364])
    axx.set_xticklabels(mds_dates[[0,31,59,90,120,151,181,212,243,273,304,334,364]])
    pl.ylabel('$^\circ$C',fontsize=12,labelpad=-10)
    pl.title(sef_locs[i]+' '+'DWR minus MIDAS Tmax, Tmin '+year,size=10)
    
    if i == 1:
        axx.legend(loc=3,ncol=1,columnspacing=0.5,handletextpad=0.5)

fig.text(0.005,0.95,'(a)',size=12)
fig.text(0.005,0.62,'(b)',size=12)

Tx_ire, Tn_ire, ire_dates = IRE_func(iredir)

seffiles = glob.glob(sefdir+'*'+sef_locs[2]+'*')
tdry_dwr = SEF_tdry(seffiles,ire_dates)


ax3 = pl.subplot(313)
ax3.plot(Tx_ire,lw=1.5,label='Tmax')
ax3.plot(Tn_ire,lw=1.5,label='Tmin')
ax3.plot(tdry_dwr,lw=1.5,label='Tdry')
ax3.grid(axis='y',ls=':',color='grey')
pl.xlim(0,364)
pl.xticks([0,31,59,90,120,151,181,212,243,273,304,334,364])
ax3.set_xticklabels(ire_dates[[0,31,59,90,120,151,181,212,243,273,304,334,364]])
pl.ylabel('$^\circ$C',fontsize=12)
pl.title(sef_locs[2]+' DWR Tdry & Mateus et al. (2020) Tmax, Tmin 1863',size=10)

ax3.legend(loc=2,ncol=1,columnspacing=0.5,handletextpad=0.5)

fig.text(0.005,0.29,'(c)',size=12)

pl.tight_layout()
pl.subplots_adjust(top=0.97,bottom=0.04,wspace=0.16,hspace=0.23,right=0.972)

#pl.savefig(homedir+'weatherrescue/midas_dwr_temps_comp_3stations.png',dpi=400)
#pl.savefig(homedir+'weatherrescue/midas_dwr_temps_comp_3stations.pdf',dpi=400)