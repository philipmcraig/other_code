# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:29:09 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import pandas as pd
import glob
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

pl.close('all')

home = '/home/users/qx911590/'
indecis = home + 'INDECIS/'

#stations = ['Tj Jiu','Turnu Magurele','Deva','Miercurea Ciuc',
#             'Bucuresti-Baneasa','Vidin','Kneja','Calarasi',
#             'Ramnicu Valcea','Rosiori de Vede','Sibiu','Drobeta Turnu Severin',
#             'Craiova']

#permissions = pl.loadtxt(indecis+'permissions.txt',skiprows=0)
df = pd.read_csv(indecis+'permissions.txt',header=0)

data = pl.asarray(df)

PUB = 'public'

#romania = pl.where((data[:,5]=='ROMANIA') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
#bulgaria = pl.where((data[:,5]=='BULGARIA') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
#ukraine = pl.where((data[:,5]=='UKRAINE') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
#slovakia = pl.where((data[:,5]=='SLOVAKIA') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
#hungary = pl.where((data[:,5]=='HUNGARY') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
#moldova = pl.where((data[:,5]=='MOLDOVA REPUBLIC OF') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
#serbia = pl.where((data[:,5]=='SERBIA') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
greece = pl.where((data[:,5]=='NORWAY') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]

stations = data[greece,4]
stats = list(stations)
stats = list(set(stats))
stations = pl.asarray(stats)

print stations

#rlats = data[romania,6]/10; rlons = pl.float32(data[romania,7])/10
#blats = data[bulgaria,6]/10; blons = pl.float32(data[bulgaria,7])/10
#ulats, ulons = data[ukraine,6]/10, pl.float32(data[ukraine,7])/10
#slats, slons = data[slovakia,6]/10, pl.float32(data[slovakia,7])/10
#hlats, hlons = data[hungary,6]/10, pl.float32(data[hungary,7])/10
#mlats, mlons = data[moldova,6]/10, pl.float32(data[moldova,7])/10
#ylats, ylons = data[serbia,6]/10, pl.float32(data[serbia,7])/10

proj = ccrs.PlateCarree()
ext = [19,31,43,49]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')

"""ax = pl.subplot(projection=proj,extent=ext)
ax.coastlines(linewidth=0.5,resolution='50m')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)

#ax.plot(rlons,rlats,marker='x',lw=0)
ax.scatter(rlons,rlats)
#ax.scatter(blons,blats)
ax.scatter(ulons,ulats)
ax.scatter(slons,slats)
ax.scatter(hlons,hlats)
ax.scatter(mlons,mlats)
ax.scatter(ylons,ylats)

ta_o = pl.zeros([len(stations),68])

for s in range(len(stations)):

    statname = stations[s]#'Craiova'
    
    for fname in glob.glob(indecis+'TG/TG_SOUID*'):
        with open(fname) as f:
            contents = f.read()
        if statname in contents:
            break
        elif statname.upper() in contents:
            break
    
    #print fname
    
    statdata = pd.read_csv(fname,skiprows=19)
    statdata = pd.DataFrame(statdata)#,columns=['STAID','SOUID','DATE','tg','Q_tg'])
    
    temp = statdata[statdata.columns[3]]
    missing = pl.where(temp==-9999)[0]
    temp[missing] = pl.float32('nan')
    temp = temp/10.
    
    date = statdata[statdata.columns[2]]
    d = pl.array(date)
    years = [str(i)[:4] for i in d]; years = pl.array(years).astype(int)
    months = [str(i)[4:6] for i in d]; months = pl.array(months).astype(int)
    days = [str(i)[6:] for i in d]; days = pl.array(days).astype(int)
    
    
    if years.min()<=1950 and years.max()>=2017:
        y1 = pl.where(years==1950)[0][0]
        y2 = pl.where(years==2017)[0][-1]
        Y = pl.linspace(1950,2017,68)
    elif years.min()>1950 and years.max()>=2017:
        y1 = pl.where(years==years.min())[0][0]
        y2 = pl.where(years==2017)[0][-1]
        Y = pl.linspace(years.min(),2017,2017-years.min()+1)
    elif years.min()<=1950 and years.max()<2017:
        y1 = pl.where(years==1950)[0][0]
        y2 = pl.where(years==years.max())[0][-1]
        Y = pl.linspace(1950,years.max(),years.max()-1950+1)
    elif years.min()>1950 and years.max()<2017:
        y1 = pl.where(years==years.min())[0][0]
        y2 = pl.where(years==years.max())[0][-1]
        Y = pl.linspace(years.min(),years.max(),years.max()-years.min()+1)
    
    fullyears = pl.linspace(1950,2017,68)
    Z = len(fullyears)-len(Y)
    monthly_means = pl.zeros([68,12]); monthly_means[:,:] = pl.float32('nan')
    
    for yr in range(len(Y)):
        #Z = pl.where(fullyears==Y[yr])[0][0]
        for mn in range(1,13):
            #Y = years[y1:y2][yr]
            #M = months[y1:y2][mn]
            X = pl.where((years[y1:y2+1]==Y[yr]) & (months[y1:y2+1]==mn))[0]
            monthly_means[yr+Z,mn-1] = pl.mean(temp[y1:y2+1][X[0]:X[-1]+1])
    
    ta_o[s,:] = pl.nanmean(monthly_means[:,3:10],axis=1)

tmean = pl.nanmean(ta_o,axis=0)

pl.figure(2)
pl.plot(pl.linspace(0,67,68),tmean,color='k',lw=2.5)
#for i in range(len(stations)):
#    if i == 0:
#        c = 'r'
#    elif i == 1:
#        c = 'b'
#    elif i == 17:
#        c = 'darkgoldenrod'
#    else:
#        c = 'grey'
pl.plot(pl.linspace(0,67,68),ta_o.T,color='grey',lw=0.7)
pl.xlim(0,fullyears.size)

anoms = tmean - pl.nanmean(tmean)

pl.figure(3)
pl.plot(pl.linspace(0,67,68),anoms)
pl.grid(axis='y',ls='--')

pl.show()"""
