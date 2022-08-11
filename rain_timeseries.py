# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:08:00 2018

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
from hf_func import OSGB36toWGS84
from rt_func import WGS84toOSGB36
from functions import NearestIndex
#from mpl_toolkits.basemap import Basemap
import sklearn.metrics as skm
import cartopy
import cartopy.crs as ccrs

def RMSE(obs,mod):
    """Calculate root mean squared error and take nan values into account.
    
    Args:
        obs (array): observed data
        mod (array): predicted/modelled data
    
    Returns:
        rmse (float): root mean square error of data
    """
    z = pl.where(pl.isnan(obs)==True)
    obs = pl.delete(obs,z)
    mod = pl.delete(mod,z)
    
    #rmse = pl.sqrt(skm.mean_squared_error(obs,mod))
    rmse = pl.sqrt(((mod-obs)**2).mean())
    
    return rmse

def MBE(obs,mod):
    """Calculate mean bias error and take nan values into account.
    
    Args:
        obs (array): observed data
        mod (array): predicted/modelled data
    
    Returns:
        mbe (float): mean bias error of data
    """
    z = pl.where(pl.isnan(obs)==True)
    obs = pl.delete(obs,z)
    mod = pl.delete(mod,z)
    
    mbe = (obs-mod).mean()
    
    return mbe

pl.close('all')
resdir = '/home/users/qx911590/weatherrescue/'

filenames = glob.glob(resdir+'UK_daily_rainfall/*')
filenames = pl.asarray(filenames)
filenames = pl.sort(filenames)

dwrfiles = glob.glob(resdir+'csv_fixed/1903/*')
dwrfiles = pl.asarray(dwrfiles)
dwrfiles = pl.sort(dwrfiles)

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book
dwrinds = [9,10,16,17,18,19,20,22,23,24,25,26,27,28,29,30,32,33,56,57]
dwrrain = pl.zeros([dwrfiles.size,len(dwrinds)],dtype=object)

for name in range(dwrfiles.size):
    df = pd.read_csv(dwrfiles[name],header=None,names=names)
    logs = pl.array(df) # dataframe into array
    #space = pl.where(logs[:,-1])
    dwrrain[name] = logs[:,-1][dwrinds]

space = pl.where(dwrrain==' ')
space2 = pl.where(dwrrain=='  ')
dwrrain[space] = pl.float32('nan')
dwrrain[space2] = pl.float32('nan')
dwrrain = dwrrain.astype(float)
#dwrrain[:16] = pl.float32('nan')

dwrstats = logs[dwrinds,0]

statlocs = pd.read_csv(resdir+'latlon1903_pc.txt',delim_whitespace=True)
statlocs = pl.asarray(statlocs)

sl_inds = [8,9,15,16,17,18,19,21,22,23,24,25,26,27,28,29,31,32,55,56]

rain = pl.zeros([len(filenames),290,180])

for i in range(len(filenames)):
    rain[i] = pl.genfromtxt(filenames[i],skip_header=6)
z = pl.where(rain==-9999)
rain[z] = pl.float32('nan')

x_coords = pl.arange(-200000+2500,700001-2500,5000)
y_coords = pl.arange(-200000+2500,1250001-2500,5000)

#lonlat = pl.zeros([x_coords.size,y_coords.size],dtype=object)
lons = pl.zeros([x_coords.size,y_coords.size])
lats = pl.zeros([x_coords.size,y_coords.size])

for i in range(x_coords.size):
    for j in range(y_coords.size):
        a,b = OSGB36toWGS84(x_coords[i],y_coords[j])
        #lonlat[i,j] = (a,b)
        lons[i,j] = a; lats[i,j] = b

#pl.figure(figsize=(9,9.5))
#ax = pl.axes(projection=ccrs.TransverseMercator(central_longitude=-2),
#             extent=[-9,5,49,61])
#ax.coastlines(resolution='50m',linewidth=0.5,color='grey')

for stat in range(len(sl_inds)):
    loc = (statlocs[sl_inds[stat],2],statlocs[sl_inds[stat],1]) # lon,lat
    #print loc
#lonf = lons.flatten(); latf = lats.flatten()
#
#ind1 = NearestIndex(lonf,PB_loc[0]); ind2 = NearestIndex(latf,PB_loc[1])

    a = WGS84toOSGB36(loc[1],loc[0])
    ind1 = NearestIndex(x_coords,a[0]); ind2 = NearestIndex(y_coords,a[1])
    if pl.isnan(rain[0,-ind2,ind1]) == True:
        R = rain[0,-ind2-4:-ind2+5,ind1-4:ind1+5]
        N = pl.where(pl.isnan(R)==False)
        #I = NearestIndex(N[0],4)
        #J = NearestIndex(N[1],4)
        d = pl.sqrt((4-N[0][:])**2+(4-N[1][:])**2)
        dm = pl.where(d==d.min())
        i = N[0][dm[0][0]]; j = N[1][dm[0][0]]
        latind = rain.shape[1] - ind2 - (4 - i)
        lonind = ind1 - (4 - j)
    else:
        latind = rain.shape[1]-ind2; lonind = ind1
    
    rmse = RMSE(dwrrain[1:,stat]*25.4,rain[:-1,latind,lonind])
    mbe = MBE(dwrrain[1:,stat]*25.4,rain[:-1,latind,lonind])
    print logs[dwrinds[stat],0], round(rmse,2), round(mbe,2)
    
#    if stat in (4,16):
#        ha = 'right'
#    else:
#        ha = 'left'
#    ax.plot(loc[0],loc[1],transform=ccrs.PlateCarree(),color='r',marker='o')
#    pl.text(loc[0]+0.2,loc[1]-0.15,dwrstats[stat][:-1],transform=ccrs.Geodetic(),
#            horizontalalignment=ha)
    #pl.figure(stat+1)
    fig,ax = pl.subplots(1,2,figsize=(10,5))
    
    ax1 = pl.subplot(121)
    ax1.plot(rain[:,latind,lonind],label='Met Office')
    ax1.plot(dwrrain[:,stat]*25.4,label='Weather Rescue')
    pl.grid(axis='y'); pl.xlim(0,365); pl.ylim(0,70); pl.legend()
    pl.ylabel('mm',fontsize=15); pl.xlabel('days',fontsize=15)
    pl.title('(a) '+dwrstats[stat]+'rainfall')
    
    ax2 = pl.subplot(122)
    ax2.plot(dwrrain[1:,stat]*25.4-rain[:-1,latind,lonind])
    pl.grid(axis='y')
    pl.xlim(0,364); pl.ylim(-35,25)
    pl.ylabel('mm',fontsize=15); pl.xlabel('days',fontsize=15)
    pl.title('(b) Weather Rescue minus Met Office')
    
    #pl.title('Met Office minus Weather Rescue: Nottingham')
    pl.tight_layout()
    #pl.savefig(resdir+'raincomp_plots/rain_panels_1903_'+dwrstats[stat][:-1]+'.png')

#pl.tight_layout()