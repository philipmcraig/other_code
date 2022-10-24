# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:21:35 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import xarray as xr
import pandas as pd
from scipy import stats
from scipy import signal
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def GridLines(ax,top,bottom,left,right):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([-40,-30,-20,-10,0,10,20,30,40,50])
    gl.ylocator = mticker.FixedLocator([20,30,40,50,60,70,80])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':11}
    gl.ylabel_style = {'color': 'k','size':11}
    
    return None


pl.close('all')

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

teleindex = 'nao'
name = 'rti'

tele = pd.read_csv(indecis+teleindex+'_index.tim',header=5,delim_whitespace=True)
tele = pl.asarray(tele)

ncfile = xr.open_dataset(ncdir+name+'_season.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
months = pl.asarray([i[5:7] for i in time])
#
ind = pl.where((years=='1979') & (months=='04'))
ind = 1#ind[0][0]

tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
tele_sns = pl.zeros([tele.shape[0],4])
tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
tele_sns[0,0] = pl.float32('nan')


#tele_sns = pl.reshape(tele_sns,newshape=(tele_sns.shape[0]*tele_sns.shape[1]))
lon2 = lon.values[81:326]
lat2 = lat.values[34:200]

#data_notrend = pl.detrend(data.values,axis=0)
data_djf = data.values[4::4,81:326,34:200]
tele_dt = pl.detrend_linear(tele_sns[1:,0]) # detrended index

x = pl.linspace(1,67,67)
tele_trnd = stats.linregress(x,tele_sns[1:,0]).slope

#detrend_y = pl.zeros_like(data_jja); detrend_y[:] = pl.float32('nan')
signal = pl.zeros([lon2.size,lat2.size]); signal[:] = pl.float32('nan')
var_trnd = pl.zeros([lon2.size,lat2.size]); var_trnd[:] = pl.float32('nan')
#b = m.copy(); r = m.copy(); p = m.copy(); se = m.copy()
#not_nan_ind = ~pl.isnan(data_jja)

for i in range(lon2.size):
    for j in range(lat2.size):
        y = data_djf[:,i,j]
        not_nan_ind = ~pl.isnan(y)
        if pl.where(not_nan_ind==True)[0].size == 0:
            pass
        else:
            m, b, r, p, se = stats.linregress(x[not_nan_ind],y[not_nan_ind])
            var_trnd[i,j] = m
            detrend_y = y - m*x# + b) # detrended variable
            signal[i,j] = stats.linregress(tele_dt,detrend_y).slope # NAO signal

nosig = var_trnd - tele_trnd*signal[:,:]
#pl.imshow(pl.flipud(out.T),norm=pl.Normalize(-0.01,0.01),cmap='seismic')

ax = pl.axes(projection=ccrs.PlateCarree(),extent=[-20,42,35,70])
#ax.set_extent([])
ax.coastlines(linewidth=0.5,resolution='50m')
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = [-200,-150,-100,-75,-50,-25,-10,-1,
                    1,10,25,50,75,100,150,200]
cs = ax.contourf(lon2,lat2,signal.T,norm=pl.Normalize(-200,200),cmap='seismic',
            transform=ccrs.PlateCarree(),alpha=0.6,extend='both',
            levels=levs)
cb = pl.colorbar(cs,orientation='horizontal',shrink=1.0,pad=0.05)
cb.set_label('mm NAOI$^{-1}$',fontsize=14)
cb.set_ticks(levs)
cb.set_ticklabels(levs)

GridLines(ax,True,False,True,True)
pl.tight_layout()
pl.subplots_adjust(left=0.09,right=0.91)
#pl.savefig(indecis+'figures/'+name+'_djf_remove'+teleindex+'.png')
#pl.savefig(indecis+'figures/'+name+'_djf_signal_'+teleindex+'.png')