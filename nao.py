# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:25:54 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import xarray as xr
import cartopy
import cartopy.crs as ccrs
from scipy import stats
#import glob
from matplotlib.colors import Normalize
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pcraig_funcs as pc

class MidpointNormalize(Normalize):
    """Found this on the internet. Use to centre colour bar at zero.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        a, b = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return pl.ma.masked_array(pl.interp(value, a, b))

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

name = 'dtr'

nao = pl.genfromtxt(indecis+'nao_index.tim',skip_header=9)
#Y = pl.where(nao[:,0]==1979)


ncfile = xr.open_dataset(ncdir+name+'_month.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
months = pl.asarray([i[5:7] for i in time])

ind = pl.where((years=='1979') & (months=='02'))
ind = 0#ind[0][0]

#nao_sns = pl.zeros([nao.shape[0]/4,3])
#nao_sns[0::4] = pl.mean

nao = nao[ind:-24]


fig, ax = pl.subplots(1,1,figsize=(14,6))
ax1 = pl.subplot(111)
ax1.plot(nao[:-24,-1])
pl.xlim(0,792)
ax1.set_xticks(pl.arange(0,793,120))
ax1.set_xticklabels(years[0::120])
ax1.grid(axis='y',ls='--')
pl.tight_layout()

C = pl.zeros([lon.shape[0],lat.shape[0]])
trend = pl.zeros_like(C)

for i in range(lon.shape[0]):
    for j in range(lat.shape[0]):
        C[i,j] = stats.pearsonr(nao[:,-1],data.values[ind:,i,j])[0]
        trend[i,j] = stats.linregress(pl.linspace(1,nao.shape[0],nao.shape[0]),data.values[ind:,i,j])[0]

pl.figure(2)
proj = ccrs.PlateCarree(central_longitude=5)
ax = pl.axes(projection=ccrs.PlateCarree(central_longitude=5),
             extent=[-30,40,30,70])

ax.coastlines(resolution='50m',linewidth=0.5,color='grey')
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)

cf = ax.contourf(lon.values,lat.values,C.T,transform=ccrs.PlateCarree(),
                alpha=0.6,cmap='seismic',norm=pl.Normalize(-0.5,0.5),
                extend='both',levels=[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax.contour(lon.values,lat.values,C.T,transform=ccrs.PlateCarree(),levels=[0],
           colors='w',linewidths=1.5)
cb = pl.colorbar(cf,orientation='horizontal',extend='both',pad=0.06,aspect=40)
cb.set_label('$r$',fontsize=12)

GridLines(ax,False,True,True,True)
pl.subplots_adjust(top=0.99,bottom=0.0,left=0.08,right=0.92)
#pl.title('Standardized Precipitation Evapotranspiration NAO correlation',y=1.01)

#pl.savefig(indecis+'corrmaps/'+name+'_corr.png')

pl.figure(3)
ax = pl.axes(projection=ccrs.PlateCarree(central_longitude=5),
             extent=[-30,40,30,70])
ax.coastlines(resolution='50m',linewidth=0.5,color='grey')
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)

cf = ax.contourf(lon.values,lat.values,trend.T,transform=ccrs.PlateCarree(),
                alpha=0.6,cmap='seismic',norm=MidpointNormalize(midpoint=0))
                #extend='both')#,levels=[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax.contour(lon.values,lat.values,trend.T,transform=ccrs.PlateCarree(),levels=[0],
           colors='w',linewidths=1.5)
cb = pl.colorbar(cf,orientation='horizontal',extend='both',pad=0.06,aspect=40)
cb.set_label('$r$',fontsize=12)

GridLines(ax,False,True,True,True)
pl.subplots_adjust(top=0.99,bottom=0.0,left=0.08,right=0.92)