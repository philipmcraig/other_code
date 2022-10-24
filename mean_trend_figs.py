# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:23:16 2020

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
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

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

name = 'ogs10'
ssn = 'DJF'

ncfile = xr.open_dataset(ncdir+name+'_year.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data = data.values[:,70:326,34:200]

mean = pl.nanmean(data,axis=0)
std = pl.nanstd(data,axis=0)
var = stats.variation(data,axis=0)

trend = pl.zeros([lon2.size,lat2.size])
P = pl.zeros_like(trend); P[:] = pl.float32('nan')
for i in range(lon2.size):
    for j in range(lat2.size):
        out = stats.linregress(pl.linspace(1,68,68),data[:,i,j])
        trend[i,j] = out[0]
        sig = out[3]
        if sig < 0.05:
            P[i,j] = sig

proj = ccrs.PlateCarree()
ext = [-15,42,35,70]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')

fig, ax = pl.subplots(3,1,figsize=(5.5,9))
#pl.figure(figsize=(10,7))
#gs = gridspec.GridSpec(2, 4)
#ig = [gs[0,1:3],gs[1,:2],gs[1,2:]]

ax1 = pl.subplot(311,projection=proj,extent=ext)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(0,150,16); levs[0] = 1#[0,0.05,0.2,0.5,0.65,1,2,3,4]
cs = ax1.contourf(lon2,lat2,mean.T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap='plasma_r',extend='max')#,norm=MidpointNormalize(midpoint=0))
ax1.contour(lon2,lat2,mean.T,transform=ccrs.PlateCarree(),levels=[1],
            colors='k',linewidths=0.5)
#ax1.contour(lon2,lat2,mean.T,transform=ccrs.PlateCarree(),levels=[0.65],
#            colors='k',linewidths=0.75)
cb = pl.colorbar(cs,orientation='vertical',pad=0.03,shrink=0.85,aspect=12)
cb.set_label('day of year',fontsize=11)
cb.set_ticks(levs[::2])
cb.set_ticklabels(levs[::2].astype(int))

#ax1.annotate('(a) '+name+' mean',(-14,69),bbox={'facecolor':'w'},fontsize=12)
pl.title('a. '+name+' mean',fontsize=12)
GridLines(ax1,False,False,True,False)

ax2 = pl.subplot(312,projection=proj,extent=ext)

ax2.coastlines(linewidth=0.5,resolution='50m')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(-4,4,9)
cs = ax2.contourf(lon2,lat2,(trend.T)*10,transform=ccrs.PlateCarree(),levels=levs,
                  cmap='BrBG_r',extend='both')#,norm=MidpointNormalize(midpoint=0))
cb = pl.colorbar(cs,orientation='vertical',pad=0.03,shrink=0.85,aspect=12)
cb.set_label('days decade$^{-1}$',fontsize=12,labelpad=2)
cb.set_ticks(levs)
cb.set_ticklabels(levs.astype(int))

ax2.contourf(lon2,lat2,P.T,hatches=['...'],colors='none')

#ax2.annotate('(b) '+name+' trend',(-14,69),bbox={'facecolor':'w'},fontsize=12)
pl.title('b. '+name+' trend',fontsize=12)
GridLines(ax2,False,False,True,False)


ax3 = pl.subplot(313,projection=proj,extent=ext)

ax3.coastlines(linewidth=0.5,resolution='50m')
ax3.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(0,50,11)
cs = ax3.contourf(lon2,lat2,std.T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap='plasma_r',extend='max')
cb = pl.colorbar(cs,orientation='vertical',pad=0.03,shrink=0.85,aspect=12)
cb.set_label('days',fontsize=12,labelpad=2)
cb.set_ticks(levs[::2])
cb.set_ticklabels(levs[::2].astype(int))

#ax3.annotate('(c) '+name+' st. dev.',(-14,69),
#                             bbox={'facecolor':'w'},fontsize=12)
pl.title('c. '+name+' st. dev.',fontsize=12)
GridLines(ax3,False,True,True,False)

#pl.tight_layout()
pl.subplots_adjust(wspace=-0.30,hspace=0.0,left=0.09,right=0.97,top=0.99,bottom=0.03)

pl.savefig(indecis+'figures/'+name+'_annmean_trnd_std.png',dpi=377)


#pl.figure(2)
#ax = pl.axes(projection=proj,extent=ext)
#ax.coastlines(linewidth=0.5,resolution='50m')
##ax.add_feature(borders_50m,linewidth=0.5,zorder=5)
#
#ax.contour(lon2,lat2,data[45].T,transform=ccrs.PlateCarree(),levels=[0],
#            colors=['k'],linewidths=0.5)