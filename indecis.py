# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:36:22 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import glob
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

filenames = glob.glob(indecis+'*.nc')

ncfile = xr.open_dataset(filenames[1])
data = ncfile.variables['spei1'][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
months = pl.asarray([i[5:7] for i in time])

ind = pl.where((years.astype(int)==2013) & (months.astype(int)==6))
Y = pl.where((years.astype(int)==1979) & (months.astype(int)==6))

julmean = pl.nanmean(data.values[Y[0][0]::12],axis=0)
julanom = data.values[Y[0][0]::12] - julmean[None,:,:]

levels = [-4,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,4]

proj = ccrs.PlateCarree(central_longitude=5)
fig, ax = pl.subplots(1,2,figsize=(14,5))

ax1 = pl.subplot(121,projection=proj)
#ax = pl.axes(projection=ccrs.PlateCarree(central_longitude=2),
#             extent=[-35,40,30,75])
ax1.set_extent([-30,40,30,70])
ax1.coastlines(resolution='50m',linewidth=0.5,color='grey')
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)

cf1 = ax1.contourf(lon.values,lat.values,data.values[ind[0][0]].T,
            transform=ccrs.PlateCarree(),cmap='seismic_r',
            norm=pl.Normalize(-3,3),alpha=0.6,extend='min',
            levels=levels)
#cb1 = pl.colorbar(cf1,orientation='horizontal',pad=0.05,shrink=0.9)
#cb1.set_ticks(levels)
#cb1.set_ticklabels(levels)

ax1.annotate('(a) July 2013',(-29,32),fontsize=12,bbox={'facecolor':'w'})


ax2 = pl.subplot(122,projection=proj)
ax2.set_extent([-30,40,30,70])
ax2.coastlines(resolution='50m',linewidth=0.5,color='grey')
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)

cf2 = ax2.contourf(lon.values,lat.values,julanom[35].T,
            transform=ccrs.PlateCarree(),cmap='seismic_r',
            norm=pl.Normalize(-4,4),alpha=0.6,extend='min',
            levels=levels)
#pl.colorbar(cf2,orientation='horizontal',pad=0.05,shrink=0.7)

ax2.annotate('(b) anomaly from\n July climatology',(-30,32),fontsize=12,
             bbox={'facecolor':'white','alpha':1.0})

pl.subplots_adjust(top=0.99,bottom=0.14,left=0.04,right=0.96,hspace=0.15,wspace=0.1)

f = pl.gcf()
colax = f.add_axes([0.2,0.1,0.6,0.03])                   
cb = pl.colorbar(cf2,orientation='horizontal',cax=colax)
cb.set_ticks(levels)
ticklabs = pl.asarray(levels)
cb.set_ticklabels(ticklabs)
cb.ax.tick_params(labelsize=11)
cb.update_ticks()
cb.set_label('SPEI1',fontsize=12)

GridLines(ax1,False,True,True,False)
GridLines(ax2,False,True,False,True)

pl.suptitle('Standardized precipitation minus evapotranspiration index 1-month (SPEI1)',y=0.99)
#pl.savefig(indecis+'spei1_200308.png')

pl.figure(2)
ax = pl.axes(projection=ccrs.PlateCarree(central_longitude=5),
             extent=[-30,40,30,70])

ax.coastlines(resolution='50m',linewidth=0.5,color='grey')
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)

C = ax.contourf(lon.values,lat.values,julmean.T,transform=ccrs.PlateCarree(),
                alpha=0.6,cmap='seismic_r',norm=pl.Normalize(-0.03,0.03,clip=False),
                extend='both',levels=[-0.03,-0.02,-0.01,0,0.01,0.02,0.03])
ax.contour(lon.values,lat.values,julmean.T,transform=ccrs.PlateCarree(),levels=[0],
           colors='w',linewidths=1.5)
cb = pl.colorbar(C,orientation='horizontal',extend='both',pad=0.06,aspect=40)
cb.set_label('SPEI1',fontsize=12)

GridLines(ax,False,True,True,True)
pl.subplots_adjust(top=0.99,bottom=0.0,left=0.08,right=0.92)
pl.title('SPEI1 July climatology')
#pl.savefig(indecis+'spei1_augmean.png')