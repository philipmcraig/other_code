# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:25:54 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import pandas as pd
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

teleindex = 'eawr'
name = 'eto'

tele = pd.read_csv(indecis+teleindex+'_index.tim',header=5,delim_whitespace=True)
tele = pl.asarray(tele)
#tele = tele[:-2]

#tele = pl.genfromtxt(indecis+'tele_index.tim',skip_header=9)
#Y = pl.where(tele[:,0]==1979)


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
ind = ind[0][0]

tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
tele_sns = pl.zeros([tele.shape[0],4])
tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
tele_sns[0,0] = pl.float32('nan')

tele_sns = pl.reshape(tele_sns,newshape=(tele_sns.shape[0]*tele_sns.shape[1]))


#fig, ax = pl.subplots(1,1,figsize=(14,6))
#ax1 = pl.subplot(111)
#ax1.plot(tele[:-24,-1])
#pl.xlim(0,792)
#ax1.set_xticks(pl.arange(0,793,120))
#ax1.set_xticklabels(years[0::120])
#ax1.grid(axis='y',ls='--')
#pl.tight_layout()

C = pl.zeros([lon.shape[0],lat.shape[0]])
trend = pl.zeros_like(C)

for i in range(lon.shape[0]):
    for j in range(lat.shape[0]):
        C[i,j] = stats.pearsonr(tele_sns[ind+3::4],data.values[ind+3::4,i,j])[0]
        #trend[i,j] = stats.linregress(pl.linspace(1,tele_sns[ind+2:-8:4].shape[0],
        #                                        tele_sns[ind+2:-8:4].shape[0]),
        #                                        data.values[ind+2::4,i,j])[0]

Cmax = pl.nanmax(C); Cmin = pl.nanmin(C)

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
                alpha=0.6,cmap='seismic',norm=pl.Normalize(-0.7,0.7),
                extend='both',levels=pl.linspace(-0.7,0.7,15))#[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax.contour(lon.values,lat.values,C.T,transform=ccrs.PlateCarree(),levels=[0],
           colors='w',linewidths=1.5)
cb = pl.colorbar(cf,orientation='horizontal',extend='both',pad=0.06,aspect=40)
cb.set_label('$r$',fontsize=12)

GridLines(ax,False,True,True,True)
pl.subplots_adjust(top=0.99,bottom=0.0,left=0.08,right=0.92)
pl.title('Reference Evapotranspiration DJF EAWR correlation',y=1.01)
ax.annotate(' max = '+str(round(Cmax,2))+'\n min = '+str(round(Cmin,2)),
                        (-34,56))

pl.savefig(indecis+'corrmaps/'+teleindex+'/'+name+'_'+teleindex+'_djf_corr.png')

#pl.figure(3)
#ax = pl.axes(projection=ccrs.PlateCarree(central_longitude=5),
#             extent=[-30,40,30,70])
#ax.coastlines(resolution='50m',linewidth=0.5,color='grey')
#borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
#                                           '50m',edgecolor='grey',
#                                        facecolor='none')
#ax.add_feature(borders_50m,linewidth=0.5,zorder=5)
#
#cf = ax.contourf(lon.values,lat.values,trend.T,transform=ccrs.PlateCarree(),
#                alpha=0.6,cmap='seismic',norm=MidpointNormalize(midpoint=0))
#                #extend='both')#,levels=[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
#ax.contour(lon.values,lat.values,trend.T,transform=ccrs.PlateCarree(),levels=[0],
#           colors='w',linewidths=1.5)
#cb = pl.colorbar(cf,orientation='horizontal',extend='both',pad=0.06,aspect=40)
#cb.set_label('mm yr$^{-1}$',fontsize=12)
#
#GridLines(ax,False,True,True,True)
#pl.subplots_adjust(top=0.99,bottom=0.0,left=0.08,right=0.92)
#pl.title('$P-E$ JJA trend',y=1.01)

#pl.savefig(indecis+'trndmaps/'+name+'_jja_trnd.png')