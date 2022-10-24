# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:13:09 2020

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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import cmapFunctions_forPhil as CFP
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
    gl.xlabel_style = {'color': 'k','size':10}
    gl.ylabel_style = {'color': 'k','size':10}
    
    return None

pl.close('all')

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

name1 = 'ta_o'
name2 = 'gtg'

ncfile = xr.open_dataset(ncdir+name1+'_year.nc')
data_tao = ncfile.variables[name1][:]
data_tao = xr.DataArray(data_tao)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time1 = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

ncfile = xr.open_dataset(ncdir+name2+'_month.nc')
data_gtg = ncfile.variables[name2][:]
data_gtg = xr.DataArray(data_gtg)
time2 = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
x = pl.linspace(1,68,68)

data_tao = data_tao[:,70:326,34:200]
data_gtg = data_gtg[:,70:326,34:200]
data_gtg = pl.reshape(data_gtg.data,
                      newshape=(data_tao.shape[0],
                                data_gtg.shape[0]/data_tao.shape[0],
                                                        lon2.size,lat2.size))

gtg_std = pl.nanstd(data_gtg[:,3:10,:,:],axis=0)

percent = pl.zeros([7,lon2.size,lat2.size]); percent[:,:,:] = pl.float32('nan')

#for m in range(3,10):
#    percent[m-3] = (pl.nanmean(data_rti[:,m,:,:],axis=0)/pl.nanmean(data_gsr[:,:,:],axis=0))*100

cors = pl.zeros([7,lon2.size,lat2.size]); cors[:,:,:] = pl.float32('nan')
pval = cors.copy()
trends = cors.copy(); trnd_p = trends.copy()

for m in range(3,10):
    for i in range(lon2.size):
        for j in range(lat2.size):
            #r = stats.pearsonr(data_tao[:,i,j],data_gtg[:,m,i,j])
            #cors[m-3,i,j] = r[0]
            out = stats.linregress(x,data_gtg[:,m,i,j])
            trends[m-3,i,j] = out[0]
            if out[3] < 0.05:
                #pval[m-3,i,j] = r[1]
                trnd_p[m-3,i,j] = out[3]


pl.figure(figsize=(12,5.5))
gs = gridspec.GridSpec(2, 8)
ig = [gs[0,:2],gs[0,2:4],gs[0,4:6],gs[0,6:],
      gs[1,1:3],gs[1,3:5],gs[1,5:7]]

proj = ccrs.PlateCarree()
ext = [-15,42,35,70]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
cmap, clevs = CFP.get_eofColormap(cors)
clevs = pl.around(clevs,2); clevs[2] = 0
levs = pl.linspace(-1.6,1.6,17)
months = ['April','May','June','July','August','September','October']

for m in range(len(ig)):
    axx = pl.subplot(ig[m],projection=proj,extent=ext)
    axx.coastlines(linewidth=0.5,resolution='50m')
    axx.add_feature(borders_50m,linewidth=0.5,zorder=5)
    
    cs = axx.contourf(lon2,lat2,trends[m].T,transform=ccrs.PlateCarree(),#levels=clevs,
                  cmap='RdYlBu_r',extend='max',alpha=0.6,
                  norm=MidpointNormalize(midpoint=0))
    axx.contourf(lon2,lat2,trnd_p[m].T,hatches=['..'],colors='none',ec='k')

    pl.title(months[m])
    
    if m == 0:
        GridLines(axx,False,True,True,False)
    elif m == 1:
        GridLines(axx,False,True,False,False)
    elif m == 2:
        GridLines(axx,False,True,False,False)
    elif m == 3:
        GridLines(axx,False,True,False,True)
    elif m == 4:
        GridLines(axx,False,True,True,False)
    elif m == 5:
        GridLines(axx,False,True,False,False)
    elif m == 6:
        GridLines(axx,False,True,False,True)

pl.matplotlib.rcParams['hatch.linewidth'] = 0.75

f = pl.gcf()
colax = f.add_axes([0.15,0.1,0.7,0.03])
cb = pl.colorbar(cs,orientation='horizontal',cax=colax)#pad=0.05,fraction=0.10,
cb.set_label('$^\circ$C yr$^{-1}$',fontsize=14)
#cb.set_ticks(clevs)
#cb.set_ticklabels(clevs)

pl.tight_layout()
pl.subplots_adjust(hspace=-0.25,right=0.96,left=0.04,top=1.06,bottom=0.06)

#pl.savefig(indecis+'figures/'+name1+'_'+name2+'_monthly_trnds.png',dpi=407)