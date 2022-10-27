# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:54:10 2020

@author: qx911590
"""

import pylab as pl
import xarray as xr
import cartopy
import cartopy.crs as ccrs
import xarray as xr
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pcraig_funcs import NearestIndex

def GridLines(ax,top,left,right):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([60,90,120,150,180])
    gl.ylocator = mticker.FixedLocator([-45,-30,-15,0,15,30,45])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':11}
    gl.ylabel_style = {'color': 'k','size':11}
    
    return None

pl.close('all')
#sheddir = '/home/users/qx911590/np838619/Watershed/'

ncfile = xr.open_dataset('/home/users/pmcraig/etopo05.nc')
topo = xr.DataArray(ncfile.variables['ROSE'][:])
lat = xr.DataArray(ncfile.variables['ETOPO05_Y'][:])
lon = xr.DataArray(ncfile.variables['ETOPO05_X'][:])
ncfile.close()

lat = lat.values
lon = lon.values
topo = topo.values

LS15 = pl.genfromtxt('/home/users/pmcraig/LS15full_clicks.txt',skip_header=5)
Rod11 = pl.genfromtxt('/home/users/pmcraig/Rod11full_clicks.txt',skip_header=5)

proj = ccrs.PlateCarree(central_longitude=5)
ax = pl.axes(projection=ccrs.PlateCarree(),
             extent=[88,155,-30,35])#88,-30,155,20
ax.set_extent([88,155,-25,30])
ax.coastlines(resolution='50m',linewidth=0.5,color='grey')

ax.pcolormesh(lon[1055:1862],lat[720:1500],topo[720:1500,1055:1862],
              transform=ccrs.PlateCarree(),cmap='binary',
              norm=pl.Normalize(0,8000))

ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['water'])
ax.add_feature(ocean_50m)

ax.plot(LS15[:,0],LS15[:,1],color='k',lw=1.2,marker='x',transform=ccrs.PlateCarree(),
        label='Levang & Schmitt (2015)')
ax.plot(Rod11[:,0],Rod11[:,1],color='r',lw=0.6,marker='.',transform=ccrs.PlateCarree(),
        label='Rodriguez et al. (2011)')
    
ax.legend(fontsize=12)
GridLines(ax,False,True,True)
pl.tight_layout()
#pl.show()
pl.savefig('/home/users/pmcraig/alt_eaa_bnds.png')
