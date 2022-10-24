# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:06:49 2020

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
import pcraig_funcs as pc

def AreasCalc():
    """
    """
    glat = pl.arange(-89.875,89.876,0.25)
    glon = pl.arange(-179.875,179.876,0.25)
    
    # Convert lat & lon arrays to radians
    lat_rad = pl.radians(glat[:])
    lon_rad = pl.radians(pl.flipud(glon[:]))
    
    lat_half = pc.HalfGrid(lat_rad)
    nlon = lon_rad.size # number of longitude points
    delta_lambda = (2*pl.pi)/nlon


    #--------------calculate cell areas here, use function from above--------------
    # set up empty array for area, size lat_half X lon_half
    areas = pl.zeros([lon_rad.size,lat_rad.size])
    radius = 6.37*(10**6)
    # loop over latitude and longitude
    for i in range(glon.size): # loops over 256
        for j in range(lat_half.size-1): # loops over 512
            latpair = (lat_half[j+1],lat_half[j])
            areas[i,j] = pc.AreaFullGaussianGrid(radius,delta_lambda,latpair)
    
    areas_clip = areas[70:326,34:200]
    
    return areas_clip

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

name = 'ogs10'

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

ncfile = xr.open_dataset(ncdir+name+'_year.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
x = pl.linspace(1,years.size,years.size)

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data = data.values[:,70:326,34:200]

mn = pl.nanmean(data,axis=0)
sd = pl.nanstd(data,axis=0)

Mx = pl.nanmax(data,axis=0)
Mn = pl.nanmin(data,axis=0)

trend = pl.zeros_like(sd); trend[:] = pl.float32('nan')
trnd_p = trend.copy()

for i in range(lon2.size):
    for j in range(lat2.size):
        trend[i,j] = stats.linregress(x,data[:,i,j]).slope
        sig = stats.linregress(x,data[:,i,j]).pvalue
        if sig < 0.05:
            trnd_p[i,j] = sig

proj = ccrs.PlateCarree()
ext = [-23,41,35,70]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')

pl.figure(figsize=(8,6))
ax = pl.axes(projection=proj,extent=ext)
ax.coastlines(linewidth=0.5,resolution='50m')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)

#levs = [1,32,60,91,121,152,182]
cs = ax.contourf(lon2,lat2,trend.T*10,transform=ccrs.PlateCarree(),cmap='BrBG_r',
                 levels=pl.linspace(-10,10,21),alpha=1)#,extend='max')
cb = pl.colorbar(cs,orientation='horizontal',pad=0.06)
cb.set_label('days decade$^{-1}$',fontsize=12)
cb.set_ticks(pl.linspace(-10,10,21))
cb.set_ticklabels(pl.linspace(-10,10,21).astype(int))

ax.contourf(lon2,lat2,trnd_p.T,hatches=['..'],colors='none')

GridLines(ax,False,True,True,True)
pl.title(name+' trend')
pl.tight_layout()
pl.subplots_adjust(left=0.06,right=0.94)
#pl.savefig(indecis+'figures/'+name+'_trnd.png',dpi=350)

fig, ax = pl.subplots(1,2,figsize=(10.5,4.5))

ax1 = pl.subplot(121,projection=proj,extent=ext)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)

levs = [0,30,58,89,119,150,180,201,232,252,283]
cs = ax1.contourf(lon2,lat2,Mn.T,transform=ccrs.PlateCarree(),cmap='inferno_r',
                 levels=levs,alpha=1,extend='max')
pl.title('(a) earliest '+name)
GridLines(ax1,False,True,True,False)

ax2 = pl.subplot(122,projection=proj,extent=ext)
ax2.coastlines(linewidth=0.5,resolution='50m')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)

cs = ax2.contourf(lon2,lat2,Mx.T,transform=ccrs.PlateCarree(),cmap='inferno_r',
                 levels=levs,extend='max')
pl.title('(b) latest '+name)
GridLines(ax2,False,True,False,True)

f = pl.gcf()
colax = f.add_axes([0.1,0.1,0.8,0.03])                   
cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
cb.set_ticks(levs)
ticklabs = pl.asarray(levs)#; ticklabs = ticklabs/1000
cb.set_ticklabels(['1 Jan','1 Feb','1 Mar','1 Apr','1 May','1 Jun','1 Jul',
                   '1 Aug','1 Sep','1 Oct','1 Nov'])
cb.ax.tick_params(labelsize=14)
cb.update_ticks()#; cb.ax.set_aspect(0.09)
#cb.set_label('10$^3$ kg/s/steradian',fontsize=20)

pl.tight_layout()
pl.subplots_adjust(top=1,bottom=0.06,left=0.05,right=0.95)
pl.savefig(indecis+'figures/'+name+'_extrema.png')
#region = [(4.18,57.91),(13.27,57.91),(13.27,51.19),(4.18,51.19)]
#
#rPath = mplPath.Path(region)
#TF = pl.zeros([lon2.size,lat2.size])
#rmask = pl.zeros([lon2.size,lat2.size])
#rmask[:] = pl.float32('nan')
#
#for i in range(lon2.size):
#        for j in range(lat2.size):
#            X = rPath.contains_point((lon2[i],lat2[j]))
#            TF[i,j] = X
#
#Y = pl.where(TF)
#rmask[Y[0],Y[1]] = 1
#
#
#areas_clip = AreasCalc()
#
#rdata = data[:,:,:]*rmask[None,:,:]
#rareas = areas_clip*rmask
#
#Q = pl.ones_like(data)
#f = pl.isnan(data)
#d = pl.where(f==True)
#Q[d[0],d[1],d[2]] = pl.float32('nan')
#
#W = pl.nansum(rdata*rareas,axis=(1,2))/pl.nansum(rareas)
#fig,ax = pl.subplots(1,1)#pl.figure(2)
#ax1 = pl.subplot(111)
#ax1.plot(pl.linspace(1950,2017,68),W,lw=2)
#ax1.grid(axis='y')
#pl.xlim(1950,2017)#; pl.ylim(90,150)
#pl.yticks([1,15,32,46])#,labels=['1 Jan','15 Jan','1 Feb','14 Feb'])
#ax1.set_yticklabels(['1 Jan','15 Jan','1 Feb','14 Feb'])
#
#pl.title(name+' Denmark/Holland/North Germany')
#pl.savefig(indecis+'figures/'+name+'_DHNG.png',dpi=350)