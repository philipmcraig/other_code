# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:22:43 2020

@author: pmcraig
"""

from __future__ import division
import pylab as pl
import pandas as pd
import xarray as xr
from scipy import stats
import glob
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import Normalize
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

def RegionCalc(vertices,lon2,lat2,data):
    """
    """
    rPath = mplPath.Path(vertices)
    TF = pl.zeros([lon2.size,lat2.size])
    rmask = pl.zeros([lon2.size,lat2.size])
    rmask[:] = pl.float32('nan')
    
    for i in range(lon2.size):
            for j in range(lat2.size):
                X = rPath.contains_point((lon2[i],lat2[j]))
                TF[i,j] = X
    
    Y = pl.where(TF)
    rmask[Y[0],Y[1]] = 1
    
    areas = AreasCalc()
    
    rdata = data[:,:,:]*rmask[:,:]#None,
    rareas = areas*rmask
    
    Q = pl.ones_like(data)
    f = pl.isnan(data)
    d = pl.where(f==True)
    Q[d[0],d[1],d[2]] = pl.float32('nan')
    
    #P = pl.average(rdata[0],weights=pl.nan_to_num(rareas))
    W = pl.zeros([data.shape[0]])
    W[0] = pl.float32('nan')
     
    for i in range(data.shape[0]): # loop over years
        W[i] = pl.nansum(rdata[i]*rareas)/(pl.nansum(rareas*Q[i]))
    
    return W

pl.close('all')

home = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
indecis = ncasdir + 'INDECIS/'

ncfile = xr.open_dataset(indecis+'ncfiles/ta_o_year.nc')
data = xr.DataArray(ncfile.variables['ta_o'][:])
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data  = data.values[:,70:326,34:200]

gbi = [(-7.33,58.73),(-1.45,58.73),(2.08,52.83),(1.58,51.03),(-5.46,49.88),
       (-10.99,52.00)]
south_scand = [(4.82,58.00),(4.38,62.04),(14.72,63.20),(17.70,61.82),(19.32,59.70),
               (14.62,55.43),(12.79,55.37),(10.97,54.35),(8.53,54.35),(7.39,58.00)]
italy_aegean = [(7.67,43.74),(6.86,45.87),(10.51,46.82),(13.66,46.63),
                (17.22,46.03),(20.80,43.26),(21.87,42.37),(21.94,39.01),
                (20.65,38.79),(19.22,40.35),(16.00,37.71),(13.06,41.09),
                (9.25,43.90)]
iberia = [(-8.95,43.93),(-1.95,43.74),(4.28,42.04),(-1.2,35.99),(-5.55,35.23),
          (-10.09,36.93)]
ukraine = [(23.52,51.56),(25.73,52.02),(30.58,51.33),(30.95,52.12),(33.44,52.42),
           (35.65,50.45),(40.05,49.65),(39.72,47.80),(38.28,47.62),(38.23,47.14),
            (34.71,46.16),(36.55,45.38),(33.83,44.37),(32.71,45.51),(33.61,46.10),
            (31.86,46.28),(31.38,46.66),(29.66,45.41),(28.26,45.46),(29.17,46.41),
            (30.13,46.43),(29.20,48.04),(27.54,48.51),(25.11,47.82),(22.85,48.06),
             (22.13,48.44),(22.68,49.54),(24.07,50.55)]
romania = [(20.20,46.16),(21.17,46.41),(22.84,48.06),(24.95,47.79),(26.57,48.29),
           (28.23,46.62),(28.20,45.45),(29.67,45.21),(28.57,43.72),(22.59,44.21)]
hungary = [(16.18,46.91),(17.09,48.00),(18.73,47.87),(20.54,48.55),(22.14,48.44),
           (22.92,47.99),(22.03,47.61),(21.10,46.23),(19.60,46.17),(18.07,45.78)]
bulgaria = [(22.66,44.20),(23.03,43.80),(25.62,43.64),(27.02,44.17),(28.57,43.71),
            (27.97,42.03),(26.72,26.06),(26.12,41.34),(24.44,41.56),(22.93,41.32),
            (22.93,41.32),(22.32,42.31),(22.97,43.18),(22.31,43.81)]
serbia = [(18.88,45.91),(20.23,46.19),(22.14,44.51),(22.38,44.73),(22.70,44.48),
          (22.61,44.21),(22.37,43.79),(23.02,43.16),(22.38,42.34),(21.60,42.29),
        (21.76,22.70),(20.79,43.27),(20.32,42.86),(19.24,43.53),(19.11,44.50),
        (19.32,44.91),(19.14,44.95)]
moldova = [(26.60,48.26),(27.58,48.49),(29.11,47.98),(30.14,46.12),(29.20,46.55),
           (28.92,46.48),(28.97,46.05),(28.23,45.48),(28.22,46.66),(26.90,48.24)]

regions = [romania,hungary,bulgaria,serbia,ukraine,moldova]
labels = ['romania','hungary','bulgaria','serbia','ukraine',
            'moldova']
clist = ['b','darkgoldenrod','green','k','magenta','darkred']
#mrks = ['None','.','o','s','^','h']

x = pl.linspace(1,68,68)
years = pl.linspace(1950,2017,68).astype(int)
V = pl.zeros([len(regions),len(x)])
lines = []

for i in range(len(regions)):
    V[i] = RegionCalc(regions[i],lon2,lat2,data)

#ax = pl.subplots()
for i in range(len(regions)):
    if i == 0:
        lw = 3
        zorder = 10
        #c = 'b'
    else:
        lw = 1.25
        zorder = 1
       # c = 'k'
    l = pl.plot(years,V[i,:],color=clist[i],lw=lw,zorder=zorder)
    lines.append(l[0])

pl.grid(axis='y',ls='--',color='grey')
pl.ylabel('$^\circ$C',fontsize=15)
pl.xlim(1950,2017); pl.ylim(13,19)
pl.legend(handles = lines,
          labels = labels,ncol=3,loc=0)

pl.tick_params(axis='x',top=True,direction='in',pad=7)
#pl.tight_layout()
pl.subplots_adjust(top=0.97,bottom=0.06,left=0.09,right=0.99)
#pl.savefig(indecis+'figures/area_averaged_ta_o_countries.png',dpi=350)

diff = pl.zeros([5,68])
for i in range(5):
    diff[i] = V[0] - V[i+1]

pl.figure(2)
pl.plot(years,diff.T)