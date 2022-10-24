# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:07:03 2018

@author: np838619
"""

from __future__ import division
import pylab as pl
from netCDF4 import Dataset
from scipy.stats import pearsonr, linregress
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs

clusdir = '/glusterfs/scenario/users/np838619/'
eccodir = clusdir + 'ECCO/'
eccoyears = pl.linspace(1992,2014,23)
#Y = pl.where(years==1992); Y = Y[0]

eccofile = Dataset(eccodir+'release3/eccor3_sss_basins.nc','r')
atl_ecc = eccofile.variables['Atlantic'][:]
ind_ecc = eccofile.variables['Indian'][:]
pac_ecc = eccofile.variables['Pacific'][:]
arc_ecc = eccofile.variables['Arctic'][:]
sou_ecc = eccofile.variables['Southern'][:]
eccolat = eccofile.variables['lat'][:]
eccolon = eccofile.variables['lon'][:]
eccofile.close()

# # change shape of arrays to years x months x lat x lon:
atl_ecc = pl.reshape(atl_ecc,(atl_ecc.shape[0]/12,12,eccolat.size,eccolon.size))
ind_ecc = pl.reshape(ind_ecc,(ind_ecc.shape[0]/12,12,eccolat.size,eccolon.size))
pac_ecc = pl.reshape(pac_ecc,(pac_ecc.shape[0]/12,12,eccolat.size,eccolon.size))
arc_ecc = pl.reshape(arc_ecc,(arc_ecc.shape[0]/12,12,eccolat.size,eccolon.size))
sou_ecc = pl.reshape(sou_ecc,(sou_ecc.shape[0]/12,12,eccolat.size,eccolon.size))

# take mean along months axis:
atl_ecc = pl.mean(atl_ecc,axis=1); ind_ecc = pl.mean(ind_ecc,axis=1)
pac_ecc = pl.mean(pac_ecc,axis=1); arc_ecc = pl.mean(arc_ecc,axis=1)
sou_ecc = pl.mean(sou_ecc,axis=1)

trend = pl.zeros_like(atl_ecc[0]); sigs = pl.zeros_like(trend)

for i in range(eccolat.size):
    for j in range(eccolon.size):
        r = linregress(eccoyears,atl_ecc[:,i,j])
        trend[i,j] = r[0]; sigs[i,j] = r[3]
z = pl.where(sigs>0.1)
sigs[z[0],z[1]] = pl.float64('nan')

proj = ccrs.PlateCarree(central_longitude=-30)
#ax = pl.subplot(projection=proj)
ext = [-100,40,-40,70]
ax1.set_extent(ext,crs=ccrs.PlateCarree()); ax1.coastlines()
lons,lats = pl.meshgrid(eccolon,eccolat)
norm = pl.Normalize(-0.03,0.03,clip=False)
levels = [-0.03,-0.01,-0.005,0.005,0.01,0.03]
cs1 = ax1.contourf(lons,lats,trend,transform=ccrs.PlateCarree(),norm=norm,
                 levels=levels,cmap='seismic',extend='both')