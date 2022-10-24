# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:24:14 2018

@author: np838619
"""

from __future__ import division
import pylab as pl
from netCDF4 import Dataset

exec(open('/home/np838619/PminusE_data/ERA_Int/functions.py').read())

pl.close('all')
clusdir = '/glusterfs/scenario/users/np838619/'
sodadir = clusdir + 'SODA/'

maskfile = Dataset(sodadir+'soda_basin_masks.nc','r')
lat = maskfile.variables['lat'][:]
lon = maskfile.variables['lon'][:]
atlmask = maskfile.variables['Atlantic mask'][:]
indmask = maskfile.variables['Indian mask'][:]
pacmask = maskfile.variables['Pacific mask'][:]
arcmask = maskfile.variables['Arctic mask'][:]
soumask = maskfile.variables['Southern mask'][:]
maskfile.close()

filenames = PrintFiles(sodadir+'soda_basins/','soda')
filenames = pl.sort(filenames)

years = pl.linspace(1980,2014,35)

atl_sss = pl.zeros([len(years),12,330,720])
ind_sss = pl.zeros_like(atl_sss)
pac_sss = pl.zeros_like(atl_sss)
arc_sss = pl.zeros_like(atl_sss)
sou_sss = pl.zeros_like(atl_sss)

for name in range(len(filenames)):
    ncfile = Dataset(sodadir+'soda_basins/'+filenames[name],'r')
    if name == 0.:
        lat = ncfile.variables['lat'][:]
        lon = ncfile.variables['lon'][:]
    atl_sss[name] = ncfile.variables['Atlantic'][:]
    ind_sss[name] = ncfile.variables['Indian'][:]
    pac_sss[name] = ncfile.variables['Pacific'][:]
    arc_sss[name] = ncfile.variables['Arctic'][:]
    sou_sss[name] = ncfile.variables['Southern'][:]
    ncfile.close()

atl_yrs = pl.nanmean(atl_sss,axis=1)
ind_yrs = pl.nanmean(ind_sss,axis=1)
pac_yrs = pl.nanmean(pac_sss,axis=1)
arc_yrs = pl.nanmean(arc_sss,axis=1)
sou_yrs = pl.nanmean(sou_sss,axis=1)

newnc = Dataset(sodadir+'soda_yearlymeans.nc','w')

lat_dim = newnc.createDimension('lat',lat.size)
lon_dim = newnc.createDimension('lon',lon.size)
lat_in = newnc.createVariable('lat',pl.float64,('lat',))
lat_in.units = 'degrees_north'
lat_in.long_name = 'latitude'
lon_in = newnc.createVariable('lon',pl.float64,('lon',))
lon_in.units = 'degrees_east'
lon_in.long_name = 'longitude'
lat_in[:] = lat[:]
lon_in[:] = lon[:]

time_dim = newnc.createDimension('time',len(years))
time = newnc.createVariable('time',pl.float64,('time',))
time.units = 'years'
time.long_name = 'years'

atl = newnc.createVariable('Atlantic',pl.float64,('time','lat','lon'))
atl.units = 'psu'
atl.standard_name = 'Atlantic sea surface salinity'
atl[:,:,:] = atl_yrs[:,:,:]

ind = newnc.createVariable('Indian',pl.float64,('time','lat','lon'))
ind.units = 'psu'
ind.standard_name = 'Indian sea surface salinity'
ind[:,:,:] = ind_yrs[:,:,:]

pac = newnc.createVariable('Pacific',pl.float64,('time','lat','lon'))
pac.units = 'psu'
pac.standard_name = 'Pacific sea surface salinity'
pac[:,:,:] = pac_yrs[:,:,:]

arc = newnc.createVariable('Arctic',pl.float64,('time','lat','lon'))
arc.units = 'psu'
arc.standard_name = 'Arctic sea surface salinity'
arc[:,:,:] = arc_yrs[:,:,:]

sou = newnc.createVariable('Southern',pl.float64,('time','lat','lon'))
sou.units = 'psu'
sou.standard_name = 'Southern sea surface salinity'
sou[:,:,:] = sou_yrs[:,:,:]

newnc.close()