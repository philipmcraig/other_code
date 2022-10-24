# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:48:31 2018

@author: np838619
"""

from __future__ import division
import pylab as pl
from netCDF4 import Dataset
import cartopy
import cartopy.crs as ccrs

#exec(open('/home/np838619/PminusE_data/ERA_Int/functions.py').read())

pl.close('all')
clusdir = '/glusterfs/scenario/users/np838619/'
eccodir = clusdir + 'ECCO/'

years = pl.linspace(1992,2011,20)

maskfile = Dataset(eccodir+'ecco_basin_masks.nc','r')
lat = maskfile.variables['lat'][:]
lon = maskfile.variables['lon'][:]
atlmask = maskfile.variables['Atlantic mask'][:]
indmask = maskfile.variables['Indian mask'][:]
pacmask = maskfile.variables['Pacific mask'][:]
arcmask = maskfile.variables['Arctic mask'][:]
soumask = maskfile.variables['Southern mask'][:]
maskfile.close()

eccofile = Dataset(eccodir+'release3/SALT.0001.nc','r')
salt = eccofile.variables['SALT'][:]
lat = eccofile.variables['lat'][:,0]
lon = eccofile.variables['lon'][0]
depth = eccofile.variables['dep'][:]
eccofile.close()

#salt = pl.reshape(salt,(20,12,salt.shape[1],salt.shape[2],salt.shape[3]))

sss = salt[:-12,0,:,:]
sss_yrs = pl.mean(sss,axis=1)

atl_sss = sss*atlmask
ind_sss = sss*indmask
pac_sss = sss*pacmask
arc_sss = sss*arcmask
sou_sss = sss*soumask

#newnc = Dataset(eccodir+'release3/ecco_sss_basins.nc','w')
#
#lat_dim = newnc.createDimension('lat',lat.size)
#lon_dim = newnc.createDimension('lon',lon.size)
#lat_in = newnc.createVariable('lat',pl.float64,('lat',))
#lat_in.units = 'degrees_north'
#lat_in.long_name = 'latitude'
#lon_in = newnc.createVariable('lon',pl.float64,('lon',))
#lon_in.units = 'degrees_east'
#lon_in.long_name = 'longitude'
#lat_in[:] = lat[:]
#lon_in[:] = lon[:]
#
#time_dim = newnc.createDimension('time',sss.shape[0])
#time = newnc.createVariable('time',pl.float64,('time',))
#time.units = 'months'
#time.long_name = 'months'
#
#atl = newnc.createVariable('Atlantic',pl.float64,('time','lat','lon'))
#atl.units = 'psu'
#atl.standard_name = 'Atlantic sea surface salinity'
#atl[:,:,:] = atl_sss[:,:,:]
#
#ind = newnc.createVariable('Indian',pl.float64,('time','lat','lon'))
#ind.units = 'psu'
#ind.standard_name = 'Indian sea surface salinity'
#ind[:,:,:] = ind_sss[:,:,:]
#
#pac = newnc.createVariable('Pacific',pl.float64,('time','lat','lon'))
#pac.units = 'psu'
#pac.standard_name = 'Pacific sea surface salinity'
#pac[:,:,:] = pac_sss[:,:,:]
#
#arc = newnc.createVariable('Arctic',pl.float64,('time','lat','lon'))
#arc.units = 'psu'
#arc.standard_name = 'Arctic sea surface salinity'
#arc[:,:,:] = arc_sss[:,:,:]
#
#sou = newnc.createVariable('Southern',pl.float64,('time','lat','lon'))
#sou.units = 'psu'
#sou.standard_name = 'Southern sea surface salinity'
#sou[:,:,:] = sou_sss[:,:,:]
#
#newnc.close()