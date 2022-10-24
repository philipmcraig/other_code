# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 15:30:02 2020

@author: qx911590
"""

import numpy as np
from scipy import stats
import xarray as xr

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

name = 'ta_o'

ncfile = xr.open_dataset(ncdir+name+'_year.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

# set up empty array of size longitude x latitude
trend = np.zeros([lon.size,lat.size])
# set up empty array of same size for significance values
P = np.zeros_like(trend)

for i in range(lon.size): # loop over longitude
    for j in range(lat.size): # loop over latitude
        # calculate linear regression of data time series against time
        out = stats.linregress(np.linspace(1,68,68),data[:,i,j])
        # trend is the zero index in out
        # can also get trend with out.slope
        trend[i,j] = out[0]
        # p-value for statistical significance is index 3
        # can also get p-value with out.pvalue
        sig = out[3]
        if sig < 0.05:
            P[i,j] = sig