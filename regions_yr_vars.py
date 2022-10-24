# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:25:56 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
from scipy import stats
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
from matplotlib.colors import Normalize
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pcraig_funcs as pc

pl.close('all')

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'
teleind = 'nao'
name = 'gsr'

tele = pd.read_csv(indecis+teleind+'_index.tim',header=5,delim_whitespace=True)
tele = pl.asarray(tele)
tele = tele[:]

tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
#tele_sns = pl.zeros([tele.shape[0],4])
#tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
#tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
#tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
#tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
#tele_sns[0,0] = pl.float32('nan')

# need to take mean tele index over April to October for gsr
tele_ao = pl.mean(tele[:,3:10],axis=1)

ncfile = xr.open_dataset(ncdir+name+'_year.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
#months = pl.asarray([i[5:7] for i in time])

#ind = pl.where((years=='1979') & (months=='04'))
#ind = 1#ind[0][0]

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data = data.values[:,70:326,34:200]

# upper-left, upper-right,lower-right, lower-left
# (lon, lat)
sca = [(19,72),(32,72),(32,58.6),(26.1,54.3),(4.5,54.3),(4.5,62.3)] # Scandinavia
bal = [(13.5,46.5),(30,46.5),(26.6,40.6),(23,39.9),(24.6,37),(20.9,37),
                                               (19,41.5),(13.7,45.1)] # Balkans
ibr = [(-10,44.3),(-1.4,43.5),(4.1,42.2),(-2,35.6),(-10,35.6)] # Iberian peninsula
ita = [(7,45.9),(11.8,47.1),(13.2,46.5),(12.8,44.6),(18.8,40.4),(15.9,37.5),
                                       (14.8,40),(8.8,44),(7.2,43.9)] # Italy
gbi = [(-10.5,60),(2,60),(2,50),(-10.5,50)] # Great Britian and Ireland

regions = [sca,bal,ibr,ita,gbi]

#proj = ccrs.PlateCarree()
#ax = pl.axes(projection=ccrs.PlateCarree(),
#             extent=[-30,40,30,70])
#
#ax.coastlines(resolution='50m',linewidth=0.5,color='grey')
#borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
#                                           '50m',edgecolor='grey',
#                                        facecolor='none')
#ax.add_feature(borders_50m,linewidth=0.5,zorder=5)


patches = []
for i in range(len(regions)):
    polygon = Polygon(regions[i],fc='none', ec='orangered')
    patches.append(polygon)

p = PatchCollection(patches)
p.set_edgecolor('k')
p.set_facecolor('none')
ax.add_collection(p)

#ax.annotate('GBI',gbi[0],fontsize=12)
#ax.annotate('IBR',ibr[0],fontsize=12)
#ax.annotate('BAL',(bal[1][0]-4,bal[1][1]),fontsize=12)
#ax.annotate('SCA',sca[-2],fontsize=12)
#ax.annotate('ITA',(ita[5][0],ita[5][1]-1.5),fontsize=12)
#
#pl.tight_layout()
#pl.savefig(indecis+'regions.png')

rPath = mplPath.Path(sca)
TF = pl.zeros([lon2.size,lat2.size])
rmask = pl.zeros([lon2.size,lat2.size])
rmask[:] = pl.float32('nan')

for i in range(lon2.size):
        for j in range(lat2.size):
            X = rPath.contains_point((lon2[i],lat2[j]))
            TF[i,j] = X

Y = pl.where(TF)
rmask[Y[0],Y[1]] = 1

#rdata = pl.nan_to_num(data_mn)*rmask
#wherez = pl.where(rdata==0)
#rdata[wherez[0],wherez[1]] = pl.float32('nan')

#pl.figure(2)
#pl.imshow(pl.flipud(rdata.T))
#pl.tight_layout()

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
#data_clip = data.values[ind:,61:326,34:187]

rdata = data[:,:,:]*rmask[None,:,:]
rareas = areas_clip*rmask

Q = pl.ones_like(data)
f = pl.isnan(data)
d = pl.where(f==True)
Q[d[0],d[1],d[2]] = pl.float32('nan')

#P = pl.average(rdata[0],weights=pl.nan_to_num(rareas))
W = pl.zeros([data.shape[0]])
W[0] = pl.float32('nan')
 
for i in range(data.shape[0]): # loop over years
    W[i] = pl.nansum(rdata[i]*rareas)/(pl.nansum(rareas*Q[i]))

#W = pl.reshape(W,newshape=(W.size/4,4))

#sns_labs = ['DJF','MAM','JJA','SON']
c = ['blue','purple','r','k']
ls = ['-','--','-','--']
lw = [2,1,2,1]
zo = [5,1,5,1]

pl.figure(2,figsize=(12,5))
#ax1 = pl.subplot(111)
#for i in range(4):
pl.plot(pl.linspace(1950,2017,68),W,#label=sns_labs[i],
            color=c[0],ls=ls[0],zorder=zo[0],lw=lw[0])
pl.grid(axis='y',ls='--',color='grey')
pl.xlim(1950,2017)
pl.xticks(years[::4][::5].astype(int))
#pl.ylim(0,30)
pl.xlabel('years',fontsize=14)
pl.ylabel('$^\circ$C',fontsize=13)
pl.legend(ncol=2,fontsize=13)
#pl.title('IBR mean of daily mean temperature (gtg)',fontsize=16)
pl.tight_layout()
#pl.savefig(indecis+'figures/'+name+'_ibr_sns_ts.png')

#inx = tele_ao.shape[0]-(ind*4)/12
print stats.pearsonr(W,tele_ao)
#for s in range(1,4):
#    cor = stats.pearsonr(W[:,s],tele_ao[:,s])
#    print cor