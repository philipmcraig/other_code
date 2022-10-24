# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:20:53 2018

@author: np838619
"""

from __future__ import division
import pylab as pl
from netCDF4 import Dataset
import cartopy
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

exec(open('/home/np838619/Trajectory/trajfuncs.py').read())

pl.close('all')
clusdir = '/glusterfs/scenario/users/np838619/'
sodadir = clusdir + 'SODA/'

years = pl.linspace(1980,2014,35).astype('int')

filenames = PrintFiles(sodadir,'soda3.4.2')
filenames = pl.sort(filenames)

sst = pl.zeros([len(filenames),12,330,720])

for i in range(len(filenames)):
    nc = Dataset(sodadir+filenames[i],'r')
    if i == 0.:
        lat = nc.variables['yt_ocean'][:]
        lon = nc.variables['xt_ocean'][:]
        depth = nc.variables['st_ocean'][:]
    sst[i] = nc.variables['temp'][:,0,:,:]
    nc.close()

JJA = pl.mean(sst[:,5:8],axis=1)
JJA_mn = pl.mean(JJA,axis=0)
JJA_an = JJA - JJA_mn
#a = pl.where(JJA<28.5)
#JJA[a] = pl.float32('nan')

#fix,ax = pl.subplots(7,5,figsize=(20,12))

proj = ccrs.PlateCarree()
lons,lats = pl.meshgrid(lon,lat)
inds = [16,26,31]; norm = pl.Normalize(28.5,31)
levels = [28.5,29,29.5,30,30.5,31]#,31.5,32]
xt = [-120,-90,-60,-30,-15]; yt = [0,10,20,30,40]

#for i in range(35):
#    axx = pl.subplot(7,5,i+1,projection=proj)
#    axx.coastlines(); axx.set_extent([-120.5,-29.5,-0.1,40.1],crs=ccrs.PlateCarree())
#    cs = axx.contourf(lons,lats,JJA[i],cmap='plasma',norm=norm,
#                 transform=ccrs.PlateCarree(),levels=levels,extend='max')
#    axx.annotate(str(years[i]),(-50,32),fontsize=18)
#    gx = axx.gridlines(draw_labels=True,crs=ccrs.PlateCarree())
#    gx.xlocator = mticker.FixedLocator(xt)
#    gx.ylocator = mticker.FixedLocator(yt)
#    if i in (0,5,10,15,20,25,30):
#        gx.ylabels_right = False#; gx.xlabels_top = False
#    if i in (1,2,3,6,7,8,11,12,13,16,17,18,21,22,23,26,27,28,31,32,33):
#        gx.ylabels_right = False; gx.ylabels_left = False
#    if i in (4,9,14,19,24,29,34):
#         gx.ylabels_left = False
#    if i > 4:
#        gx.xlabels_top = False
#    if i < 30:
#        gx.xlabels_bottom = False
#    gx.xformatter = LONGITUDE_FORMATTER; gx.yformatter = LATITUDE_FORMATTER
#    gx.xlabel_style = {'size': 13}; gx.ylabel_style = {'size': 13}
#
#f = pl.gcf()
#colax = f.add_axes([0.3,0.05,0.4,0.03])
#cb=pl.colorbar(cs,cax=colax,orientation='horizontal')
#cb.set_label('SST ($^\circ$C)',fontsize=18,labelpad=-2)
#cb.ax.tick_params(labelsize=14)
#
#pl.subplots_adjust(wspace=-0.2,hspace=0.12,top=0.97,bottom=0.11,left=0.02,right=0.98)
#pl.tight_layout()
#pl.savefig(sodadir+'awp_sst_JJA_iv.png')
#JJA_mn = pl.mean(JJA,axis=0)

fig,ax = pl.subplots(1,1,figsize=(15,9))

ax = pl.axes(projection=proj)
ax.set_extent([-120.5,-18.5,-0.1,40.1],crs=ccrs.PlateCarree())
ax.coastlines()

cs = ax.contourf(lons,lats,JJA_mn,cmap='plasma',norm=norm,
                 transform=ccrs.PlateCarree(),levels=levels,extend='max')
f = pl.gcf()
colax = f.add_axes([0.3,0.15,0.4,0.03])
cb=pl.colorbar(cs,cax=colax,orientation='horizontal')
cb.set_label('SST ($^\circ$C)',fontsize=18,labelpad=-2)
cb.ax.tick_params(labelsize=14)

cl = ax.contour(lons,lats,JJA[15],levels=[28.5],colors='k',linewidths=[2])
cl.collections[0].set_label('1995')
cl = ax.contour(lons,lats,JJA[25],levels=[28.5],colors='b',linewidths=[2])
cl.collections[0].set_label('2005')
cl = ax.contour(lons,lats,JJA[30],levels=[28.5],colors='r',linewidths=[2])
cl.collections[0].set_label('2010')

ax.legend(fontsize=22)

gx = ax.gridlines(draw_labels=True,crs=ccrs.PlateCarree())
gx.xlocator = mticker.FixedLocator(xt)
gx.ylocator = mticker.FixedLocator(yt)
gx.xformatter = LONGITUDE_FORMATTER; gx.yformatter = LATITUDE_FORMATTER
gx.xlabels_bottom = False
gx.xlabel_style = {'size': 18}; gx.ylabel_style = {'size': 18}

pl.tight_layout()
pl.subplots_adjust(left=0.05,right=0.95)
#pl.savefig(sodadir+'awp_sst_JJA_3max.png')