# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:19:07 2018

@author: np838619
"""

import pylab as pl
from netCDF4 import Dataset
import glob
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap, shiftgrid

pl.close('all')

clusdir = '/glusterfs/scenario/users/np838619/'

locs = pl.genfromtxt(clusdir+'weatherrescue/latlon1903_pc.txt')
#locs = pl.delete(locs,18,0)

dwr_2702 = pl.genfromtxt(clusdir+'weatherrescue/csv_fixed/dwr_1903_02_28.csv',delimiter=',')
dwr_2802 = pl.genfromtxt(clusdir+'weatherrescue/csv_fixed/dwr_1903_01_01.csv',delimiter=',')

morn_stns = [43,44,45,46,47,48,49,50]
even_stns = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
             31,33,38,52,57]

filenames = glob.glob(clusdir+'weatherrescue/20CR/*') # all filenames in dir
filenames = pl.asarray(filenames) # list into array
filenames = pl.sort(filenames) # get filenames in correct order

tempfiles = filenames[11:22]
precfiles = filenames[22:33]
presfiles = filenames[33:44]
rhumfiles = filenames[44:]

#tempdata = pl.zeros([180,91])

#for i in range(tempfiles.size):
ncfile = Dataset(tempfiles[3],'r')
temp = ncfile.variables['air'][:]
lat = ncfile.variables['lat'][:]
lon = ncfile.variables['lon'][:]
ncfile.close()

temp = pl.reshape(temp,(temp.shape[0]/4,4,temp.shape[1],temp.shape[2]))
temp_2702 = temp[58] - 273.15
temp_2702 = temp_2702[[1,-1]]

ncfile = Dataset(presfiles[3],'r')
pres = ncfile.variables['pres'][:]
ncfile.close()

pres = pl.reshape(pres,(pres.shape[0]/4,4,pres.shape[1],pres.shape[2]))
pres_2702 = pres[58]
pres_2702 = pres_2702[[1,-1]]

ncfile = Dataset(precfiles[3],'r')
prec = nc.variables['prate'][:]
plat = ncfile.variables['lat'][:]
plon = ncfile.variables['lon'][:]
ncfile.close()

prec = pl.reshape(prec,(prec.shape[0]/8,8,prec.shape[1],prec.shape[2]))
prec_2702 = pl.zeros([8,prec.shape[2],prec.shape[3]])
prec_2702[:6,:,:] = prec[58,2:,:,:]
prec_2702[6:,:,:] = prec[59,:2,:,:]
prec_2702 = pl.sum(prec_2702*3*3600,axis=0) # in mm

morn27 = (5./9.)*(dwr_2702[morn_stns,4]-32)#/0.029530
even27 = (5./9.)*(dwr_2802[even_stns,2]-32)#/0.029530

fig,ax = pl.subplots(1,2,figsize=(24,9))

proj = ccrs.Mercator()#central_longitude=5
ext = [-30,25,35,70]
norm = pl.Normalize(-5,15)
levels = pl.linspace(-6,20,14)
#lons,lats = pl.meshgrid(lon,lat)

T, lonx = shiftgrid(180.0, temp_2702, lon, start=False)
lons,lats = pl.meshgrid(lonx,lat)
m = Basemap(projection='cyl',llcrnrlon=-30,urcrnrlon=25,llcrnrlat=35,
                                           urcrnrlat=70,resolution='l')
X, Y = m(lons,lats)

for i in range(2):
    pl.subplot(1,2,i+1)
    #m = Basemap(projection='cyl',llcrnrlon=-30,urcrnrlon=25,llcrnrlat=35,
    #                                        urcrnrlat=70,resolution='l')
    m.drawcoastlines(color='darkgrey',zorder=1)
    m.drawcountries(color='darkgrey',zorder=1)
    m.drawmeridians(pl.arange(-30,24,10),dashes=[1,2],linewidth=0.5,zorder=0,labels=[1,1,1,1])
    ct = m.contour(X,Y,T[i],levels=levels,zorder=2)
    pl.clabel(ct,levels,inline=True,fmt="%.0f",zorder=2)
#    axx = pl.subplot(2,4,i+1,projection=proj)
#    axx.coastlines(resolution='110m')
#    axx.set_extent(ext,ccrs.PlateCarree())
#    ct = axx.contour(lons,lats,temp_2702[i],transform=ccrs.PlateCarree(),
#                levels=levels,colors='k')
#    axx.clabel(ct,levels,fontsize=10,fmt="%.0f",manual=True,inline=True)
    m.scatter(locs[:,2],locs[:,1],c='k',latlon=True,zorder=2)
    
    if i == 0.:
        for j in range(morn27.size):
            s = morn_stns[j]
            pl.annotate(str(round(morn27[j],1)),(locs[s,2],locs[s,1]),zorder=3)
        pl.xlabel('(a) 0600',fontsize=18,labelpad=23)
        m.drawparallels(pl.arange(35,71,5),dashes=[1,2],linewidth=0.5,zorder=0,labels=[1,0,0,0])
    elif i == 1.:
        for j in range(even27.size):
            s = even_stns[j]
            if pl.isnan(even27[j]) == True:
                continue
            else:
                pl.annotate(str(round(even27[j],1)),(locs[s,2],locs[s,1]),zorder=3)
        pl.xlabel('(b) 1800',fontsize=18,labelpad=23)
        m.drawparallels(pl.arange(35,71,5),dashes=[1,2],linewidth=0.5,zorder=0,labels=[0,1,0,0])

pl.tight_layout()
pl.subplots_adjust(left=0.03,right=0.97,wspace=0.04)
#pl.suptitle('2m temperature ($^\circ$C) 26/02/1903',fontsize=18)
#pl.savefig(clusdir+'weatherrescue/temp_comparison_26021903.png')

###############################################################################

fig,ax = pl.subplots(1,2,figsize=(24,9))

plevs = pl.linspace(940,1020,9)

morn27 = (dwr_2702[morn_stns,3])/0.029530
even27 = (dwr_2802[even_stns,1])/0.029530

for i in range(2):
    pl.subplot(1,2,i+1)
    m = Basemap(projection='cyl',llcrnrlon=-30,urcrnrlon=25,llcrnrlat=35,
                                            urcrnrlat=70,resolution='l')
    P, lonx = shiftgrid(180.0, pres_2702[i], lon, start=False)
    lons,lats = pl.meshgrid(lonx,lat)
    X, Y = m(lons,lats)
    m.drawcoastlines(color='darkgrey',zorder=1)
    m.drawcountries(color='darkgrey',zorder=1)
    m.drawmeridians(pl.arange(-30,24,10),dashes=[1,2],linewidth=0.5,zorder=0,labels=[1,1,1,1])
    ct = m.contour(X,Y,P/100,levels=plevs,zorder=2)
    pl.clabel(ct,plevs,inline=True,fmt="%.0f",zorder=2)
    
    m.scatter(locs[:,2],locs[:,1],c='k',latlon=True,zorder=2)
    
    if i == 0.:
        for j in range(morn27.size):
            s = morn_stns[j]
            pl.annotate(str(int(morn27[j])),(locs[s,2],locs[s,1]))
        pl.xlabel('(a) 0600',fontsize=18,labelpad=23)
        m.drawparallels(pl.arange(35,71,5),dashes=[1,2],linewidth=0.5,zorder=0,labels=[1,0,0,0])
    elif i == 1.:
        for j in range(even27.size):
            s = even_stns[j]
            if pl.isnan(even27[j]) == True:
                continue
            else:
                pl.annotate(str(int(even27[j])),(locs[s,2],locs[s,1]))
        pl.xlabel('(b) 1800',fontsize=18,labelpad=23)
        m.drawparallels(pl.arange(35,71,5),dashes=[1,2],linewidth=0.5,zorder=0,labels=[0,1,0,0])

pl.tight_layout()
pl.subplots_adjust(left=0.03,right=0.97,wspace=0.04)
#pl.suptitle('surface pressure (hPa) 26/02/1903',fontsize=18)
#pl.savefig(clusdir+'weatherrescue/pres_comparison_26021903.png')

###############################################################################
#fig,ax = pl.subplots(1,2,figsize=(24,9))

past24 = dwr_2802[:,-1]*25.4

#for i in range(2):
#    pl.subplot(1,2,i+1)
pl.figure(3,figsize=(16,16))
m = Basemap(projection='cyl',llcrnrlon=-30,urcrnrlon=25,llcrnrlat=35,
                                            urcrnrlat=70,resolution='l')
P, lonx = shiftgrid(180.0, prec_2702, plon, start=False)
lons,lats = pl.meshgrid(lonx,plat)
X, Y = m(lons,lats)
m.drawcoastlines(color='darkgrey',zorder=1)
m.drawcountries(color='darkgrey',zorder=1)
m.drawmeridians(pl.arange(-30,24,10),dashes=[1,2],linewidth=0.5,zorder=0,labels=[1,1,1,1])
ct = m.contour(X,Y,P,levels=pl.linspace(0,20,11),zorder=2)
pl.clabel(ct,levels=pl.linspace(0,20,11),fmt="%.0f",inline=True,zorder=2)

m.scatter(locs[:,2],locs[:,1],c='k',latlon=True,zorder=2)

for j in range(past24.shape[0]):
    #s = even_stns[j]
    if pl.isnan(past24[j]) == True:
        continue
    else:
        pl.annotate(str(round(past24[j],1)),(locs[j,2],locs[j,1]))

pl.tight_layout()