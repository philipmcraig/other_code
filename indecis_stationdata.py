# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:22:58 2020

@author: pmcraig

Comparing ECA&D station data (non-blended & blended) to INDECIS grid & ERA5
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
import pcraig_funcs as pc

class MidpointNormalize(Normalize):
    """Found this on the internet. Use to centre colour bar at zero.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        a, b = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return pl.ma.masked_array(pl.interp(value, a, b))


def GridLines(ax,top,bottom,left,right):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([18,20,25,30,32])
    gl.ylocator = mticker.FixedLocator([20,45,48,50])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':10}
    gl.ylabel_style = {'color': 'k','size':10}
    
    return None

def dmslat_conv(s):
    degrees = s[1:3]
    minutes = s[4:6]
    seconds = s[7:]

    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    
    return dd

def dmslon_conv(s):
    degrees = s[2:4]
    minutes = s[5:7]
    seconds = s[8:]
    
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if s[0] == '-':
        dd = -1*dd
    
    return dd

def Nonblended(dirpath,cn_name):
    """Read the ECA&D non-blended station data.
    
    Args:
        dirpath (string): directory containing data
        cn_name (string): two-character ISO country code in caps
    
    Returns:
        ta_o (array): mean growing season temperature, stations x years
    """
    # metadata from sources.txt
    df = pd.read_csv(dirpath+'TG_nonblended/sources.txt',header=20)
    data = pl.asarray(df)
    
    country = pl.where(data[:,2]==cn_name)[0] # all stations for this country
    
    stations = data[country,1] # select the station names
    # remove trailing blank spaces from the end of string:
    stations = pl.asarray([i.strip() for i in stations])
    print stations
    
    sid = data[country,0] # station ID numbers
    
    # extract station co-ordinates & convert to decimal form
    lats = pl.asarray([dmslat_conv(i) for i in data[country,3]])
    lons = pl.asarray([dmslon_conv(i) for i in data[country,4]])
    
    ta_o = pl.zeros([len(stations),68]) # 68 years in INDECIS
    
    for s in range(len(stations)):
        SOUID = sid[s] # SOUID is the ID used in nonblended station files
        statdata = pd.read_csv(dirpath+'TG_nonblended/TG_SOUID'+str(SOUID)+'.txt',skiprows=18)
        statdata = pd.DataFrame(statdata)
        
        temp = statdata[statdata.columns[3]]
        missing = pl.where(temp==-9999)[0]
        temp[missing] = pl.float32('nan')
        temp = temp/10.
        
        date = statdata[statdata.columns[2]]
        d = pl.array(date)
        years = [str(i)[:4] for i in d]; years = pl.array(years).astype(int)
        months = [str(i)[4:6] for i in d]; months = pl.array(months).astype(int)
        days = [str(i)[6:] for i in d]; days = pl.array(days).astype(int)
        
        if years.min()<=1950 and years.max()>=2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(1950,2017,68)
        elif years.min()==years.max() and years.max()>2017:
            pass
        elif years.min()>1950 and years.max()>=2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(years.min(),2017,2017-years.min()+1)
        elif years.min()<=1950 and years.max()<2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(1950,years.max(),years.max()-1950+1)
        elif years.min()>1950 and years.max()<2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(years.min(),years.max(),years.max()-years.min()+1)
        
        fullyears = pl.linspace(1950,2017,68)
        Z = len(fullyears)-len(Y)
        monthly_means = pl.zeros([68,12]); monthly_means[:,:] = pl.float32('nan')
        
        if years.min()==years.max() and years.max()>2017:
            ta_o[s,:] = pl.float32('nan')
        else:
            for yr in range(len(Y)):
                #Z = pl.where(fullyears==Y[yr])[0][0]
                for mn in range(1,13):
                    #Y = years[y1:y2][yr]
                    #M = months[y1:y2][mn]
                    X = pl.where((years[y1:y2+1]==Y[yr]) & (months[y1:y2+1]==mn))[0]
                    monthly_means[yr+Z,mn-1] = pl.mean(temp[y1:y2+1][X[0]:X[-1]+1])
            
            ta_o[s,:] = pl.nanmean(monthly_means[:,3:10],axis=1)
        #W = pl.where(pl.isnan(ta_o[s,:11]==False))
        #if W[0].size > 0:
        #    print stations[s]

    return ta_o, lats, lons, stations

def Blended(dirpath,cn_name):
    """Read the ECA&D blended station data
    """
    # metadata from stations.txt
    df = pd.read_csv(dirpath+'TG_nonblended/stations.txt',header=20)
    data = pl.asarray(df)
    
    country = pl.where(data[:,2]==cn_name)[0] # all stations for this country
    
    stations = data[country,1] # select the station names
    # remove trailing blank spaces from the end of string:
    stations = pl.asarray([i.strip() for i in stations])
    
    sid = data[country,0] # station ID numbers
    
    # extract station co-ordinates & convert to decimal form
    lats = pl.asarray([dmslat_conv(i) for i in data[country,3]])
    lons = pl.asarray([dmslon_conv(i) for i in data[country,4]])
    
    ta_o = pl.zeros([len(stations),68])
    
    for s in range(len(stations)):
        STAID = str(sid[s]).zfill(6) # add in the leading zeros
        statdata = pd.read_csv(indecis+'TG_blended/TG_STAID'+str(STAID)+'.txt',skiprows=20)
        statdata = pd.DataFrame(statdata)
        
        temp = statdata[statdata.columns[3]]
        missing = pl.where(temp==-9999)[0]
        temp[missing] = pl.float32('nan')
        temp = temp/10.
        
        date = statdata[statdata.columns[2]]
        d = pl.array(date)
        years = [str(i)[:4] for i in d]; years = pl.array(years).astype(int)
        months = [str(i)[4:6] for i in d]; months = pl.array(months).astype(int)
        days = [str(i)[6:] for i in d]; days = pl.array(days).astype(int)
        
        if years.min()<=1950 and years.max()>=2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(1950,2017,68)
        elif years.min()==years.max() and years.max()>2017:
            pass
        elif years.min()>1950 and years.max()>=2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(years.min(),2017,2017-years.min()+1)
        elif years.min()<=1950 and years.max()<2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(1950,years.max(),years.max()-1950+1)
        elif years.min()>1950 and years.max()<2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(years.min(),years.max(),years.max()-years.min()+1)
        
        fullyears = pl.linspace(1950,2017,68)
        Z = len(fullyears)-len(Y)
        monthly_means = pl.zeros([68,12]); monthly_means[:,:] = pl.float32('nan')
        
        if years.min()==years.max() and years.max()>2017:
            ta_o[s,:] = pl.float32('nan')
        else:
            for yr in range(len(Y)):
                #Z = pl.where(fullyears==Y[yr])[0][0]
                for mn in range(1,13):
                    #Y = years[y1:y2][yr]
                    #M = months[y1:y2][mn]
                    X = pl.where((years[y1:y2+1]==Y[yr]) & (months[y1:y2+1]==mn))[0]
                    monthly_means[yr+Z,mn-1] = pl.mean(temp[y1:y2+1][X[0]:X[-1]+1])
            
            ta_o[s,:] = pl.nanmean(monthly_means[:,3:10],axis=1)
    
    return ta_o, lats, lons, stations

def ERA5_data(dirpath,statlon,statlat):
    """Read the ERA5 t2m data & interpolate to station co-ordinates.
    """
    ncfile1 = xr.open_dataset(dirpath+'ERA5/era5_surf_mm_1950-1978.nc')
    era5lat = xr.DataArray(ncfile1.latitude)
    era5lon = xr.DataArray(ncfile1.longitude)
    t2m_early = xr.DataArray(ncfile1.t2m)
    ncfile1.close()
    
    ncfile2 = xr.open_dataset(dirpath+'ERA5/era5_surf_mm_1979_2017.nc')
    t2m_late = xr.DataArray(ncfile2.t2m)
    ncfile2.close()
    
    t2m = pl.zeros([t2m_early.shape[0]+t2m_late.shape[0],era5lat.size,
                                                        era5lon.size])
    
    t2m[:t2m_early.shape[0],:,:] = t2m_early.values
    t2m[t2m_early.shape[0]:,:,:] = t2m_late.values
    
    t2m = pl.reshape(t2m,newshape=(int(t2m.shape[0]/12),12,\
                                                    era5lat.size,era5lon.size))
    era5_tao = pl.mean(t2m[:,3:10,:,:],axis=1)
    
    era5_stats = pl.zeros([statlon.size,t2m.shape[0]])
    
    for i in range(era5_stats.shape[0]):
        point = (statlon[i],statlat[i])
        for j in range(era5_stats.shape[1]):
            era5_stats[i,j] = pc.BilinInterp(point,era5lon,era5lat,era5_tao[j]-273.15)
    
    return era5_tao, t2m, era5_stats, era5lat, era5lon

def INDECIS_data(dirpath,statlon,statlat):
    """Read the INDECIS gridded data & interpolate to station co-ordinates
    """
    ncfile = xr.open_dataset(dirpath+'ncfiles/ta_o_year.nc')
    indlat = xr.DataArray(ncfile.latitude)
    indlon = xr.DataArray(ncfile.longitude)
    ta_o = xr.DataArray(ncfile.ta_o)
    ncfile.close()
    
#    ta_o = pl.reshape(ta_o,newshape=(int(ta_o.shape[0]/12),12,\
#                                            indlon.size,indlat.size))
    ind_stats = pl.zeros([statlon.size,ta_o.shape[0]])
    
    for i in range(ind_stats.shape[0]):
        point = (statlon[i],statlat[i])
        for j in range(ind_stats.shape[1]):
            ind_stats[i,j] = pc.BilinInterp(point,indlon,indlat,ta_o[j].T)
    
    return ind_stats

pl.close('all')

home = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
indecis = ncasdir + 'INDECIS/'

orogfile = xr.open_dataset(ncasdir+'etopo05.nc')
orolat = xr.DataArray(orogfile.ETOPO05_Y)
orolon = xr.DataArray(orogfile.ETOPO05_X)
orog = xr.DataArray(orogfile.ROSE)
orogfile.close()
orolat = pl.flipud(orolat.values); orog = pl.flipud(orog.data)

#oro2 = orog.copy()
#t = pl.where(orog<0); oro2[t[0],t[1]]=0.

#nb_tao, nb_lats, nb_lons, stations = Nonblended(indecis,'RS')

bl_tao, bl_lats, bl_lons, stations = Blended(indecis,'MD')

era5_tao, t2m, era5_stats, era5lat, era5lon = ERA5_data(ncasdir,bl_lons,bl_lats)

ind_stats = INDECIS_data(indecis,bl_lons,bl_lats)

era5_mn = pl.nanmean(era5_tao,axis=0)
era5_trnd = pl.zeros_like(era5_tao[0])

#for i in range(era5_trnd.shape[0]):
#    for j in range(era5_trnd.shape[1]):
#        era5_trnd[i,j] = stats.linregress(pl.linspace(1,68,68),era5_tao[:,i,j]).slope

borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                        edgecolor='face',
                                        facecolor='w')#cfeature.COLORS['water'])
#levs = [-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,-0.01,0.01,0.05,0.1,0.15,0.2,0.25,0.3]

dflt_clrs = pl.rcParams['axes.prop_cycle'].by_key()['color']

clist = ['k']*bl_lons.size
clist[0] = dflt_clrs[0]
#clist[19] = dflt_clrs[1]
#clist[22] = dflt_clrs[2]
#clist[23] = dflt_clrs[3]
#clist[24] = dflt_clrs[4]
#clist[40] = dflt_clrs[5]

#elist = ['none']*nb_lons.size
#clist[3], clist[16], clist[23] = 'r', 'r', 'r'
#elist[16] = 'r'
#elist[23] = 'r'


romext = [19.5,30.5,42.5,49]
eurext = [-15,42,35,70]
cmap = pl.get_cmap('YlOrRd')
proj = ccrs.PlateCarree()

#fig, ax = pl.subplots(1,2,figsize=(9,4))
#
ax = pl.subplot(projection=proj,extent=romext)
ax.coastlines(linewidth=0.5,resolution='50m')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)
ax.pcolormesh(orolon.values,orolat,orog,cmap='Greys',norm=pl.Normalize(0,2000),
              alpha=0.7)
              
#ax1.add_feature(ocean_50m,alpha=1)
#cs1 = ax1.contourf(era5lon.values,era5lat.values,era5_mn-273.15,
#            cmap=cmap,transform=ccrs.PlateCarree(),extend='max',
#            norm=pl.Normalize(0,20),levels=pl.linspace(0,20,11))
#cb1 = pl.colorbar(cs1,orientation='horizontal',pad=0.08,shrink=0.9)
#cb1.set_label('$^\circ$C',size=14)
#
#ax1.annotate('ERA5 ta_o annual mean',(-14,69),bbox={'facecolor':'w'},fontsize=12)
#GridLines(ax1,True,True,True,False)
#
#ax2 = pl.subplot(122,projection=proj,extent=eurext)
#ax2.coastlines(linewidth=0.5,resolution='50m')
#ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
#ax2.add_feature(ocean_50m,alpha=1)
ax.scatter(bl_lons,bl_lats,color=clist)#,edgecolor=elist)
#cs2 = ax2.contourf(era5lon.values,era5lat.values,era5_trnd*10,
#            cmap=cmap,transform=ccrs.PlateCarree(),extend='max',
#            norm=pl.Normalize(0,0.5),levels=pl.linspace(0,0.5,11))
#cb2 = pl.colorbar(cs2,orientation='horizontal',pad=0.08,shrink=0.9)
#cb2.set_label('$^\circ$C decade$^{-1}$',size=14)
#
#ax2.annotate('ERA5 ta_o trend',(-14,69),bbox={'facecolor':'w'},fontsize=12)
GridLines(ax,True,True,True,True)
#
pl.subplots_adjust(left=0.07,right=0.93)
#pl.savefig(indecis+'figures/moldova_station_locations',dpi=350)

c = ['b','orange','g']
years = pl.linspace(1950,2017,68)

fig, ax = pl.subplots(3,1,figsize=(12,4))

ax1 = pl.subplot(131)
#for i in range(3):
ax1.plot(years,bl_tao.T,color='grey')#,label=stations[0])
#l2 = ax1.plot(years,bl_tao[19],lw=1.5)
#l3 = ax1.plot(years,bl_tao[22],lw=1.5)
#l4 = ax1.plot(years,bl_tao[23],label=stations[23],lw=1.5)
#l5 = ax1.plot(years,bl_tao[24],label=stations[24],lw=1.5)
#l6 = ax1.plot(years,bl_tao[40],label=stations[40],lw=1.5)
#ax1.plot(years,nb_tao[[7,24]].T,color='k',lw=0.8)
#pl.axvline(x=1993,ls='--',color='grey')
ax1.grid(axis='y',ls='--',color='grey')
pl.xlim(1950,2017); pl.ylim(0,23)
pl.ylabel('$^\circ$C',fontsize=15)
pl.title('station data')

#ax1.legend(loc=2)

ax2 = pl.subplot(132)
ax2.plot(years,era5_stats.T,color='grey')
#pl.axvline(x=1993,ls='--',color='grey')
ax2.grid(axis='y',ls='--',color='grey')
pl.xlim(1950,2017); pl.ylim(0,23)
pl.title('ERA5')

#ax2.legend(handles=[l4[0],l5[0],l6[0]],labels=list(stations[[23,24,40]]),loc=2)

ax3 = pl.subplot(133)
ax3.plot(years,ind_stats.T,color='grey')
#pl.axvline(x=1993,ls='--',color='grey')
ax3.grid(axis='y',ls='--',color='grey')
pl.xlim(1950,2017); pl.ylim(0,23)
pl.title('INDECIS')

pl.subplots_adjust(left=0.05,right=0.98)

#pl.savefig(indecis+'figures/stationdata_romania_alt.png',dpi=350)


df = pd.read_csv(indecis+'permissions.txt')
data = pl.asarray(df)

PUB = 'non-public'

bulgaria = pl.where((data[:,5]=='BULGARIA') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
bulstats = data[bulgaria,4]
bulstats = pl.asarray([i.strip() for i in bulstats])

blats = data[bulgaria,6]/10; blons = pl.float32(data[bulgaria,7])/10
#
#era5_tao, t2m, era5_stats, era5lat, era5lon = ERA5_data(ncasdir,blons,blats)
#
#ind_stats = INDECIS_data(indecis,blons,blats)
#
#fig, ax = pl.subplots(1,2,figsize=(9,4))
#
#ax1 = pl.subplot(121)
#ax1.plot(years,era5_stats[[0,1,4,5,10,11,12]].T,color='grey',lw=0.9)
#ax1.grid(axis='y',ls='--',color='grey')
#pl.xlim(1950,2017); pl.ylim(15,22)
#pl.ylabel('$^\circ$C',fontsize=15)
#pl.title('ERA5')
#
#ax2 = pl.subplot(122)
#ax2.plot(years,ind_stats[[0,1,4,5,10,11,12]].T,color='grey',lw=0.9)
#ax2.grid(axis='y',ls='--',color='grey')
#pl.xlim(1950,2017); pl.ylim(15,22)
#pl.title('INDECIS')
#
#pl.subplots_adjust(left=0.07,right=0.97,wspace=0.15)
#pl.savefig(indecis+'figures/stationdata_bulgaria_nonpub.png')

pl.figure(3)
ax = pl.subplot(projection=proj,extent=romext)
ax.coastlines(linewidth=0.5,resolution='50m')
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)
ax.pcolormesh(orolon.values,orolat,orog,cmap='Greys',norm=pl.Normalize(0,2000),
              alpha=0.7)

ax.scatter(blons[[0,1,4,5,10,11,12]],blats[[0,1,4,5,10,11,12]],color='k')

GridLines(ax,True,True,True,True)
#
pl.subplots_adjust(left=0.07,right=0.93)
#pl.savefig(indecis+'figures/bularia_station_locations',dpi=350)