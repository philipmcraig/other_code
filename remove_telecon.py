# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:21:35 2020

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

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

teleindex = 'nao'
caps = 'NAO'
name = 'gtg'

tele = pd.read_csv(indecis+teleindex+'_index.tim',header=5,delim_whitespace=True)
tele = pl.asarray(tele)

ncfile = xr.open_dataset(ncdir+name+'_season.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])
months = pl.asarray([i[5:7] for i in time])
#
ind = pl.where((years=='1979') & (months=='07'))
ind = ind[0][0]

tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
tele_sns = pl.zeros([tele.shape[0],4])
tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
tele_sns[0,0] = pl.float32('nan')


#tele_sns = pl.reshape(tele_sns,newshape=(tele_sns.shape[0]*tele_sns.shape[1]))
lon2 = lon.values[70:326]
lat2 = lat.values[34:200]

#data_notrend = pl.detrend(data.values,axis=0)
#data_djf = data.values[4::4,70:326,34:200]
data  = data.values[3::4,70:326,34:200]
tele_dt = pl.detrend_linear(tele_sns[:,3]) # detrended index

x = pl.linspace(1,68,68)
tele_trnd = stats.linregress(x,tele_sns[:,3]).slope

#detrend_y = pl.zeros_like(data_jja); detrend_y[:] = pl.float32('nan')
signal = pl.zeros([lon2.size,lat2.size]); signal[:] = pl.float32('nan')
sig_p = signal.copy()
var_trnd = pl.zeros([lon2.size,lat2.size]); var_trnd[:] = pl.float32('nan')
vt_p = var_trnd.copy()
#b = m.copy(); r = m.copy(); p = m.copy(); se = m.copy()
#not_nan_ind = ~pl.isnan(data_jja)

for i in range(lon2.size):
    for j in range(lat2.size):
        y = data[:,i,j]
        not_nan_ind = ~pl.isnan(y)
        if pl.where(not_nan_ind==True)[0].size == 0:
            pass
        else:
            m, b, r, p, se = stats.linregress(x[not_nan_ind],y[not_nan_ind])
            var_trnd[i,j] = m
            if p < 0.05:
                vt_p[i,j] = p
            detrend_y = y - (m*x + b) # detrended variable
            out = stats.linregress(tele_dt,detrend_y)
            signal[i,j] = out[0]#stats.linregress(tele_dt,detrend_y).slope # NAO signal
            sp = out[3]#stats.linregress(tele_dt,detrend_y).pvalue
            if sp < 0.05:
                sig_p[i,j] = sp

#nosig = var_trnd - tele_trnd*signal[:,:]

tc_comp = pl.zeros([x.size,lon2.size,lat2.size]); tc_comp[:] = pl.float32('nan')
for yr in range(x.size):
    tc_comp[yr] = tele_sns[yr,3]*signal[:,:]

tc_comp_trnd = pl.zeros_like(signal); tc_comp_trnd[:] = pl.float32('nan')
tct_p = tc_comp_trnd.copy()
nosig = tc_comp_trnd.copy(); nosig_p = nosig.copy()
for i in range(lon2.size):
    for j in range(lat2.size):
        out1 = stats.linregress(x,tc_comp[:,i,j])
        tc_comp_trnd[i,j] = out1[0]#stats.linregress(x,tc_comp[:,i,j]).slope
        sp = out1[3]#stats.linregress(x,tc_comp[:,i,j]).pvalue
        if sp < 0.05:
            tct_p[i,j] = sp
        
        out2 = stats.linregress(x,data[:,i,j]-tc_comp[:,i,j])
        nosig[i,j] = out2[0]#stats.linregress(x,data_djf[:,i,j]-tc_comp[:,i,j]).slope
        sp = out2[3]#stats.linregress(x,data_djf[:,i,j]-tc_comp[:,i,j]).pvalue
        if sp < 0.05:
            nosig_p[i,j] = sp

fig, ax = pl.subplots(2,2,figsize=(12,12))
proj = ccrs.PlateCarree()
ext = [-22,42,35,70]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
###############################################################################
ax1 = pl.subplot(221,projection=proj,extent=ext)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(-0.04,0.04,9)#[-4,-3,-2,-1,0,1,2,3,4]
cs = ax1.contourf(lon2,lat2,var_trnd.T,cmap='seismic',#norm=pl.Normalize(-3,3),
            transform=ccrs.PlateCarree(),alpha=0.8,extend='both',
            levels=levs)
cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)

ax1.contourf(lon2,lat2,vt_p.T,hatches=['.'],colors='none')

cb.set_label('mm yr$^{-1}$',fontsize=12)
cb.set_ticks(levs)
cb.set_ticklabels(levs)
GridLines(ax1,True,False,True,False)
ax1.annotate('(a) SON '+name+' trend',(-19,69),bbox={'facecolor':'w'},fontsize=12)
################################################################################
ax2 = pl.subplot(222,projection=proj,extent=ext)
##ax = pl.axes(projection=ccrs.PlateCarree(),extent=[-20,42,35,70])
##ax.set_extent([])
ax2.coastlines(linewidth=0.5,resolution='50m')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(-1,1,11)#[-200,-150,-100,-75,-50,-25,-10,0,10,25,50,75,100,150,200]
cs = ax2.contourf(lon2,lat2,signal.T,cmap='seismic',#norm=pl.Normalize(-200,200),
            transform=ccrs.PlateCarree(),alpha=0.8,extend='both',
            levels=levs)

ax2.contourf(lon2,lat2,sig_p.T,hatches=['.'],colors='none')

cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
cb.set_label('mm '+caps+'I$^{-1}$',fontsize=12)
cb.set_ticks(levs)
cb.set_ticklabels(levs)
#cb.ax.tick_params(labelsize=8)
GridLines(ax2,True,False,False,True)
ax2.annotate('(b) '+caps+' signal',(-19,69),bbox={'facecolor':'w'},fontsize=12)
###############################################################################
ax3 = pl.subplot(223,projection=proj,extent=ext)
ax3.coastlines(linewidth=0.5,resolution='50m')
ax3.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(-0.004,0.004,9)#[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
cs = ax3.contourf(lon2,lat2,tc_comp_trnd.T,cmap='seismic',#norm=pl.Normalize(-0.5,0.5),
                  transform=ccrs.PlateCarree(),alpha=0.8,extend='both',
                    levels=levs)

ax3.contourf(lon2,lat2,tct_p.T,hatches=['.'],colors='none')

cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
cb.set_label('mm yr$^{-1}$',fontsize=12)
cb.set_ticks(levs)
cb.set_ticklabels(levs)
cb.ax.tick_params(labelsize=8)
GridLines(ax3,False,False,True,False)
ax3.annotate('(c) '+caps+' component trend',(-19,69),bbox={'facecolor':'w'},fontsize=12)
###############################################################################
ax4 = pl.subplot(224,projection=proj,extent=ext)
ax4.coastlines(linewidth=0.5,resolution='50m')
ax4.add_feature(borders_50m,linewidth=0.5,zorder=5)
levs = pl.linspace(-0.05,0.05,11)
cs = ax4.contourf(lon2,lat2,nosig.T,cmap='seismic',#norm=pl.Normalize(-4,4),
                  transform=ccrs.PlateCarree(),alpha=0.8,extend='both',
                    levels=levs)

ax4.contourf(lon2,lat2,nosig_p.T,hatches=['.'],colors='none')

cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
cb.set_label('mm yr$^{-1}$',fontsize=12)
cb.set_ticks(levs)
cb.set_ticklabels(levs)
GridLines(ax4,False,False,False,True)
ax4.annotate('(d) residual trend',(-19,69),bbox={'facecolor':'w'},fontsize=12)

pl.tight_layout()
pl.subplots_adjust(left=0.04,right=0.96,bottom=0.07,top=0.96,hspace=0.08)


pl.savefig(indecis+'figures/'+name+'_son_remove_'+teleindex+'_panels.png',dpi=350)