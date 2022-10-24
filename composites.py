# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:34:32 2020

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
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import cmapFunctions_forPhil as CFP
from pcraig_funcs import NearestIndex

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
teleind = ['nao','ea','scand','eawr']
caps = [i.upper() for i in teleind]
name = 'ta_o'

ncfile = xr.open_dataset(ncdir+name+'_year.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data = data.values[:,70:326,34:200]

mean = pl.nanmean(data,axis=0)

tele_jfm = pl.zeros([len(teleind),len(years)])
tele_ao = tele_jfm.copy()
for i in range(len(teleind)):
    tele = pd.read_csv(indecis+teleind[i]+'_index.tim',header=5,delim_whitespace=True)
    tele = pl.asarray(tele)
    tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
#    tele_sns = pl.zeros([tele.shape[0],4])
#    tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
#    tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
#    tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
#    tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
#    tele_sns[0,0] = pl.float32('nan')
    tele_jfm[i,:] = pl.mean(tele[:,:3],axis=1)
    tele_ao[i,:] = pl.mean(tele[:,3:11],axis=1)

jfm_std = pl.nanstd(tele_jfm,axis=1)
jfm_mn = pl.nanmean(tele_jfm,axis=1)
ao_std = pl.std(tele_ao,axis=1)
ao_mn = pl.mean(tele_ao,axis=1)

#teles_std = pl.nanstd(tele_sns,axis=0) # seasonal std
#teles_mn = pl.nanmean(tele_sns,axis=0) # seasonal means

#telem_std = pl.nanstd(tele,axis=0)
#telem_mn = pl.nanmean(tele,axis=0)

#ssn_ind = 1
#mon_ind = 3

gtr = []; lsr = []
above = pl.zeros([len(teleind),lon2.size,lat2.size]); below = above.copy()
for i in range(len(teleind)):
    gtr.append(pl.where(tele_ao[i]>ao_mn[i]+ao_std[i])[0])
    lsr.append(pl.where(tele_ao[i]<ao_mn[i]-ao_std[i])[0])

    above[i] = pl.nanmean(data[gtr[i],:,:],axis=0)
    below[i] = pl.nanmean(data[lsr[i],:,:],axis=0)

#cmap, clevs = CFP.get_eofColormap(above-mean)

proj = ccrs.PlateCarree()
ext = [-15,42,35,70]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
cmap = 'seismic'
levs = pl.linspace(-1,1,21)

fig, ax = pl.subplots(4,2,figsize=(8,10)) # (11,6) for 2x2, (11,9) for 3x2
#pl.figure(figsize=(10,7))
#gs = gridspec.GridSpec(2, 4)
#ig = [gs[0,1:3],gs[1,:2],gs[1,2:]]

###############################################################################
ax1 = pl.subplot(421,projection=proj,extent=ext)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)
#levs = pl.linspace(-20,20,41)#[0,31,59,90,120,151]
cs = ax1.contourf(lon2,lat2,(above[0]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

#f = pl.gcf()
#colax = f.add_axes([0.75,0.58,0.02,0.38])
#cb = pl.colorbar(cs,orientation='vertical',cax=colax)
#cb.set_label('days',fontsize=12)

GridLines(ax1,False,False,True,False)
pl.title('(a) '+name+'$>1$ std GS '+caps[0])
#ax1.annotate('(a) '+name+' JFM mean',
#             (-20,69),bbox={'facecolor':'w'},fontsize=12)

###############################################################################
ax2 = pl.subplot(422,projection=proj,extent=ext)
ax2.coastlines(linewidth=0.5,resolution='50m')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
#levs = pl.linspace(-20,20,41)
cs = ax2.contourf(lon2,lat2,(below[0]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax2,False,False,False,True)
pl.title('(b) '+name+'$<1$ std GS '+caps[0])
#ax2.annotate('(b) '+name+' $<1$ std JFM '+caps[0],
#             (-20,69),bbox={'facecolor':'w'},fontsize=12)

###############################################################################
ax3 = pl.subplot(423,projection=proj,extent=ext)
ax3.coastlines(linewidth=0.5,resolution='50m')
ax3.add_feature(borders_50m,linewidth=0.5,zorder=5)
#levs = pl.linspace(0,200,21)
cs = ax3.contourf(lon2,lat2,(above[1]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax3,False,False,True,False)
pl.title('(c) '+name+'$>1$ std GS '+caps[1])
#ax3.annotate('(c) '+name+' $<1$ std JFM '+teleind[:].upper(),
#             (-20,69),bbox={'facecolor':'w'},fontsize=12)

###############################################################################
ax4 = pl.subplot(424,projection=proj,extent=ext)
ax4.coastlines(linewidth=0.5,resolution='50m')
ax4.add_feature(borders_50m,linewidth=0.5,zorder=5)
#levs = pl.linspace(0,200,21)
cs = ax4.contourf(lon2,lat2,(below[1]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax4,False,False,False,True)
pl.title('(d) '+name+'$<1$ std GS '+caps[1])

###############################################################################
ax5 = pl.subplot(425,projection=proj,extent=ext)
ax5.coastlines(linewidth=0.5,resolution='50m')
ax5.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax5.contourf(lon2,lat2,(above[2]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax5,False,False,True,False)
pl.title('(e) '+name+'$>1$ std GS '+caps[2][:3])

###############################################################################
ax6 = pl.subplot(426,projection=proj,extent=ext)
ax6.coastlines(linewidth=0.5,resolution='50m')
ax6.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax6.contourf(lon2,lat2,(below[2]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax6,False,False,False,True)
pl.title('(f) '+name+'$<1$ std GS '+caps[2][:3])

###############################################################################
ax7 = pl.subplot(427,projection=proj,extent=ext)
ax7.coastlines(linewidth=0.5,resolution='50m')
ax7.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax7.contourf(lon2,lat2,(above[3]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax7,False,True,True,False)
pl.title('(g) '+name+'$>1$ std GS '+caps[3])

###############################################################################
ax8 = pl.subplot(428,projection=proj,extent=ext)
ax8.coastlines(linewidth=0.5,resolution='50m')
ax8.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax8.contourf(lon2,lat2,(below[3]-mean).T,transform=ccrs.PlateCarree(),levels=levs,
                  cmap=cmap,extend='both')

GridLines(ax8,False,True,False,True)
pl.title('(h) '+name+'$<1$ std GS '+caps[3])

f = pl.gcf()
colax = f.add_axes([0.15,0.06,0.7,0.02])
cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
#cb.set_ticklabels(['1 Jan','1 Feb','1 Mar','1 Apr','1 May','1 Jun'])
cb.ax.tick_params(labelsize=12)
cb.set_label('$^\circ$C',fontsize=12,labelpad=-0.5)

pl.tight_layout()
pl.subplots_adjust(wspace=-0.4,bottom=0.11,left=0.0,right=1.0,hspace=0.22)

pl.savefig(indecis+'figures/'+name+'_'+teleind[0]+'_'+teleind[1]+'_'+teleind[2]+'_'+teleind[3]+'_AO_composites_anoms.png',
           dpi=400)