# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:16:01 2020

@author: qx911590
"""

from __future__ import division
import pylab as pl
import xarray as xr
import pandas as pd
from scipy import stats
#from scipy import signal
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
import pcraig_funcs as pc

def PickSeason(sns_name,tele_sns,data):
    """
    Args:
        sns_name (string): 3 letters defining a season (DJF, MAM, JJA, SON)
        tele_sns (array): N years X 4 array of teleconnection index averaged
                            over each season
        data (array): N years X lon X lat array of variable
    
    Returns:
        data_out (array): N or N-1 years X lon2 X lat2 array of variable
        TSI (array): N or N-1 years X 1 array of teleconnection index averaged 
                    over a particular season
        x (array) = 1D array of N or N-1 years
    """
    if sns_name == 'DJF':
        data_out = data.values[1:,70:326,34:200]
        TSI = tele_sns[1:,0]
        #x = pl.linspace(1,data_out.shape[0],data_out.shape[0])
    elif sns_name == 'MAM':
        data_out = data.values[:,70:326,34:200]
        TSI = tele_sns[:,1]
        #x = pl.linspace(1,68,68)
    elif sns_name == 'JJA':
        data_out = data.values[:,70:326,34:200]
        TSI = tele_sns[:,2]
        #x = pl.linspace(1,68,68)
    elif sns_name == 'SON':
        data_out = data.values[:,70:326,34:200]
        TSI = tele_sns[:,3]
        #x = pl.linspace(1,67,67)
    #elif sns_name = 'JFM':
     #   data_out = data.values[:,70:326,34:200]
        
    
    x = pl.linspace(1,data_out.shape[0],data_out.shape[0])

    return data_out, TSI, x

def TeleconSignal(vardata,tele_dt,x):
    """
    Args:
        vardata (array): N or N-1 years X lon2 X lat2 array of variable data
        tele_dt (array): detrended teleconnection index for a particular season
    
    Returns:
        signal (array): lon2 x lat2 array of teleconnection signal
        sig_p (array): lon2 x lat2 array of p values less than 0.05 for signal
        var_trnd (array): lon2  x lat2 array of variable trend
        vt_p (array): lon2 x lat2 array of p values less than 0.05 for var_trnd
    """
    signal = pl.zeros([vardata.shape[1],vardata.shape[2]])
    signal[:] = pl.float32('nan'); sig_p = signal.copy()
    var_trnd = pl.zeros([vardata.shape[1],vardata.shape[2]])
    var_trnd[:] = pl.float32('nan'); vt_p = var_trnd.copy()
    
    for i in range(vardata.shape[1]):
        for j in range(vardata.shape[2]):
            y = vardata[:,i,j]
            not_nan_ind = ~pl.isnan(y)
            if pl.where(not_nan_ind==True)[0].size == 0:
                pass
            else:
                m, b, r, p, se = stats.linregress(x[not_nan_ind],y[not_nan_ind])
                var_trnd[i,j] = m
                if p < 0.1:
                    vt_p[i,j] = p
                detrend_y = y - (m*x + b) # detrended variable
                out = stats.linregress(tele_dt,detrend_y)
                signal[i,j] = out[0]#stats.linregress(tele_dt,detrend_y).slope # NAO signal
                sp = out[3]#stats.linregress(tele_dt,detrend_y).pvalue
                if sp < 0.05:
                    sig_p[i,j] = sp
    
    return signal, sig_p, var_trnd, vt_p

def ComponentAndResidual(vardata,signal,tele_dt,x):
    """
    """
    tc_comp = pl.zeros_like(vardata,)
    tc_comp[:] = pl.float32('nan')
    tc_comp_trnd = pl.zeros_like(signal); tc_comp_trnd[:] = pl.float32('nan')
    tct_p = tc_comp_trnd.copy()
    nosig = tc_comp_trnd.copy(); nosig_p = nosig.copy()
    
    for yr in range(vardata.shape[0]):
        tc_comp[yr] = tele_dt[yr]*signal[:,:]
    
    for i in range(vardata.shape[1]):
        for j in range(vardata.shape[2]):
            out1 = stats.linregress(x,tc_comp[:,i,j])
            tc_comp_trnd[i,j] = out1[0]#stats.linregress(x,tc_comp[:,i,j]).slope
            sp = out1[3]#stats.linregress(x,tc_comp[:,i,j]).pvalue
            if sp < 0.05:
                tct_p[i,j] = sp
            
            out2 = stats.linregress(x,vardata[:,i,j]-tc_comp[:,i,j])
            nosig[i,j] = out2[0]#stats.linregress(x,data[:,i,j]-tc_comp[:,i,j]).slope
            sp = out2[3]#stats.linregress(x,data[:,i,j]-tc_comp[:,i,j]).pvalue
            if sp < 0.05:
                nosig_p[i,j] = sp
        
    return tc_comp, tc_comp_trnd, tct_p, nosig, nosig_p

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

def AreasCalc():
    """
    """
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
    
    return areas_clip

def RegionCalc(vertices,lon2,lat2,data):
    """
    """
    rPath = mplPath.Path(vertices)
    TF = pl.zeros([lon2.size,lat2.size])
    rmask = pl.zeros([lon2.size,lat2.size])
    rmask[:] = pl.float32('nan')
    
    for i in range(lon2.size):
            for j in range(lat2.size):
                X = rPath.contains_point((lon2[i],lat2[j]))
                TF[i,j] = X
    
    Y = pl.where(TF)
    rmask[Y[0],Y[1]] = 1
    
    areas = AreasCalc()
    
    rdata = data[:,:,:]*rmask[:,:]#None,
    rareas = areas*rmask
    
    Q = pl.ones_like(data)
    f = pl.isnan(data)
    d = pl.where(f==True)
    Q[d[0],d[1],d[2]] = pl.float32('nan')
    
    #P = pl.average(rdata[0],weights=pl.nan_to_num(rareas))
    W = pl.zeros([data.shape[0]])
    W[0] = pl.float32('nan')
     
    for i in range(data.shape[0]): # loop over years
        W[i] = pl.nansum(rdata[i]*rareas)/(pl.nansum(rareas*Q[i]))
    
    return W

pl.close('all')

homedir = '/home/users/qx911590/'
indecis = homedir + 'INDECIS/'
ncdir = indecis + 'ncfiles/'

teleindex = ['nao','ea','scand','eawr']
caps = [i.upper() for i in teleindex]
name = 'ogs10'



ncfile = xr.open_dataset(ncdir+name+'_year.nc')
data = ncfile.variables[name][:]
data = xr.DataArray(data)
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
time = xr.DataArray(ncfile.variables['time'][:])
ncfile.close()

time = time.values.astype(str)

years = pl.asarray([i[:4] for i in time])


tele_jfm = pl.zeros([len(teleindex),len(years)])
tele_jja = tele_jfm.copy()
tele_ao = tele_jfm.copy()
for i in range(len(teleindex)):
    tele = pd.read_csv(indecis+teleindex[i]+'_index.tim',header=5,delim_whitespace=True)
    tele = pl.asarray(tele)
    tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
    tele_sns = pl.zeros([tele.shape[0],4])
    tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
    tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
    tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
    tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
    tele_sns[0,0] = pl.float32('nan')
    tele_jfm[i,:] = pl.mean(tele[:,:3],axis=1)
#    tele_jja[i,:] = tele_sns[:,2]
    tele_ao[i,:] = pl.mean(tele[:,3:11],axis=1)

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data  = data.values[:,70:326,34:200]
x = pl.linspace(1,68,68)

tele_dt = pl.zeros_like(tele_jfm)

for i in range(len(teleindex)):
    tele_dt[i] = pl.detrend_linear(tele_jfm[i]) # detrended index
#tele_trnd = stats.linregress(x,tele_jfm[i]).slope

signal = pl.zeros([len(teleindex),lon2.size,lat2.size])
sig_p = signal.copy()
var_trnd = signal.copy(); vt_p = signal.copy()

tc_comp = pl.zeros([len(teleindex),len(years),lon2.size,lat2.size])
tc_comp_trnd = signal.copy(); tct_p = signal.copy()
nosig = signal.copy(); nosig_p = signal.copy()

for i in range(len(teleindex)):
    signal[i], sig_p[i], var_trnd[i], vt_p[i] = TeleconSignal(data,tele_dt[i],x)
    tc_comp[i], tc_comp_trnd[i], tct_p[i], nosig[i], nosig_p[i] =\
                            ComponentAndResidual(data,signal[i],tele_jfm[i],x)
                            # replace telecon time series with detrended series in C&R
                            # var_trnd added to C&R
                            # tele_trnd added to C&R

"""fig, ax = pl.subplots(2,3,figsize=(13,6)) # 2x3 = (13,6). 4x3 = (12,10)
proj = ccrs.PlateCarree()
ext = [-15,42,35,70]
borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
levs_sig = pl.linspace(-15,15,11)
levs_tct = pl.linspace(-0.4,0.4,9)
levs_ns = pl.linspace(-4,4,9)
cmap = 'BrBG_r'

##############################################################################
ax1 = pl.subplot(231,projection=proj,extent=ext)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax1.contourf(lon2,lat2,signal[0].T,transform=ccrs.PlateCarree(),
                          cmap=cmap,levels=levs_sig,alpha=0.6,extend='both')
ax1.contourf(lon2,lat2,sig_p[0].T,hatches=['...'],colors='none')

GridLines(ax1,False,False,True,False)
#ax1.annotate('(a) JFM NAO signal',(-13,68),bbox={'facecolor':'w'},fontsize=12)
pl.title('(a) '+name+' JFM '+caps[0]+' signal',fontsize=12)

###############################################################################
ax2 = pl.subplot(232,projection=proj,extent=ext)
ax2.coastlines(linewidth=0.5,resolution='50m')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax2.contourf(lon2,lat2,signal[1].T,transform=ccrs.PlateCarree(),
                          cmap=cmap,levels=levs_sig,alpha=0.8,extend='both')
ax2.contourf(lon2,lat2,sig_p[1].T,hatches=['...'],colors='none')
#ax4.annotate('(d) JFM EA signal',(-13,68),bbox={'facecolor':'w'},fontsize=12)
pl.title('(b) '+name+' JFM '+caps[1]+' signal',fontsize=12)

#f = pl.gcf()
#colax = f.add_axes([0.1,0.075,0.39,0.03])
#cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
#cb.set_label('mm '+'index $^{-1}$',fontsize=12,labelpad=-1)
#cb.set_ticks(levs_sig)
#cb.set_ticklabels(levs_sig.astype(int))
#cb.ax.tick_params(labelsize=10)
GridLines(ax2,False,False,False,False)

###############################################################################
ax3 = pl.subplot(233,projection=proj,extent=ext)
ax3.coastlines(linewidth=0.5,resolution='50m')
ax3.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax3.contourf(lon2,lat2,signal[2].T,transform=ccrs.PlateCarree(),
                          cmap=cmap,levels=levs_sig,alpha=0.8,extend='both')
ax3.contourf(lon2,lat2,sig_p[2].T,hatches=['...'],colors='none')
#ax4.annotate('(d) JFM EA signal',(-13,68),bbox={'facecolor':'w'},fontsize=12)
pl.title('(c) '+name+' JFM '+caps[2][:3]+' signal',fontsize=12)
GridLines(ax3,False,False,False,True)

f = pl.gcf()
colax = f.add_axes([0.93,0.54,0.02,0.37])
cb = pl.colorbar(cs,orientation='vertical',cax=colax)
cb.set_label('days index $^{-1}$',fontsize=12,labelpad=0)
cb.set_ticks(levs_sig)
cb.set_ticklabels(levs_sig.astype(int))
cb.ax.tick_params(labelsize=10)

###############################################################################
ax4 = pl.subplot(234,projection=proj,extent=ext)
ax4.coastlines(linewidth=0.5,resolution='50m')
ax4.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax4.contourf(lon2,lat2,nosig[0].T,transform=ccrs.PlateCarree(),
                          cmap=cmap,levels=levs_ns,alpha=0.8,extend='both')
ax4.contourf(lon2,lat2,sig_p[0].T,hatches=['...'],colors='none')
#ax4.annotate('(d) JFM EA signal',(-13,68),bbox={'facecolor':'w'},fontsize=12)
pl.title('(d) '+name+' JFM '+caps[0]+' residual trend',fontsize=12)
GridLines(ax4,False,True,True,False)


###############################################################################
ax5 = pl.subplot(235,projection=proj,extent=ext)
ax5.coastlines(linewidth=0.5,resolution='50m')
ax5.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax5.contourf(lon2,lat2,(nosig[1].T)*10,transform=ccrs.PlateCarree(),
                          cmap=cmap,levels=levs_ns,alpha=0.8,extend='both')
ax5.contourf(lon2,lat2,nosig_p[1].T,hatches=['...'],colors='none')

GridLines(ax5,False,True,False,False)
#ax3.annotate('(c) JFM residual trend',(-13,68),\
#            bbox={'facecolor':'w','alpha':1},fontsize=12)
pl.title('(e) '+name+' JFM '+caps[1]+' residual trend',fontsize=12)

###############################################################################

ax6 = pl.subplot(236,projection=proj,extent=ext)
ax6.coastlines(linewidth=0.5,resolution='50m')
ax6.add_feature(borders_50m,linewidth=0.5,zorder=5)
cs = ax6.contourf(lon2,lat2,(nosig[2].T)*10,transform=ccrs.PlateCarree(),
                          cmap=cmap,levels=levs_ns,alpha=0.8,extend='both')
ax6.contourf(lon2,lat2,nosig_p[2].T,hatches=['...'],colors='none')
pl.title('(f) '+name+' JFM '+caps[2][:3]+' residual trend',fontsize=12)


#f = pl.gcf()
#colax = f.add_axes([0.51,0.075,0.39,0.03])
#cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
#cb.set_label('mm '+'year $^{-1}$',fontsize=12,labelpad=-1)
#cb.set_ticks(levs_ns)
#cb.set_ticklabels(levs_ns.astype(int))
GridLines(ax6,False,True,False,False)
#ax6.annotate('(f) JFM residual trend',(-13,68),bbox={'facecolor':'w'},fontsize=12)

###############################################################################
#ax7 = pl.subplot(247,projection=proj,extent=ext)
#ax7.coastlines(linewidth=0.5,resolution='50m')
#ax7.add_feature(borders_50m,linewidth=0.5,zorder=5)
#cs = ax7.contourf(lon2,lat2,(nosig[2].T)*10,transform=ccrs.PlateCarree(),
#                          cmap=cmap,levels=levs_ns,alpha=0.8,extend='both')
#ax7.contourf(lon2,lat2,nosig_p[2].T,hatches=['...'],colors='none')
#pl.title('(g) '+name+' GS '+caps[2][:3]+' residual trend',fontsize=12)
#
#GridLines(ax7,False,True,False,False)

###############################################################################
#ax8 = pl.subplot(248,projection=proj,extent=ext)
#ax8.coastlines(linewidth=0.5,resolution='50m')
#ax8.add_feature(borders_50m,linewidth=0.5,zorder=5)
#cs = ax8.contourf(lon2,lat2,(nosig[3].T)*10,transform=ccrs.PlateCarree(),
#                          cmap='seismic',levels=levs_ns,alpha=0.8,extend='both')
#ax8.contourf(lon2,lat2,nosig_p[3].T,hatches=['...'],colors='none')
#pl.title('(h) '+name+' GS '+caps[3]+' residual trend',fontsize=12)

f = pl.gcf()
colax = f.add_axes([0.93,0.075,0.02,0.37])
cb = pl.colorbar(cs,orientation='vertical',cax=colax)
cb.set_label('days decade $^{-1}$',fontsize=12,labelpad=1)
cb.set_ticks(levs_ns)
cb.set_ticklabels(levs_ns.astype(int))
cb.ax.tick_params(labelsize=10)
GridLines(ax6,False,True,False,True)"""

###############################################################################
#ax9 = pl.subplot(439,projection=proj,extent=ext)
#ax9.coastlines(linewidth=0.5,resolution='50m')
#ax9.add_feature(borders_50m,linewidth=0.5,zorder=5)
#cs = ax9.contourf(lon2,lat2,nosig[2].T,transform=ccrs.PlateCarree(),
#                          cmap='seismic',levels=levs_ns,alpha=0.8,extend='both')
#ax9.contourf(lon2,lat2,(nosig_p[2].T)*10,hatches=['...'],colors='none')
#pl.title('(i) '+name+' GS residual trend',fontsize=12)

#f = pl.gcf()
#colax = f.add_axes([0.647,0.055,0.23,0.03])
#cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
#cb.set_label('$^\circ$C '+'year $^{-1}$',fontsize=12,labelpad=-1)
#cb.set_ticks(levs_ns)
#cb.set_ticklabels(levs_ns.astype(int))
#GridLines(ax9,False,False,False,True)

###############################################################################
#ax10 = pl.subplot(4,3,10,projection=proj,extent=ext)
#ax10.coastlines(linewidth=0.5,resolution='50m')
#ax10.add_feature(borders_50m,linewidth=0.5,zorder=5)
#cs = ax10.contourf(lon2,lat2,signal[3].T,transform=ccrs.PlateCarree(),
#                          cmap='seismic',levels=levs_sig,alpha=0.8,extend='both')
#ax10.contourf(lon2,lat2,sig_p[3].T,hatches=['...'],colors='none')
##ax4.annotate('(d) JFM EA signal',(-13,68),bbox={'facecolor':'w'},fontsize=12)
#pl.title('(j) '+name+' GS '+caps[3]+' signal',fontsize=12)
#
#f = pl.gcf()
#colax = f.add_axes([0.10,0.055,0.23,0.025])
#cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
#cb.set_label('$^\circ$C '+'index $^{-1}$',fontsize=12,labelpad=-1)
#cb.set_ticks(levs_sig[::2])
#cb.set_ticklabels(levs_sig[::2])
#cb.ax.tick_params(labelsize=10)
#GridLines(ax10,False,True,True,False)

###############################################################################
#ax12 = pl.subplot(4,3,12,projection=proj,extent=ext)
#ax12.coastlines(linewidth=0.5,resolution='50m')
#ax12.add_feature(borders_50m,linewidth=0.5,zorder=5)
#cs = ax12.contourf(lon2,lat2,(nosig[3].T)*10,transform=ccrs.PlateCarree(),
#                          cmap='seismic',levels=levs_ns,alpha=0.8,extend='both')
#ax12.contourf(lon2,lat2,nosig_p[3].T,hatches=['...'],colors='none')
#pl.title('(l) '+name+' GS residual trend',fontsize=12)
#
#f = pl.gcf()
#colax = f.add_axes([0.695,0.055,0.23,0.025])
#cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
#cb.set_label('$^\circ$C '+'decade $^{-1}$',fontsize=12,labelpad=-1)
#cb.set_ticks(levs_ns)
#cb.set_ticklabels(levs_ns)
#GridLines(ax12,False,True,False,True)

###############################################################################
#pl.tight_layout()
#pl.subplots_adjust(top=0.96,bottom=0.15,wspace=-0.3,left=0.00,right=1)
#pl.subplots_adjust(top=0.955,bottom=0.045,left=0.04,right=0.89,wspace=0.1,hspace=0.15)

#pl.savefig(indecis+'figures/'+name+'_'+teleindex[0]+'_'+teleindex[1]+'_'+teleindex[2]+'_jfm_panels_nocomptr.png',
#           dpi=400)



#english_channel = [(-1.7,46),(9.2,51.9),(7.9,53.8),(-5.9,53.4)]
#baltic_sea = [(8.2,53.4),(8.2,57.5),(19.6,66.1),(28.2,66.1),(28.2,57.8),
#              (19.5,53.4)]
#denmark = [(6.3,53.9),(6.3,57.5),(11.9,57.5),(11.9,53.9)]
#ireland = [(-10.9,51.4,),(-10.9,55.4,),(-5.3,55.4),(-5.3,52.2),(-8.8,51.4)]
#scand_all = [(5.31,58.49),(4.34,62.02),(19.9,71.19),(30.09,71.41),(31.76,62.91),
#             (26.93,60.1),(19.02,59.49),(14.54,54.97),(8.21,54.92)]
#scand_ne = [(9.2,62.0),(21.8,71.5),(27.5,71.5),(27.5,65.2),(22.8,65.2),(16.8,62.0)]
#france = [(-1.8,43.3),(-2.5,48.0),(6.3,50.5),(8.0,42.5),(3.5,42.3)]
#balkans = [(13.58,46.53),(22.72,48.04),(26.56,48.24),(29.55,45.11),(26.17,40.97),
#           (22.59,40.39),(24.47,38.07),(22.56,36.82),(19.38,40.65),(19.00,41.97),
#            (13.86,44.88)]
#balkans_nw = [(20.4,44.2),(14.1,45.0),(17.0,47.7),(23.2,48.4)]
#balkans_se = [(19.5,41.6),(22.9,44.9),(26.7,45.4),(27.6,47.0),(30.6,46.62),
#              (26.3,40.6),(22.6,40.2),(22.3,37.9)]
#italy = [(8.3,43.9),(8.2,45.9),(12.4,45.6),(14.3,43.0),(13.6,41.2),(9.4,44.2)]
#easteur_n = [(29.6,51.1),(24.9,53.8),(24.0,56.4),(20.4,56.4),(23.2,59.6),(31.4,55.4)]
#easteur_s = [(28.4,43.5),(24.9,43.5),(23.3,45.4),(24.0,49.5),(33.4,48.0),
#             (34.0,45.9),(30.7,46.0)]
#iberia = [(-8.95,43.93),(-1.95,43.74),(4.28,42.04),(-1.2,35.99),(-5.55,35.23),
#          (-10.09,36.93)]
#gbi = [(-6.31,59.43),(-1.01,59.43),(2.58,52.82),(1.45,50.74),(-5.55,49.41),
#       (-11.22,59.11),(-11.22,59.46)]
#ngerpol = [(7.20,53.49),(20.26,54.43),(18.90,49.61),(14.88,51.08),(11.98,50.30),
#           (6.15,50.30)]
#east_rom = [(22.49,44.67),(25.04,47.75),(29.25,47.48),(30.64,46.29),(28.55,43.57)]
#hungary = [(16.18,46.91),(17.09,48.00),(18.73,47.87),(20.54,48.55),(22.14,48.44),
#           (22.92,47.99),(22.03,47.61),(21.10,46.23),(19.60,46.17),(18.07,45.78)]
#
#V = RegionCalc(balkans,lon2,lat2,data)
#V_an = V - pl.nanmean(V)
#S = pl.zeros([len(teleindex),len(years)])
#c = ['orange','magenta']

#pl.figure(2)
#pl.plot(years,V_an,label=name+' anomalies',color='k',lw=2)
#print pl.std(V_an)
#
#for i in range(len(teleindex)):
#    S[i] = RegionCalc(balkans,lon2,lat2,tc_comp[i])
#    print pl.std(S[i])
#    pl.plot(years[:],S[i],label='JJA '+caps[i]+' component',color=c[i])
#
#pl.grid(axis='y',ls='--',color='grey')
#pl.xlim(years[0].astype(int),years[-1].astype(int))
#pl.ylim(-160,180)
#pl.ylabel('days',fontsize=13,labelpad=-7)
#pl.legend()
#pl.title('Hungary',fontsize=14)
#pl.savefig(indecis+'figures/'+name+'_'+teleindex[0]+'_'+teleindex[1]+\
#            '_component_ts_hun.png',dpi=400)