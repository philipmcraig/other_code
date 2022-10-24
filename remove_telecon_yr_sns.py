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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
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

def PickSeason(sns_name,tele_sns,data):
    """
    Args:
        sns_name (string): 3 letters defining a season (DJF, MAM, JJA, SON)
        tele_sns (array): N years X 4 array of teleconnection index averaged
                            over each season
        data (array): N*4 years X lon X lat array of variable
    
    Returns:
        data_out (array): N or N-1 years X lon2 X lat2 array of variable
        TSI (array): N or N-1 years X 1 array of teleconnection index averaged 
                    over a particular season
        x (array) = 1D array of N or N-1 years
    """
    if sns_name == 'DJF':
        data_out = data.values[4::4,70:326,34:200]
        TSI = tele_sns[1:,0]
        #x = pl.linspace(1,data_out.shape[0],data_out.shape[0])
    elif sns_name == 'MAM':
        data_out = data.values[1::4,70:326,34:200]
        TSI = tele_sns[:,1]
        #x = pl.linspace(1,68,68)
    elif sns_name == 'JJA':
        data_out = data.values[2::4,70:326,34:200]
        TSI = tele_sns[:,2]
        #x = pl.linspace(1,68,68)
    elif sns_name == 'SON':
        data_out = data.values[3::4,70:326,34:200]
        TSI = tele_sns[:,3]
        #x = pl.linspace(1,67,67)
    
    x = pl.linspace(1,data_out.shape[0],data_out.shape[0])
    
    return data_out, TSI, x

def PickMonth(month_no,season,tele,data):
    """
    """
    if month_no in range(0,10) and season != 'DJF':
        #data_out = data.values[:,70:326,34:200]
        TSI = tele[:,month_no-1]
    elif month_no in range(0,10) and season == 'DJF':
        TSI = tele[:-1,month_no-1]
    elif month_no in range(10,12):
        #data_out = data.values[1:,70:326,34:200]
        TSI = tele[:-1,month_no-1]
     
    if season == 'DJF':
        data_out = data.values[4::4,70:326,34:200]
    elif season == 'MAM':
        data_out = data.values[1::4,70:326,34:200]
    elif season == 'JJA':
        data_out = data.values[2::4,70:326,34:200]
    elif season == 'SON':
        data_out = data.values[3::4,70:326,34:200]
    
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
    #x = x[29:]
    
    for i in range(vardata.shape[1]):
        for j in range(vardata.shape[2]):
            y = vardata[:,i,j]
            not_nan_ind = ~pl.isnan(y)
            if pl.where(not_nan_ind==True)[0].size == 0:
                pass
            else:
                m, b, r, p, se = stats.linregress(x[not_nan_ind],y[not_nan_ind])
                var_trnd[i,j] = m
                if p < 0.05:
                    vt_p[i,j] = p
                detrend_y = y - (m*x + b) # detrended variable
                out = stats.linregress(tele_dt[:],detrend_y)
                signal[i,j] = out[0]#stats.linregress(tele_dt,detrend_y).slope # NAO signal
                sp = out[3]#stats.linregress(tele_dt,detrend_y).pvalue
                if sp < 0.05:
                    sig_p[i,j] = sp
    
    return signal, sig_p, var_trnd, vt_p

def ComponentAndResidual(vardata,signal,TSI,x):
    """
    """
    tc_comp = pl.zeros_like(vardata,)
    tc_comp[:] = pl.float32('nan')
    tc_comp_trnd = pl.zeros_like(signal); tc_comp_trnd[:] = pl.float32('nan')
    tct_p = tc_comp_trnd.copy()
    nosig = tc_comp_trnd.copy(); nosig_p = nosig.copy()
    
    for yr in range(vardata.shape[0]):
        tc_comp[yr] = TSI[yr]*signal[:,:]
    
    for i in range(vardata.shape[1]):
        for j in range(vardata.shape[2]):
            out1 = stats.linregress(x,tc_comp[:,i,j])
            tc_comp_trnd[i,j] = out1[0]#stats.linregress(x,tc_comp[:,i,j]).slope
            sp = out1[3]#stats.linregress(x,tc_comp[:,i,j]).pvalue
            if sp < 0.05:
                tct_p[i,j] = sp
            
            out2 = stats.linregress(x[:],vardata[:,i,j]-tc_comp[:,i,j])
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
    gl.xlabel_style = {'color': 'k','size':10}
    gl.ylabel_style = {'color': 'k','size':10}
    
    return None

def Figure2x2(lon2,lat2,INPUTS,pvals,time_name,tc_caps):
    """
    """
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
    levs = pl.linspace(-0.04,0.04,17)#pl.linspace(-1,1,11)#[-4,-3,-2,-1,0,1,2,3,4]
    cs = ax1.contourf(lon2,lat2,INPUTS[0].T,cmap='seismic_r',#norm=pl.Normalize(-3,3),
                transform=ccrs.PlateCarree(),alpha=0.8,extend='both',
                levels=levs)
    cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
    
    ax1.contourf(lon2,lat2,pvals[0].T,hatches=['...'],colors='none')
    
    cb.set_label('$^\circ$C yr$^{-1}$',fontsize=12)
    cb.set_ticks(levs[::2])
    cb.set_ticklabels(levs[::2])
    GridLines(ax1,True,False,True,False)
    ax1.annotate('(a) '+name+' trend',(-19,69),bbox={'facecolor':'w'},fontsize=12)
    ################################################################################
    ax2 = pl.subplot(222,projection=proj,extent=ext)
    ax2.coastlines(linewidth=0.5,resolution='50m')
    ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
    levs = pl.linspace(-2,2,11)#pl.linspace(-15,15,11)
    #[-200,-150,-100,-75,-50,-25,-10,0,10,25,50,75,100,150,200]
    cs = ax2.contourf(lon2,lat2,INPUTS[1].T,cmap='seismic_r',#norm=pl.Normalize(-200,200),
                transform=ccrs.PlateCarree(),alpha=0.8,extend='max',
                levels=levs)
    
    ax2.contourf(lon2,lat2,pvals[1].T,hatches=['...'],colors='none')
    
    cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
    cb.set_label('$^\circ$C '+tc_caps+'I$^{-1}$',fontsize=12)
    cb.set_ticks(levs)
    cb.set_ticklabels(levs)
    #cb.ax.tick_params(labelsize=8)
    GridLines(ax2,True,False,False,True)
    ax2.annotate('(b) '+time_name+' '+caps+' signal',(-19,69),
                 bbox={'facecolor':'w'},fontsize=12)
    ###############################################################################
    ax3 = pl.subplot(223,projection=proj,extent=ext)
    ax3.coastlines(linewidth=0.5,resolution='50m')
    ax3.add_feature(borders_50m,linewidth=0.5,zorder=5)
    levs = pl.linspace(-0.04,0.04,17)#[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5]
    cs = ax3.contourf(lon2,lat2,INPUTS[2].T,cmap='seismic_r',#norm=pl.Normalize(-0.5,0.5),
                      transform=ccrs.PlateCarree(),alpha=1,extend='max',
                        levels=levs)
    
    ax3.contourf(lon2,lat2,pvals[2].T,hatches=['..'],colors='none',alpha=0.0)
    
    cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
    cb.set_label('$^\circ$C yr$^{-1}$',fontsize=12)
    cb.set_ticks(levs[::2])
    cb.set_ticklabels(levs[::2])
    #cb.ax.tick_params(labelsize=8)
    GridLines(ax3,False,False,True,False)
    ax3.annotate('(c) '+time_name+' '+caps+' component trend',(-19,69),
                 bbox={'facecolor':'w'},fontsize=12)
    ###############################################################################
    ax4 = pl.subplot(224,projection=proj,extent=ext)
    ax4.coastlines(linewidth=0.5,resolution='50m')
    ax4.add_feature(borders_50m,linewidth=0.5,zorder=5)
    levs = pl.linspace(-0.04,0.04,11)
    cs = ax4.contourf(lon2,lat2,INPUTS[3].T,cmap='seismic_r',#norm=pl.Normalize(-4,4),
                      transform=ccrs.PlateCarree(),alpha=0.8,extend='both',
                        levels=levs)
    
    ax4.contourf(lon2,lat2,pvals[3].T,hatches=['...'],colors='none')
    
    cb = pl.colorbar(cs,orientation='horizontal',shrink=0.9,pad=0.02)
    cb.set_label('$^\circ$C yr$^{-1}$',fontsize=12)
    cb.set_ticks(levs[::2])
    cb.set_ticklabels(levs[::2])
    GridLines(ax4,False,False,False,True)
    ax4.annotate('(d) residual trend',(-19,69),bbox={'facecolor':'w'},fontsize=12)
    
    pl.tight_layout()
    pl.subplots_adjust(left=0.04,right=0.96,bottom=0.07,top=0.96,hspace=0.08)
    
    #pl.savefig(indecis+'figures/'+name+'_remove_'+teleindex+'_djf_panels.png',dpi=350)
    
    return None

def SignalsOnly(vardata,tele_sns,lon2,lat2,teleindex,caps,name):
    """
    """
    sig_array = pl.zeros([4,lon2.size,lat2.size])
    sig_array[:] = pl.float32('nan')
    p_array = sig_array.copy()
    
    seasons = ['DJF','MAM','JJA','SON']
    
    for i in range(len(seasons)):
        data, TSI, x = PickSeason(seasons[i],tele_sns,vardata)
        
        #if i == 0:
        tele_dt = pl.detrend_linear(TSI[29:])
        #tele_trnd = stats.linregress(x,TSI)
        
        signal, sig_p, var_trnd, vt_p = TeleconSignal(data[29:],tele_dt,x[29:])
        
        sig_array[i] = signal
        p_array[i] = sig_p

    fig, ax = pl.subplots(2,2,figsize=(8,6))
    proj = ccrs.PlateCarree()
    ext = [-22,42,35,70]
    borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')
    
    levs = pl.linspace(-5,5,21)
    for i in range(len(seasons)):
        axx = pl.subplot(2,2,i+1,projection=proj,extent=ext)
        axx.coastlines(linewidth=0.5,resolution='50m')
        axx.add_feature(borders_50m,linewidth=0.5,zorder=5)
        cs = axx.contourf(lon2,lat2,sig_array[i].T,transform=ccrs.PlateCarree(),
                          cmap='seismic',levels=levs,alpha=0.8,extend='both')#,
                          #norm=MidpointNormalize(midpoint=0))
        
        axx.contourf(lon2,lat2,p_array[i].T,hatches=['...'],colors='none')
        
        axx.annotate(seasons[i],(-19,68),bbox={'facecolor':'w'},fontsize=12)
        
        if i == 0:
            GridLines(axx,False,True,True,False)
        elif i == 1:
            GridLines(axx,False,True,False,True)
        elif i == 2:
            GridLines(axx,False,True,True,False)
        elif i == 3:
            GridLines(axx,False,True,False,True)
        
    f = pl.gcf()
    colax = f.add_axes([0.15,0.1,0.7,0.03])
    cb = pl.colorbar(cs,orientation='horizontal',cax=colax)#pad=0.05,fraction=0.10,
    cb.set_label('days '+caps+'I$^{-1}$',fontsize=14)
    #cb.ax.tick_params(labelsize=16); cb.ax.set_aspect(0.09)
    
    pl.tight_layout()
    pl.subplots_adjust(bottom=0.12,left=0.06,right=0.94,hspace=-0.12,wspace=0.08,
                       top=1.0)
    pl.suptitle(caps+' '+name+' signal',y=0.97,fontsize=15)
    pl.savefig('/home/users/qx911590/INDECIS/figures/'+name+'_'+teleindex+\
                                                '_signal_2x2.png')
    
    return sig_array, p_array

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
    
    rdata = data[:,:,:]*rmask[None,:,:]
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

teleindex = 'eawr'
caps = 'EAWR'
name = 'uai'

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
#months = pl.asarray([i[5:7] for i in time])
#
#ind = pl.where((years=='1979') & (months=='07'))
#ind = ind[0][0]

tele = pl.reshape(tele[:,-1],newshape=(tele.shape[0]/12,12))
tele_sns = pl.zeros([tele.shape[0],4])
tele_sns[:,1] = pl.mean(tele[:,2:5],axis=1) # MAM
tele_sns[:,2] = pl.mean(tele[:,5:8],axis=1) # JJA
tele_sns[:,3] = pl.mean(tele[:,8:11],axis=1) # SON
tele_sns[1:,0] = (tele[:-1,-1]+tele[1:,0]+tele[1:,1])/3 # DJF
tele_sns[0,0] = pl.float32('nan')

# need to take mean tele index over April to October for gsr
#tele_ao = pl.mean(tele[:,3:10],axis=1)


#tele_sns = pl.reshape(tele_sns,newshape=(tele_sns.shape[0]*tele_sns.shape[1]))
lon2 = lon.values[70:326]
lat2 = lat.values[34:200]

# Pick a season (DJF, MAM, JJA, SON)
#data, TSI, x = PickSeason('DJF',tele_sns,data)
# or pick a month (1 to 12)
#data, TSI, x = PickMonth(2,tele,data)

#data  = data.values[:,70:326,34:200]
#tele_dt = pl.detrend_linear(tele_sns[1:,0]) # detrended index

#x = pl.linspace(1,67,67)
#tele_trnd = stats.linregress(x,tele_sns[1:,0]).slope # variable trend

#signal, sig_p, var_trnd, vt_p = TeleconSignal(data,tele_dt,x)

sig_array, sig_p = SignalsOnly(data,tele_sns,lon2,lat2,teleindex,caps,name)
sig_array[1,193,139] = pl.float32('nan')

#tc_comp, tc_comp_trnd, tct_p, nosig, nosig_p = ComponentAndResidual(data,signal,TSI,x)


#INPUTS = pl.array([var_trnd,signal,tc_comp_trnd,nosig])
#pvals = pl.array([vt_p,sig_p,tct_p,nosig_p])
#
#Figure2x2(lon2,lat2,INPUTS,pvals,'DJF',caps)

english_channel = [(-1.7,46),(9.2,51.9),(7.9,53.8),(-5.9,53.4)]
baltic_sea = [(8.2,53.4),(8.2,57.5),(19.6,66.1),(28.2,66.1),(28.2,57.8),
              (19.5,53.4)]
denmark = [(6.3,53.9),(6.3,57.5),(11.9,57.5),(11.9,53.9)]
ireland = [(-10.9,51.4,),(-10.9,55.4,),(-5.3,55.4),(-5.3,52.2),(-8.8,51.4)]
scand_ne = [(9.2,62.0),(21.8,71.5),(27.5,71.5),(27.5,65.2),(22.8,65.2),(16.8,62.0)]
france = [(-1.8,43.3),(-2.5,48.0),(6.3,50.5),(8.0,42.5),(3.5,42.3)]
balkans_nw = [(20.4,44.2),(14.1,45.0),(17.0,47.7),(23.2,48.4)]
balkans_se = [(19.5,41.6),(22.9,44.9),(26.7,45.4),(27.6,47.0),(30.6,46.62),
              (26.3,40.6),(22.6,40.2),(22.3,37.9)]
italy = [(8.3,43.9),(8.2,45.9),(12.4,45.6),(14.3,43.0),(13.6,41.2),(9.4,44.2)]
easteur_n = [(29.6,51.1),(24.9,53.8),(24.0,56.4),(20.4,56.4),(23.2,59.6),(31.4,55.4)]
easteur_s = [(28.4,43.5),(24.9,43.5),(23.3,45.4),(24.0,49.5),(33.4,48.0),
             (34.0,45.9),(30.7,46.0)]

#V = RegionCalc(scand_ne,lon2,lat2,data)
#V_an = V - V.mean()
#S = RegionCalc(scand_ne,lon2,lat2,tc_comp)
#
#years = pl.linspace(1950,2017,68)
#a = pl.where(TSI>1); a = a[0]
#b = pl.where(TSI<-1); b = b[0]
#
#pos = pl.array([years[a],TSI[a],S[a],V_an[a]]).T
#neg = pl.array([years[b],TSI[b],S[b],V_an[b]]).T
#
#fig, ax = pl.subplots(1,2,figsize=(9,4))#pl.figure(2)
#
#ax1 = pl.subplot(121)
#ax1.plot(years,V_an,label=name+' anomalies')
#ax1.plot(years,S,label='MAM '+caps+' component')
#ax1.grid(axis='y',ls='--',color='grey')
#pl.xlim(years[0],years[-1])
#pl.ylabel('days',fontsize=13)
#pl.legend()
#
#ax2 = pl.subplot(122)#pl.figure(3)
#ax2.plot(years,TSI,color='k',label='MAM '+caps+' index')
#ax2.grid(axis='y',ls='--',color='grey')
#pl.xlim(years[0],years[-1])
#pl.ylabel('index',fontsize=13)
#pl.legend()
#
#pl.suptitle('NE Scandinavia')
#pl.subplots_adjust(left=0.08,right=0.99)
#pl.savefig(indecis+'/figures/'+name+'_'+teleindex+'_mam_component_ts_nesc.png')

#f = open(indecis+'/output_files/'+name+'_'+teleindex+'_mam_posinds_nesc.csv','w')
#f.write('year, index, component, anomaly \n')
#pl.savetxt(f,pos,delimiter=',')
#f.close()
#
#f = open(indecis+'/output_files/'+name+'_'+teleindex+'_mam_neginds_nesc.csv','w')
#f.write('year, index, component, anomaly \n')
#pl.savetxt(f,neg,delimiter=',')
#f.close()