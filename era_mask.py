# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:48:40 2018

@author: np838619
"""

from __future__ import division
import pylab as pl
from netCDF4 import Dataset
import cartopy
import cartopy.crs as ccrs
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
from mpl_toolkits.basemap import Basemap

def BasinPolys():
    """
    """
    ##### NEED ALL LINES DEFINING SEGMENTS FOR TRAJECTORY RELEASE LINES #######
    sheddir = '/home/np838619/Watershed/shed_defs/' # directory with data
    
    endpts1 = pl.genfromtxt(sheddir+'Am_clicks.txt',skip_header=5) # Americas
    endpts2 = pl.genfromtxt(sheddir+'AfMe_clicks.txt',skip_header=5) # Africa
    endpts3 = pl.genfromtxt(sheddir+'EAA_clicks.txt',skip_header=5) # East Asia
    
    # Arctic lines:
    endpts4 = pl.genfromtxt(sheddir+'ArA_clicks.txt',skip_header=5) # Atlantic
    endpts5 = pl.genfromtxt(sheddir+'ArI_clicks.txt',skip_header=5) # Indian
    endpts6 = pl.genfromtxt(sheddir+'ArP_clicks.txt',skip_header=5) # Pacific
    
    # Southern Ocean lines
    endpts7 = pl.genfromtxt(sheddir+'SOA_clicks.txt',skip_header=5) # Atlantic
    endpts8 = pl.genfromtxt(sheddir+'SOI_clicks.txt',skip_header=5) # Indian
    endpts9 = pl.genfromtxt(sheddir+'SOP_clicks.txt',skip_header=5) # Pacific
    
    # Need traj release points for Arctic & Southern catchments because
    # end points lines means that latitude lines aren't correctly followed on
    # polar projections
    endpts10 = pl.genfromtxt(sheddir+'Ar_traj_release_new.txt',skip_header=5)
    endpts11 = pl.genfromtxt(sheddir+'SO_traj_release_new.txt',skip_header=5)
    
    ###### NEED 1 MAP FOR ATLANTIC/INDIAN CATCHMENTS, ANOTHER FOR PACIFIC #####
    ####### ALSO NEED SEPERATE MAPS FOR ARCTIC AND SOUTHERN CATCHMENTS ########
    m1 = Basemap(projection='cyl',llcrnrlon=-180,llcrnrlat=-80,urcrnrlon=180,urcrnrlat=80)
    m2 = Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=-80,urcrnrlon=360,urcrnrlat=80)
    m3 = Basemap(projection='npaeqd',boundinglat=25,lon_0=270,round=True)
    m4 = Basemap(projection='spstere',boundinglat=-13,lon_0=270,round=True)

    ####### DEFINE POLYGONS FOR ATLANTIC, INDIAN AND PACIFIC CATCHMENT ########
    atl_bnd = CatchLine(endpts1,endpts4,endpts2,endpts7) # Atlantic boundary
    ind_bnd = CatchLine(endpts2,endpts5,endpts3,endpts8) # Indian boundary
    #e6 = endpts6; e1 = endpts1#; e1[:,0] = e1[:,0] + 360.
    # Negative lon co-ords in endpts1 & endpts6, correct this:
    e1 = Add360(endpts1); e6 = Add360(endpts6)
    #for i in range(e6.shape[0]):
    #    if e6[i,0] < 0.:
    #        e6[i,0] = e6[i,0] + 360.
    pac_bnd = CatchLine(endpts3,e6,e1,endpts9) # Pacific boundary
    # empty arrays for Arctic & Southern map co-ords, convert to map co-ords:
    arc_bnd = pl.zeros_like(endpts10); arc_bnd[:,0], arc_bnd[:,1] = m3(endpts10[:,0],endpts10[:,1])
    sou_bnd = pl.zeros_like(endpts11); sou_bnd[:,0], sou_bnd[:,1] = m4(endpts11[:,0],endpts11[:,1])
    
    # Make Polygons for Ocean catchments:
    atl_ply = CatchPoly(atl_bnd,m1); ind_ply = CatchPoly(ind_bnd,m1)
    pac_ply = CatchPoly(pac_bnd,m2); arc_ply = CatchPoly(arc_bnd,m3)
    sou_ply = CatchPoly(sou_bnd,m4)
    
    return atl_bnd, ind_bnd, pac_bnd, arc_bnd, sou_bnd

def BasinMasks(lat,lon,boundary,basin,mask):
    """
    """
    ###### NEED 1 MAP FOR ATLANTIC/INDIAN CATCHMENTS, ANOTHER FOR PACIFIC #####
    ####### ALSO NEED SEPERATE MAPS FOR ARCTIC AND SOUTHERN CATCHMENTS ########
    m1 = Basemap(projection='cyl',llcrnrlon=-180,llcrnrlat=-80,urcrnrlon=180,urcrnrlat=80)
    m2 = Basemap(projection='cyl',llcrnrlon=0,llcrnrlat=-80,urcrnrlon=360,urcrnrlat=80)
    m3 = Basemap(projection='npaeqd',boundinglat=25,lon_0=270,round=True)
    m4 = Basemap(projection='spstere',boundinglat=-13,lon_0=270,round=True)
    
    basmask = pl.zeros_like(mask)
    lon2 = lon - 180
    
    basPath = mplPath.Path(list(boundary))
    #atlPath = mplPath.Path(list(atl_bnd))
    #indPath = mplPath.Path(list(ind_bnd))
    #pacPath = mplPath.Path(list(pac_bnd))
    #arcPath = mplPath.Path(list(arc_bnd))
    #souPath = mplPath.Path(list(sou_bnd))
    
    for i in range(lat.size):
        for j in range(lon.size):
            if basin =='atl':
                a,b = m1(lon2[j],lat[i],inverse=False) # Atlantic
                X = basPath.contains_point((a,b))
            elif basin == 'ind':
                a,b = m1(lon[j],lat[i],inverse=False) # Indian
                X = basPath.contains_point((a,b))
            elif basin == 'pac':
                a,b = m2(lon[j],lat[i],inverse=False) # Pacific
                X = basPath.contains_point((a,b))
            elif basin == 'arc':
                a,b = m3(lon[j],lat[i],inverse=False) # Arctic
                X = basPath.contains_point((a,b))
            elif basin == 'sou':
                a,b = m4(lon[j],lat[i],inverse=False) # Southern
                X = basPath.contains_point((a,b))
            
            if X == True:
                basmask[i,j] = 1
            else:
                basmask[i,j] = pl.float32('nan')
    
    return basmask

exec(open('/home/np838619/Trajectory/trajfuncs.py').read())
exec(open('/home/np838619/PminusE_data/ERA_Int/functions.py').read())

pl.close('all')
clusdir = '/glusterfs/scenario/users/np838619/'
eradir = clusdir + 'ERA/'
empdir = '/home/np838619/PminusE_data/ERA_Int/'

ncfile = Dataset(empdir+'direct_EmP.nc','r')
eralat = ncfile.variables['lat'][:]
eralon = ncfile.variables['lon'][:]
lsm = ncfile.variables['LSM'][0]
ncfile.close()

#land_pts = pl.where(sss_mn==sss_mn.min())
#for i in range(land_pts[0].size):
#    sss_mn[land_pts[0][i],land_pts[1][i]] = pl.float32('nan')


#n = pl.isnan(sss_mn)
#mask = pl.zeros([lat.size,lon.size])
#for i in range(lat.size):
#    for j in range(lon.size):
#        if n[i,j] == True:
#            mask[i,j] = 1.

atl_bnd,ind_bnd,pac_bnd,arc_bnd,sou_bnd = BasinPolys()

atlPath = mplPath.Path(list(atl_bnd))
atl_mask = BasinMasks(eralat,eralon,atl_bnd,'atl',lsm)
AM = pl.zeros_like(atl_mask)
AM[:,:256] = atl_mask[:,256:]; AM[:,256:] = atl_mask[:,:256]
atl_mask = AM; del AM

indPath = mplPath.Path(list(ind_bnd))
ind_mask = BasinMasks(eralat,eralon,ind_bnd,'ind',lsm)

pacPath = mplPath.Path(list(pac_bnd))
pac_mask = BasinMasks(eralat,eralon,pac_bnd,'pac',lsm)

arcPath = mplPath.Path(list(arc_bnd))
arc_mask = BasinMasks(eralat,eralon,arc_bnd,'arc',lsm)

souPath = mplPath.Path(list(sou_bnd))
sou_mask = BasinMasks(eralat,eralon,sou_bnd,'sou',lsm)


#newnc = Dataset(eradir+'era_basin_masks.nc','w')
#
#lat_dim = newnc.createDimension('lat',eralat.size)
#lon_dim = newnc.createDimension('lon',eralon.size)
#lat_in = newnc.createVariable('lat',pl.float64,('lat',))
#lat_in.units = 'degrees_north'
#lat_in.long_name = 'latitude'
#lon_in = newnc.createVariable('lon',pl.float64,('lon',))
#lon_in.units = 'degrees_east'
#lon_in.long_name = 'longitude'
#lat_in[:] = eralat[:]
#lon_in[:] = eralon[:]
#
#atlmsk_in = newnc.createVariable('Atlantic mask',pl.float64,('lat','lon'))
#atlmsk_in.standard_name = 'Atlantic Ocean ERA mask'
#atlmsk_in[:,:] = atl_mask[:,:]
#
#indmsk_in = newnc.createVariable('Indian mask',pl.float64,('lat','lon'))
#indmsk_in.standard_name = 'Indian Ocean ERA mask'
#indmsk_in[:,:] = ind_mask[:,:]
#
#pacmsk_in = newnc.createVariable('Pacific mask',pl.float64,('lat','lon'))
#pacmsk_in.standard_name = 'Pacific Ocean ERA mask'
#pacmsk_in[:,:] = pac_mask[:,:]
#
#arcmsk_in = newnc.createVariable('Arctic mask',pl.float64,('lat','lon'))
#arcmsk_in.standard_name = 'Arctic Ocean ERA mask'
#arcmsk_in[:,:] = arc_mask[:,:]
#
#soumsk_in = newnc.createVariable('Southern mask',pl.float64,('lat','lon'))
#soumsk_in.standard_name = 'Southern Ocean ERA mask'
#soumsk_in[:,:] = sou_mask[:,:]
#
#mask_in = newnc.createVariable('Land-sea mask',pl.float64,('lat','lon'))
#mask_in.standard_name = 'ERA land-sea mask'
#mask_in[:,:] = lsm[:,:]
#
#newnc.close()