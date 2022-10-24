from __future__ import division
import pylab as pl
from mpl_toolkits.basemap import Basemap
from os import listdir
from os.path import isfile, join
import itertools
from netCDF4 import Dataset
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
from scipy.stats import pearsonr
from scipy.interpolate import interp1d


def NearestIndex(array_in,point_in):
    """Function to the the nearest index to a specified geographic co-ordinate from an
    array of latitude or longitude co-ordinates
    
    Args:
        array_in (array): longitude or latitude array
        point_in (float): longitude or latitude co-ordinate
    
    Returns:
        index (int): index of array which has value closest to point_in
    """
    index = pl.absolute(array_in-point_in).argmin()
    
    return index

def BilinInterp(relpt,lon,lat,flux):
    """Function to interpolate to trajectory release point from 4 nearest grid
    points. e.g. https://en.wikipedia.org/wiki/Bilinear_interpolation
    
    Args:
        relpt (array): longitude & latitude co-ordinates of release point
        lon (array): longitude array from ERA-Interim
        lat (array): latitude array from ERA-Interim
        flux (array): variable from ERA-Interm requiring interpolation
    
    Returns:
        F (float): variable interpolated to release point
    """
    # First find p1,p2,p3,p4: the points forming a rectangle around relpt
    # Start with empty arrays for p1,p2,p3,p4; p = (x,y,flux):
    p1 = pl.zeros([3]); p2 = pl.zeros([3]); p3 = pl.zeros([3]); p4 = pl.zeros([3])
    # if release point longitude co-ordinate greater than max. ERA-Int longitude
    if relpt[0] > lon[-1]:
        a = -1 # take max ERA-Interim longitude as nearest longitude index
    else:
        a = NearestIndex(lon,relpt[0]) # nearest longitude index
    
    if relpt[-1] > lat[0]:
        latx = pl.zeros([lat.size+1]); latx[1:] = lat; latx[0] = 90.0
        lat = latx.copy()
        b = NearestIndex(lat,relpt[1])
        flux2 = pl.zeros([flux.shape[0]+1,flux.shape[1]])
        flux2[1:,:] = flux; flux2[0] = flux[0]
        flux = flux2
    elif relpt[-1] < lat[-1]:
        latx = pl.zeros([lat.size+1]); latx[:-1] = lat; latx[-1] = -90.0
        lat = latx.copy()
        b = NearestIndex(lat,relpt[1])
        flux2 = pl.zeros([flux.shape[0]+1,flux.shape[1]])
        flux2[:-1,:] = flux; flux2[-1] = flux[-1]
        flux = flux2
    else:
        b = NearestIndex(lat,relpt[1]) # nearest latitude index
    
    if relpt[0] == lon[a] and relpt[1] == lat[b]:
        return flux[b,a]
    elif relpt[0] == lon[a]:
        F = interp1D(relpt,lon,lat,flux)
        return F
    elif relpt[1] == lat[b]:
        F = interp1D(relpt,lon,lat,flux)
        return F
    
    if lon[a] < relpt[0]: # nearest lon west of relpt
        p1[0] = lon[a]; p3[0] = lon[a];  p2[0] = lon[a+1]; p4[0] = lon[a+1]
    elif lon[a] > relpt[0]: # nearest lon east of relpt
        p2[0] = lon[a]; p4[0] = lon[a]; p1[0] = lon[a-1]; p3[0] = lon[a-1]
        
    # does not take 0 meridian into account yet

    
    if lat[b] < relpt[1]: # nearest lat south of relpt
        p1[1] = lat[b]; p2[1] = lat[b]; p3[1] = lat[b-1]; p4[1] = lat[b-1]
    elif lat[b] > relpt[1]: # nearest lat north of relpt
        p3[1] = lat[b]; p4[1] = lat[b]; p1[1] = lat[b+1]; p2[1] = lat[b+1]
    #elif lat[b] == relpt[1]: # lat equal to relpt
    #    p3[1] = lat[b]; p4[1] = lat[b]; p1[1] = lat[b]; p2[1] = lat[b]
    
    # values of flux at p1,p2,p3,p4:
    nrth_lat = pl.where(lat==p3[1]); sth_lat = pl.where(lat==p1[1])
    west_lon = pl.where(lon==p1[0]); east_lon = pl.where(lon==p2[0])
    p1[2] = flux[sth_lat[0][0],west_lon[0][0]]
    p2[2] = flux[sth_lat[0][0],east_lon[0][0]]
    p3[2] = flux[nrth_lat[0][0],west_lon[0][0]]
    p4[2] = flux[nrth_lat[0][0],east_lon[0][0]]
    
    # if release point longitude co-ordinate greater than max. ERA-Int longitude
    if relpt[0] > lon[-1]:
        # dx is 360 + min ERA-Int longitude minus max ERA-Int longitude
        dx = (360. + lon[0]) - lon[-1]
    else:
        dx = p2[0] - p1[0] # dx is difference between 2 nearest longitudes
    dy = p3[1] - p2[1] # dy is difference between 2 nearest latitudes
    
    # if release point longitude co-ordinate greater than max. ERA-Int longitude
    if relpt[0] > lon[-1]:
        # need 360 + p2 longitude
        f1 = (((360+p2[0])-relpt[0])/dx)*p1[2] + ((relpt[0]-p1[0])/dx)*p2[2]
        f2 = (((360+p2[0])-relpt[0])/dx)*p3[2] + ((relpt[0]-p1[0])/dx)*p4[2]
    else:
        f1 = ((p2[0]-relpt[0])/dx)*p1[2] + ((relpt[0]-p1[0])/dx)*p2[2]
        f2 = ((p2[0]-relpt[0])/dx)*p3[2] + ((relpt[0]-p1[0])/dx)*p4[2]
    
    F = ((p3[1]-relpt[1])/dy)*f1 + ((relpt[1]-p2[1])/dy)*f2
    
    return F


def interp1D(relpt,lon,lat,flux):
    """
    """
    # if release point longitude co-ordinate greater than max. ERA-Int longitude
    if relpt[0] > lon[-1]:
        a = -1 # take max ERA-Interim longitude as nearest longitude index
    else:
        a = NearestIndex(lon,relpt[0]) # nearest longitude index
    b = NearestIndex(lat,relpt[1]) # nearest latitude index
    
    # two IF things for is the lat or lon co-ordinate the problem
    if relpt[0] == lon[a]:
        if relpt[1] > lat[b]:
            X = (lat[b],lat[b-1]); Y = (flux[b,a],flux[b-1,a])
        elif relpt[1] < lat[b]:
            X = (lat[b+1],lat[b]); Y = (flux[b+1,a],flux[b,a])
        f = interp1d(X,Y); F = f(relpt[1])
    elif relpt[1] == lat[b]:
        if relpt[0] > lon[a]:
            if lon[a] == lon[-1]:
                X = (lon[a],lon[a+1]+360); Y = (flux[b,a],flux[b,0])
            else:
                X = (lon[a],lon[a+1]); Y = (flux[b,a],flux[b,a+1])
        elif relpt[0] < lon[a]:
            X = (lon[a-1],lon[a]); Y = (flux[b,a-1],flux[b,a])
        f = interp1d(X,Y); F = f(relpt[0])
    
    return F

def AreaFullGaussianGrid(radius,delta_lambda,latpair):
    """ Function for calculating the area of a grid cell on the surface of a sphere
    where the grid is a full Gaussian Grid, e.g.:
    https://badc.nerc.ac.uk/help/coordinates/cell-surf-area.html
    
    Args:
        radius (float): predefined radius of Earth, 6.37*(10**6) meters
        delta_lambda (float): longitudinal distance between grid points in radians,
                            for a full Gaussian grid delta_lambda is constant
        latpair (tuple): pair of latitude points in radians for calculating 
                        delta_mu, the difference of the sines of each latitude
        
    Returns:
        area (float): area of a grid cell between 2 latitudes in m**2
    
    Example:
        >>> for i in range(latitude):
        ...    for j in range(longitude):
        ...        areas[i,j] = AreaFullGaussianGrid(radius,deta_lambda,latpair[i])
    """

    delta_mu = pl.sin(latpair[0])-pl.sin(latpair[1])
    
    area = (radius**2)*delta_lambda*delta_mu

    return area

def HalfGrid(lat):
    """Function to create latitude half-grid
    
    Args:
        lat (array): latitude array in radians
    
    Returns:
        lath (array): half-grid latitude array in radians
    """
    # set up empty array, size one more than lat
    lath = pl.zeros([lat.size+1])
    # set first & last elements of lat_half seperately as pi/2 & -pi/2:
    lath[0] = -(pl.pi)/2; lath[-1] = (pl.pi)/2
    # loop over lat_half from index 1 to index -2:
    for i in range(1,lath.size-1): # loops over 256
        lath[i] = 0.5*(lat[i]+lat[i-1])
    
    return lath