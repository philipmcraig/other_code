#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:20:04 2018

@author: qx911590
"""
import pylab as pl

def WGS84toOSGB36(lat, lon):
    #First convert to radians
    #These are on the wrong ellipsoid currently: GRS80. (Denoted by _1)
    lat_1 = lat*pl.pi/180
    lon_1 = lon*pl.pi/180
 
    #Want to convert to the Airy 1830 ellipsoid, which has the following:
    a_1, b_1 =6378137.000, 6356752.3141 #The GSR80 semi-major and semi-minor axes used for WGS84(m)
    e2_1 = 1- (b_1*b_1)/(a_1*a_1)   #The eccentricity of the GRS80 ellipsoid
    nu_1 = a_1/pl.sqrt(1-e2_1*pl.sin(lat_1)**2)
 
    #First convert to cartesian from spherical polar coordinates
    H = 0 #Third spherical coord.
    x_1 = (nu_1 + H)*pl.cos(lat_1)*pl.cos(lon_1)
    y_1 = (nu_1+ H)*pl.cos(lat_1)*pl.sin(lon_1)
    z_1 = ((1-e2_1)*nu_1 +H)*pl.sin(lat_1)
 
    #Perform Helmut transform (to go between GRS80 (_1) and Airy 1830 (_2))
    s = 20.4894*10**-6 #The scale factor -1
    tx, ty, tz = -446.448, 125.157, -542.060 #The translations along x,y,z axes respectively
    rxs,rys,rzs = -0.1502, -0.2470, -0.8421  #The rotations along x,y,z respectively, in seconds
    rx, ry, rz = rxs*pl.pi/(180*3600.), rys*pl.pi/(180*3600.), rzs*pl.pi/(180*3600.) #In radians
    x_2 = tx + (1+s)*x_1 + (-rz)*y_1 + (ry)*z_1
    y_2 = ty + (rz)*x_1  + (1+s)*y_1 + (-rx)*z_1
    z_2 = tz + (-ry)*x_1 + (rx)*y_1 +  (1+s)*z_1
 
    #Back to spherical polar coordinates from cartesian
    #Need some of the characteristics of the new ellipsoid
    a, b = 6377563.396, 6356256.909 #The GSR80 semi-major and semi-minor axes used for WGS84(m)
    e2 = 1- (b*b)/(a*a)   #The eccentricity of the Airy 1830 ellipsoid
    p = pl.sqrt(x_2**2 + y_2**2)
 
    #Lat is obtained by an iterative proceedure:
    lat = pl.arctan2(z_2,(p*(1-e2))) #Initial value
    latold = 2*pl.pi
    while abs(lat - latold)>10**-16:
        lat, latold = latold, lat
        nu = a/pl.sqrt(1-e2*pl.sin(latold)**2)
        lat = pl.arctan2(z_2+e2*nu*pl.sin(latold), p)
 
    #Lon and height are then pretty easy
    lon = pl.arctan2(y_2,x_2)
    H = p/pl.cos(lat) - nu
 
    #E, N are the British national grid coordinates - eastings and northings
    F0 = 0.9996012717                   #scale factor on the central meridian
    lat0 = 49*pl.pi/180                    #Latitude of true origin (radians)
    lon0 = -2*pl.pi/180                    #Longtitude of true origin and central meridian (radians)
    N0, E0 = -100000, 400000            #Northing & easting of true origin (m)
    n = (a-b)/(a+b)
 
    #meridional radius of curvature
    rho = a*F0*(1-e2)*(1-e2*pl.sin(lat)**2)**(-1.5)
    eta2 = nu*F0/rho-1
 
    M1 = (1 + n + (5/4)*n**2 + (5/4)*n**3) * (lat-lat0)
    M2 = (3*n + 3*n**2 + (21/8)*n**3) * pl.sin(lat-lat0) * pl.cos(lat+lat0)
    M3 = ((15/8)*n**2 + (15/8)*n**3) * pl.sin(2*(lat-lat0)) * pl.cos(2*(lat+lat0))
    M4 = (35/24)*n**3 * pl.sin(3*(lat-lat0)) * pl.cos(3*(lat+lat0))
 
    #meridional arc
    M = b * F0 * (M1 - M2 + M3 - M4)         
 
    I = M + N0
    II = nu*F0*pl.sin(lat)*pl.cos(lat)/2
    III = nu*F0*pl.sin(lat)*pl.cos(lat)**3*(5- pl.tan(lat)**2 + 9*eta2)/24
    IIIA = nu*F0*pl.sin(lat)*pl.cos(lat)**5*(61- 58*pl.tan(lat)**2 + pl.tan(lat)**4)/720
    IV = nu*F0*pl.cos(lat)
    V = nu*F0*pl.cos(lat)**3*(nu/rho - pl.tan(lat)**2)/6
    VI = nu*F0*pl.cos(lat)**5*(5 - 18*pl.tan(lat)**2 + pl.tan(lat)**4 + 14*eta2 - 58*eta2*pl.tan(lat)**2)/120
 
    N = I + II*(lon-lon0)**2 + III*(lon- lon0)**4 + IIIA*(lon-lon0)**6
    E = E0 + IV*(lon-lon0) + V*(lon- lon0)**3 + VI*(lon- lon0)**5
 
    #Job's a good'n.
    return E,N