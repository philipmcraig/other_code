# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:35:28 2019

@author: qx911590
"""

from __future__ import division
import pylab as pl
import glob
from mpl_toolkits.basemap import Basemap
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


pl.close('all')
homedir = '/home/users/qx911590/'
resdir = homedir + 'weatherrescue/'

#stations = pl.genfromtxt(homedir+'allyears.csv',delimiter=',',dtype=None)
#names = stations[:,0]
stations = pd.read_csv(homedir+'allyears_1861-1875.csv',header=None)
stations = pl.asarray(stations)
names = stations[:,0]

#pl.figure(figsize=(12,8))
#m = Basemap(projection='cyl',llcrnrlon=-32,llcrnrlat=35,urcrnrlon=28,
#            urcrnrlat=70,resolution='i')
#m.drawcoastlines(linewidth=0.5,color='grey')
#m.drawcountries(linewidth=0.5,color='grey')
#
#for i in range(len(names)):
#    m.plot(stations[i,2],stations[i,1],latlon=True,lw=0,marker='.',ms=5,
           #color='r')#,label=str(int(i+1))+'. '+names[i])
#    if i == 12:
#        pl.annotate(str(int(i+1)),xy=(stations[i,2]-1,stations[i,1]))#,xytext=None,
#    elif i == 13:
#        pl.annotate(str(int(i+1)),xy=(stations[i,2]+0.25,stations[i,1]-0.25))
#    else:
#        pl.annotate(str(int(i+1)),xy=(stations[i,2],stations[i,1]))

#handles = [i+1 for i in range(len(names))]
#pl.legend(loc=(0,-0.2),ncol=10,columnspacing=0.2,handletextpad=0.1,handlelength=0)
#pl.legend(handles,names,loc=0)
#pl.tight_layout()

fig = pl.figure(figsize=(12.5,10))
ax = pl.axes(projection=ccrs.PlateCarree())
ax.set_extent([-12,26,37,67])
ax.coastlines(resolution='50m',linewidth=0.5,zorder=5,color='lightgrey')

borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='lightgrey',
                                        facecolor=cfeature.COLORS['land'])
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)#cfeature.BORDERS
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['water'])
ax.add_feature(ocean_50m,alpha=0.5)
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
ax.add_feature(land_50m,alpha=0.5)

#shape_feature = ShapelyFeature(Reader(homedir+'cntry1914').geometries(),
#                                ccrs.PlateCarree(), edgecolor='black')
#ax.add_feature(shape_feature,facecolor=cfeature.COLORS['land'],linewidth=0.5,
#               edgecolor='lightgrey')

ax.plot(stations[:,2],stations[:,1],lw=0,marker='o',ms=3,color='r',
        transform=ccrs.Geodetic(),zorder=6)

for i in range(len(names)):
    ax.plot(stations[i,2],stations[i,1],lw=0,marker='o',ms=0,color=None,
        transform=ccrs.Geodetic(),zorder=0,alpha=0,label=str(i+1)+'. '+names[i])
    if i == 11:
        pl.text(stations[i,2]-0.65,stations[i,1],str(i+1),zorder=7,weight='bold')
    elif i == 15:
        pl.text(stations[i,2],stations[i,1],str(i+1),zorder=7,weight='bold')
    elif i == 16:
        pl.text(stations[i,2]-0.7,stations[i,1]-0.2,str(i+1),zorder=7,weight='bold')
    elif i == 23:
        pl.text(stations[i,2]-0.7,stations[i,1]-0.2,str(i+1),zorder=7,weight='bold')
    elif i == 26:
        pl.text(stations[i,2],stations[i,1]-0.4,str(i+1),zorder=7,weight='bold')
    elif i == 28:
        pl.text(stations[i,2],stations[i,1]-0.4,str(i+1),zorder=7,weight='bold')
    elif i == 33:
        pl.text(stations[i,2]-0.2,stations[i,1]+0.1,str(i+1),zorder=7,weight='bold')
    elif i == 34:
        pl.text(stations[i,2]-0.7,stations[i,1]-0.5,str(i+1),zorder=7,weight='bold')
    elif i == 35:
        pl.text(stations[i,2],stations[i,1]-0.3,str(i+1),zorder=7,weight='bold')
    elif i == 40:
        pl.text(stations[i,2],stations[i,1]+0.1,str(i+1),zorder=7,weight='bold')
    elif i == 41:
        pl.text(stations[i,2],stations[i,1]-0.3,str(i+1),zorder=7,weight='bold')
    elif i == 49:
        pl.text(stations[i,2]-0.75,stations[i,1],str(i+1),zorder=7,weight='bold')
    elif i == 54:
        pl.text(stations[i,2]+0.1,stations[i,1]-0.5,str(i+1),zorder=7,weight='bold')
    elif i == 57:
        pl.text(stations[i,2],stations[i,1]-0.4,str(i+1),zorder=7,weight='bold')
    elif i == 66:
        pl.text(stations[i,2]-0.5,stations[i,1]-0.4,str(i+1),zorder=7,weight='bold')
    else:
        pl.text(stations[i,2],stations[i,1],str(i+1),zorder=7,weight='bold')

#pl.rcParams["legend.labelspacing"] = 0
pl.legend(loc=(0.54,0.01),ncol=4,columnspacing=-0.5,handlelength=0,borderpad=0.2)

#ax.set_xticks([-30,-20,-10,0,10,20,26],crs=ccrs.PlateCarree())
#lon_formatter = LongitudeFormatter(zero_direction_label=False)
#ax.xaxis.set_major_formatter(lon_formatter)

#ax.set_yticks([35,45,55,65])
#lat_formatter = LatitudeFormatter()
#ax.yaxis.set_major_formatter(lat_formatter)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = True; gl.ylabels_right=False
gl.xlocator = mticker.FixedLocator([-40,-30,-20,-10,0,10,20,26,30])
gl.ylocator = mticker.FixedLocator([25,35,45,55,65,75])
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'k','size':11}
gl.ylabel_style = {'color': 'k','size':11}

pl.tight_layout()
pl.subplots_adjust(top=0.99,bottom=0.04,right=1.00,left=0.01)
#pl.savefig(homedir+'allstations_map_paper.pdf',dpi=300)
#pl.show()

left, bottom, width, height = [0.001, 0.78, 0.4, 0.2]
ax2 = fig.add_axes([left, bottom, width, height],projection=ccrs.PlateCarree())
ax2.set_extent([-90,-40,40,65])
ax2.coastlines(resolution='50m',linewidth=0.5,zorder=5,color='lightgrey')
ax2.add_feature(borders_50m,linewidth=0.5,zorder=5)
ax2.add_feature(ocean_50m,alpha=0.5)
ax2.add_feature(land_50m,alpha=0.5)

ax2.plot(stations[-1,2],stations[-1,1],lw=0,marker='o',ms=3,color='r',
        transform=ccrs.Geodetic(),zorder=6)
pl.text(stations[-1,2],stations[-1,1],'71',zorder=7,weight='bold')

gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False; gl.ylabels_right = True
gl.xlocator = mticker.FixedLocator([-100,-90,-80,-70,-60,-50,-40,-30])
gl.ylocator = mticker.FixedLocator([25,35,45,55,65,75])
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'k','size':10}
gl.ylabel_style = {'color': 'k','size':10}

#pl.savefig(homedir+'allstations_map_paper_2022.png',dpi=400)
#pl.savefig(homedir+'allstations_map_paper_2022.pdf',dpi=400)