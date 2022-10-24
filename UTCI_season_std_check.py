from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib as mpl
import numpy as np

plt.close('all')
#path = 'C:\\Users\\junguoliao\\Desktop\\INDECIS project\\program\\utci_season.nc'
path = '/home/users/qx911590/INDECIS/ncfiles/utci_season.nc'
file = Dataset(path)


lat, lon = file.variables['latitude'][:], file.variables['longitude'][:]
time, utci = file.variables['time'][:], file.variables['utci'][:]

x,y = np.meshgrid(lat, lon)

Name_season = ['DJF', 'MAM', 'JJA', 'SON' ]

'''
# Plot four seasonal mean utci plots in subplots (for check)
season_ave = 'season_1' 
season_ave = np.nanmean(utci[0::4, :, :] , axis = 0)

fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(221)
ax.set_title('ave of UTCI in DJF (check)')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_ave, cmap = 'rainbow')
cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 15), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)



season_ave = 'season_2' 
season_ave = np.nanmean(utci[1::4, :, :] , axis = 0)

ax = fig.add_subplot(222)
ax.set_title('ave of UTCI in MAM')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_ave, cmap = 'rainbow')
cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 15), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)


season_ave = 'season_3' 
season_ave = np.nanmean(utci[2::4, :, :] , axis = 0)

ax = fig.add_subplot(223)
ax.set_title('ave of UTCI in JJA')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_ave, cmap = 'rainbow')
cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 15), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)


season_ave = 'season_4' 
season_ave = np.nanmean(utci[3::4, :, :] , axis = 0)

ax = fig.add_subplot(224)
ax.set_title('ave of UTCI in SON')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_ave, cmap = 'rainbow')
cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 10), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)

plt.tight_layout()
plt.savefig('UTCI_seasonal_ave_check.pdf')
plt.show()
'''
# Plot four utci std plots in subplots (for check)

maxes = []
mins = []

levels = np.linspace(1,8,8)
norm = plt.Normalize(1,8)

season_std = 'season_1' 
season_std = np.nanstd(utci[0::4, :, :], axis = 0)
print np.nanmax(season_std), np.nanmin(season_std)
maxes.append(np.max(season_std))
mins.append(np.min(season_std))

fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(221)
plt.gcf().subplots_adjust(right = 0.8)
ax.set_title('std of UTCI in DJF (check)')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_std, cmap = 'rainbow',norm=norm,levels=levels,extend='max')
#cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 15), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)



season_std = 'season_2' 
season_std = np.nanstd(utci[1::4, :, :], axis = 0)
print np.nanmax(season_std), np.nanmin(season_std)
maxes.append(np.max(season_std))
mins.append(np.min(season_std))

ax = fig.add_subplot(222)
ax.set_title('std of UTCI in MAM')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_std, cmap = 'rainbow',norm=norm,levels=levels,extend='max')
#cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 15), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)


season_std = 'season_3' 
season_std = np.nanstd(utci[2::4, :, :], axis = 0)
print np.nanmax(season_std), np.nanmin(season_std)
maxes.append(np.max(season_std))
mins.append(np.min(season_std))

ax = fig.add_subplot(223)
ax.set_title('std of UTCI in JJA')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
m.contourf(y, x, season_std, cmap = 'rainbow',norm=norm,levels=levels,extend='max')
#cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 15), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)


season_std = 'season_4' 
season_std = np.nanstd(utci[3::4, :, :], axis = 0)
print np.nanmax(season_std), np.nanmin(season_std)
maxes.append(np.max(season_std))
mins.append(np.min(season_std))

ax = fig.add_subplot(224)
ax.set_title('std of UTCI in SON')
m = Basemap(projection = 'cyl',llcrnrlon = -30 , urcrnrlon = 45,
            llcrnrlat = 35, urcrnrlat = 75)
m.drawcountries()
m.drawcoastlines(color = 'black', linewidth = 0.5)
cs = m.contourf(y, x, season_std, cmap = 'rainbow',norm=norm,levels=levels,extend='max')
#cbar = m.colorbar()
m.drawmeridians(np.arange(-30, 48, 10), labels = [1,0,0,1], linewidth = 0.6)
m.drawparallels(np.arange(30, 75, 10), labels = [1,0,0,0], linewidth = 0.6)



stdmax = np.max(maxes)
stdmin = np.min(mins)
#plt.tight_layout()


cbar_ax = plt.gcf().add_axes([0.85, 0.15, 0.03, 0.7])
#plt.clim(vmin = 1, vmax = 13)
plt.colorbar(cs, cax = cbar_ax)
#plt.savefig('UTCI_seasonal_std_check.pdf')
plt.show()
