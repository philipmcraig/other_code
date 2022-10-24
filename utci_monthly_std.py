from netCDF4 import Dataset
import pandas as pd
import numpy as np
#import matplotlib as plt


path = '/home/users/qx911590/INDECIS/ncfiles/utci_month.nc'
#file = Dataset(path)

lat, lon = file.variables['latitude'], file.variables['longitude']
time, utci = file.variables['time'], file.variables['utci']
ncfile = Dataset(path)
utci = ncfile.variables['utci'][:]
lat = ncfile.variables['latitude'][:]
lon = ncfile.variables['longitude'][:]
time = ncfile.variables['time'][:]
ncfile.close()

x,y = np.meshgrid(lat, lon)
time_label = np.arange(1950, 2018, 1)

#utci_std = np.zeros([len(lon), len(lat)])
utci_ave = np.zeros([len(lon), len(lat)])


#for i in range(len(lon)):
#    for j in range(len(lat)):
#        utci_std[i,j] = np.std(utci[i,j,:])
    #print ('over', i)
utci_std = np.nanstd(utci,axis=0)
# Write the standard deviation data into a csv

data_column = []
for i in range (len(lat)):
    data_column.append('lat' + str(i))
print ('data column done')
 
   
data_row = []
for i in range(len(lon)):
    data_row.append('lon' + str(i))
print ('data row done')


data_frame = pd.DataFrame('position', columns = data_column, index = data_row )
data_frame.index.name = 'position'

for i in range(len(lon)):
    for j in range(len(lat)):
        data_frame.iloc[i, j] = utci_std[i, j]
    print ('second', i)

output = str('utci_monthly_std.csv')
data_frame.to_csv(output)
print ('done')
