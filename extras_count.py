# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:55:24 2018

@author: np838619
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = '/home/users/qx911590/'

total = 0
err_tot = 0
dat_tot = 0
names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book
years = pl.linspace(1900,1910,11).astype(int)
for Y in range(years.size):
    #homedir = 'M:\\weatherrescue\\csv_fixed'
    #filepath = os.path.join(homedir,'1901\\*')
    filenames1 = glob.glob(homedir+'weatherrescue/csvfiles/'+str(years[Y])+'/*') # all filenames in dir
    filenames1 = pl.asarray(filenames1) # list into array
    filenames1 = pl.sort(filenames1) # get filenames in correct order
    
    filenames2 = glob.glob(homedir+'weatherrescue/csv_fixed/'+str(years[Y])+'/*') # all filenames in dir
    filenames2 = pl.asarray(filenames2) # list into array
    filenames2 = pl.sort(filenames2) # get filenames in correct order
    
    
    count = 0
    err_ct = 0
    dat_ct = 0
    
    for name in range(filenames1.size): # loop over filenames
        df1 = pd.read_csv(filenames1[name],header=None,names=names)
        df2 = pd.read_csv(filenames2[name],header=None,names=names)
        # open as pandas dataframe because numpy/genfromtxt don't like '#N/A'
    
        logs1 = pl.array(df1) # dataframe into array
        logs2 = pl.array(df2)
        #logs = pl.genfromtxt(filenames[name],delimiter=',')
        
        data = pl.where(logs2[1:,[5]]!='-99999')
        dat_ct = dat_ct + data[0].size
    
        errs = pl.where(logs1==' #N/A') # where do the errors occur?
        #prs = pl.where(logs[:,1]>31)
        #fls = pl.where(logs=='-999')
        
    #    if errs[0].size > 0:
    #        print filenames[name][54:-4]
        
        #if prs[0].size > 0:
         #   print filenames[name][54:-4], logs[prs[0][:]]
        
        if errs[0].size > 0:
            err_ct = err_ct + errs[0].size
            for i in range(errs[0].size):
                if logs2[errs[0][i],errs[1][i]] == '-99999':
                    count = count + 1
            #count = count + errs[0].size
            #print name,filenames[name][63:-4], logs[errs[0][0]]
    
        #if logs[10,0] == 'Blonduos ':
        #    print filenames[name]
    
        # open a file to write the errors to:
        #f = open(clusdir+'weatherrescue/spreadsheet_errs/errs_'+str(filenames[name][63:73])+'.txt','w')
        #for i in range(errs[0].size): # loop over no. of errors
            #print logs[errs[0][i],0], names[errs[1][i]]
            # write the station name & variable to file:
        #    f.write(logs[errs[0][i],0]+' '+names[errs[1][i]]+'\n')
        #f.close() # close file
        
    #    g = open(clusdir+'weatherrescue/spreadsheet_errs/false_'+str(filenames[name][63:73])+'.txt','w')
    #    for i in range(fls[0].size):
    #        g.write(logs[fls[0][i],0]+' '+names[fls[1][i]]+'\n')
    #    g.close()
    
    total = total + count
    err_tot = err_tot + err_ct
    dat_tot = dat_tot + dat_ct

print total
print err_tot

df3 = pd.read_csv(homedir+'weatherrescue/csv_fixed/1900_2pm.csv')
logs3 = pl.array(df3)
data2pm = pl.where(logs3[:,3::3]!='-99999')

print dat_tot + data2pm[0].size