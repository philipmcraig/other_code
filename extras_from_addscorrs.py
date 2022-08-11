# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:17:03 2019

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = '/home/users/qx911590/'
wrdir = homedir + 'weatherrescue/'
years = pl.linspace(1900,1910,11).astype(int)

total = 0

for Y in range(years.size):
    addfiles = glob.glob(wrdir+'additions_corrections/'+str(years[Y])+'/newadds*')
    addfiles = pl.asarray(addfiles)
    addfiles = pl.sort(addfiles)
    
    corfiles = glob.glob(wrdir+'additions_corrections/'+str(years[Y])+'/newcorrs*')
    corfiles = pl.asarray(corfiles)
    corfiles = pl.sort(corfiles)
    
    count1 = 0
    
    for name in range(addfiles.size):
        df1 = pd.read_csv(addfiles[name],header=None)
        df2 = pd.read_csv(corfiles[name],header=None)
        
        logs1 = pl.array(df1)
        logs2 = pl.array(df2)
        
        vals1 = pl.where((logs1[:,2:]!='-999') & (logs1[:,2:]!='') &\
                        (logs1[:,2:]!=' ') & (logs1[:,2:]!='  '))
        
        if vals1[0].size > 0:
            count1 = count1 + vals1[0].size
        
        
    total = total + count1
    
print total