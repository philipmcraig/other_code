# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 15:43:08 2018

@author: np838619
"""

import pylab as pl
import glob
import os
import pandas as pd

resdir = '/home/users/qx911590/weatherrescue/'

dwrfiles = glob.glob(resdir+'temp/dwr*')
dwrfiles = pl.asarray(dwrfiles)
dwrfiles = pl.sort(dwrfiles)

#addfiles = glob.glob(clusdir+'additions_corrections/1900/newadd*') # all filenames in dir
#addfiles = pl.asarray(addfiles) # list into array
#addfiles = pl.sort(addfiles) # get filenames in correct order
#
#
#corfiles = glob.glob(clusdir+'additions_corrections/1900/newcor*') # all filenames in dir
#corfiles = pl.asarray(corfiles) # list into array
#corfiles = pl.sort(corfiles) # get filenames in correct order

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book

for name in range(dwrfiles.size):
    #A = pl.genfromtxt(filenames[name],delimiter=',',dtype=None)
    #dwr = pd.read_csv(clusdir+'test/dwr_1900_01_01.csv',na_filter=True,header=None)
    dwr = pd.read_csv(dwrfiles[name],na_filter=True,header=None)
    #dwr = dwr.replace('  ',pl.NaN)
    dwr  = pl.asarray(dwr)
    
    year = dwrfiles[name][-14:-10]
    month = dwrfiles[name][-9:-7]
    day = int(dwrfiles[name][-6:-4])
    
    #if month not in ('02','05'): 
    #if month == '11' and day == 27:
    #    continue
    #else:
    add = pd.read_csv(resdir+'additions_corrections/'+year+'/newadds_'+year+'_'+month+'.csv',
                      na_filter=False,header=None)
    add = add.replace(' ',pl.NaN)
    add = pl.asarray(add)

    adate = pl.where(add[:,0]==day); adate = adate[0]
    
    for i in range(adate.size):
        date = adate[i]
        stat = add[date,1] + ' '
        st_rw = pl.where(dwr==stat); st_rw = st_rw[0][0]
        #print st_rw
        
        n = pl.where(add[date,2:]=='')
        add[date,2:][n] = pl.float32('nan')
        x = pl.isnan(add[adate[i],2:].astype(float))#; x = x[0]
        for j in range(x.size):
            if x[j] == False:
                dwr[st_rw,j+1] = add[adate[i],2:][j]# insert to dwr array
            
        #del add
    
    cor = pd.read_csv(resdir+'additions_corrections/'+year+'/newcorrs_'+year+'_'+month+'.csv',
                      na_filter=False,header=None)
    cor = cor.replace('',pl.NaN)
    cor = pl.asarray(cor)
    
    cdate = pl.where(cor[:,0]==day); cdate = cdate[0]
    
    for i in range(cdate.size):
        date = cdate[i]
        stat = cor[date,1] + ' ' 
        st_rw = pl.where(dwr==stat); st_rw = st_rw[0][0]
        #print st_rw
        
        y = pl.where(cor[i,2:].astype(float)==-999)
        if y[0].size > 0.:
            dwr[st_rw,y[0]+1] = ' '
        x = pl.isnan(cor[cdate[i],2:].astype(float))
        for j in range(x.size):
            if x[j] == False and cor[cdate[i],2:][j] != '-999':
                dwr[st_rw,j+1] = cor[cdate[i],2:][j]
                
            #del cor
                    #if cor[1,2:][j] == '999':
                    #    dwr[st_rw,j+1] = ' '
                    # 
        
    dwr = pd.DataFrame(dwr)
    dwr = dwr.replace(pl.NaN,' ')
        
    day = "{0:0=2d}".format(day)
    dwr.to_csv(resdir+'/temp/dwr_'+year+'_'+month+'_'+day+'.csv',
               header=None,index=None)

#    if 'add' in globals():
#        del dwr, add, cor
#    else:
#        del dwr, cor

#print dwr
#for i in range(dwr.shape[0]):
#    if A[i][1] == 'Newton Reigny':
#        continue
#    date = A[i][0]
#    if date < 1:
#        continue
#    else:
#        mn = filenames[name][88:90]
#        dt = "{0:0=2d}".format(int(date))
#        df = pd.read_csv(clusdir+'csv_fixed/dwr_1903_'+mn+'_'+dt+'.csv',
#                                 header=None,names=names)
#        
#        logs = pl.array(df)
#        stat = A[i][1]+' ' # station name plus a space
#        st_rw = pl.where(logs==stat); st_rw = st_rw[0][0]
#        
#        # loop over entries in row:
#        for ent in range(2,9):
#            if A[i][ent] == '':
#                logs[st_rw][ent-1]= ' '
#            elif A[i][ent] == '*':
#                logs[st_rw][ent-1]= ' '
#            elif A[i][ent] == '?':
#                logs[st_rw][ent-1]= ' '
#            #elif A[i][ent].dtype == 'S6':
#            #    logs[st_rw][ent-1]= ' '
#            #elif A[i][ent].dtype == 'S3':
#            #    logs[st_rw][ent-1]= ' '
#            elif A[i][ent] == '-':
#                logs[st_rw][ent-1] = 0.
#            elif A[i][ent][0] == '?':
#                logs[st_rw][ent-1]= ' '
#            else:
#                logs[st_rw][ent-1] = A[i][ent]
#    
#        logs = pd.DataFrame(logs)
        #logs.to_csv(clusdir+'csv_fixed/dwr_1903_'+mn+'_'+dt+'.csv',
        #            header=None,index=None)

# if entry == '*': leave blank
# if entry[0] == '?': leave blank
# if entry == '-': zero