# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:40:32 2018

@author: np838619
"""

import pylab as pl
import pandas as pd
import glob

resdir = '/home/users/qx911590/weatherrescue/'

year = '1910'
filenames = glob.glob(resdir+'additions_corrections/'+year+'/additions*')
filenames = pl.asarray(filenames) # list into array
filenames = pl.sort(filenames) # get filenames in correct order
#filenames = filenames[[2,-1]]

for name in range(filenames.size):
    adds = pd.read_csv(filenames[name],na_filter=False,header=None)
    
    A=pl.asarray(adds)
    
    qm = pl.where(A=='?')
    if qm[0].size > 0.:
        for i in range(qm[0].size):
            A[qm[0][i],qm[1][i]] = ' '
    
    star = pl.where(A[:,:-1]=='*')
    if star[0].size > 0.:
        for i in range(star[0].size):
            A[star[0][i],star[1][i]] = ' '
    
    star2 = pl.where(A[:,-1]=='*')
    if star2[0].size > 0:
        for i in range(star2[0].size):
            A[star2[0][i],-1] = '0'
    
    hsh = pl.where(A=='#')
    if hsh[0].size > 0:
        for i in range(hsh[0].size):
            A[hsh[0][i],hsh[1][i]] = ' '
    
    dash = pl.where(A[:,:-1]=='-')
    if dash[0].size > 0.:
        for i in range(dash[0].size):
            A[dash[0][i],dash[1][i]] = ' '
    
    dash2 = pl.where(A[:,-1]=='-')
    if dash2[0].size > 0:
        for i in range(dash2[0].size):
            A[dash2[0][i],-1] = '0'
    
    B = A[:,2:]
    B[:,0] = [str(i) for i in B[:,0]]
    B[:,1] = [str(i) for i in B[:,1]]
    B[:,2] = [str(i) for i in B[:,2]]
    B[:,3] = [str(i) for i in B[:,3]]
    B[:,4] = [str(i) for i in B[:,4]]
    B[:,5] = [str(i) for i in B[:,5]]
    B[:,6] = [str(i) for i in B[:,6]]
    B[:,7] = [str(i) for i in B[:,7]]
    
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if len(B[i][j]) > 0 and B[i][j][0] == '?':
                B[i,j] = ' '
            elif len(B[i][j]) > 0 and B[i][j][-1] == '?':
                B[i,j] = ' '
    
    A[:,2:] = B; del B
    
    newadds = pd.DataFrame(A)
#    newadds.to_csv(resdir+'additions_corrections/'+year+'/newadds'+filenames[name][71:-4]+'.csv',
#                   header=None,index=None)