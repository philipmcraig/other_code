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
filenames = glob.glob(resdir+'additions_corrections/'+year+'/corrections*')
filenames = pl.asarray(filenames) # list into array
filenames = pl.sort(filenames) # get filenames in correct order
#filenames = filenames[[2,-1]]

for name in range(filenames.size):
    corrs = pd.read_csv(filenames[name],na_filter=False,header=None)
    
    C=pl.asarray(corrs)
    
    qm = pl.where(C=='?')
    if qm[0].size > 0.:
        for i in range(qm[0].size):
            C[qm[0][i],qm[1][i]] = '-999'
    
    star = pl.where(C=='*')
    if star[0].size > 0.:
        for i in range(star[0].size):
            C[star[0][i],star[1][i]] = ''
    
    hsh = pl.where(C=='#')
    if hsh[0].size > 0:
        for i in range(hsh[0].size):
            C[hsh[0][i],hsh[1][i]] = '-999'
    
    dash = pl.where(C=='-')
    if dash[0].size > 0.:
        for i in range(dash[0].size):
            C[dash[0][i],dash[1][i]] =''
    
    #space = pl.where(C=='')
    #if space[0].size > 0:
    #    for i in range(space[0].size):
    #        C[space[0][i],space[1][i]] = 
    
    D = C[:,2:]
    D[:,0] = [str(i) for i in D[:,0]]
    D[:,1] = [str(i) for i in D[:,1]]
    D[:,2] = [str(i) for i in D[:,2]]
    D[:,3] = [str(i) for i in D[:,3]]
    D[:,4] = [str(i) for i in D[:,4]]
    D[:,5] = [str(i) for i in D[:,5]]
    D[:,6] = [str(i) for i in D[:,6]]
    D[:,7] = [str(i) for i in D[:,7]]
    
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i][j] == '':
                pass
            elif D[i][j][0] == '?':
                D[i,j] = '-999'
                #print i,j
    
    C[:,2:] = D; del D
    
    x = []
    for i in range(C.shape[0]):
        if all(C[i][2:]=='') == True:# and C[i][2] == ' ':
            x.append(i)
            #print i
    
    C = pl.delete(C,x,axis=0)
    
    newcorrs = pd.DataFrame(C)
#    newcorrs.to_csv(resdir+'additions_corrections/'+year+'/newcorrs'+filenames[name][73:-4]+'.csv',
#                   header=None,index=None)