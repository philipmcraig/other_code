# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 14:17:12 2018

@author: Philip
"""

import pylab as pl
from openpyxl import load_workbook
import pandas as pd
import os

year = '1910'
#path = 'C:\Users\Philip\OneDrive\Documents\Reading\\additions_corrections\\1900'
path = '/home/users/qx911590/weatherrescue/additions_corrections/' + year

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
#months = ['09']

for m in range(len(months)):
    filepath = os.path.join(path,'WeatherRescue_'+year+'_'+months[m]+'.xlsx')
    wb = load_workbook(filepath)
    #print wb.get_sheet_names()
    #df = pd.read_excel(filepath)
    
    additions = pd.DataFrame(wb['Additions'].values)
    corrections = pd.DataFrame(wb['Corrections'].values)
    
    adds = pl.asarray(additions)
    cors = pl.asarray(corrections)
    
    cols = [0,1,2,3,7,9,10,16,17,20] # columns to be removed
    
    A = adds[3:,cols]#; A[:,0] = A[:,0].day
    for i in range(A.shape[0]):
        if A[i,0] != None:
            A[i,0] = A[i,0].day
    A = pd.DataFrame(A)
    
    C = cors[3:,cols]#; C[:,0] = C[:,0].day
    for i in range(C.shape[0]):
        if C[i,0] != None:
            C[i,0] = C[i,0].day
    C = pd.DataFrame(C)
    
    newpath = os.path.join(path,'additions_'+filepath[76:-5]+'.csv')
    #A.to_csv(newpath,header=None,index=None)
    
    newpath = os.path.join(path,'corrections_'+filepath[76:-5]+'.csv')
    #C.to_csv(newpath,header=None,index=None)