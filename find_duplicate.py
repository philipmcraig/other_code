#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:26:52 2018

@author: qx911590
"""

from __future__ import division
import pylab as pl
import pandas as pd
import glob

resdir = '/home/users/qx911590/weatherrescue/'

filenames = glob.glob(resdir+'csv_fixed/1903/*')
filenames = pl.asarray(filenames)
filenames = pl.sort(filenames)

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book


for name in range(1,30):
    dwr1 = df = pd.read_csv(filenames[name-1],header=None,names=names)
    dwr2 = df = pd.read_csv(filenames[name],header=None,names=names)
    
    logs1 = pl.array(dwr1); logs2 = pl.array(dwr2)
    
    for i in range(9,34):
        if logs1[i,-1] == logs2[i,-1]:
            print filenames[name-1][59:-4], filenames[name][59:-4], logs1[i][0]
    
    #for i in range(logs1.shape[0]):
    #    for j in range(1,logs1.shape[1]):
    #        if logs1[i,j] == '':
     #           pass
      #      elif logs1[i,j] == ' ':
       #         pass
        #    elif logs1[i,j] == logs2[i,j]:
       #         print filenames[name-1][59:-4], filenames[name][59:-4], logs1[i][0]