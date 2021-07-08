# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:31:02 2021

@author: Admin
"""

import numpy as np
import h5py
import os

os.chdir('C:\\Users\\Admin\\Desktop\\combustion')
file = 'data60k_final.npy'

data = np.load(file)

t = np.arange(0.015,0.021,10**-7)#np.linspace(0.015,0.017,30000)
t = t[0:int(np.shape(data)[1])]
                                 
                                 
with h5py.File('C:\\Users\\Admin\\Desktop\\storage\\combustion\\gems60kfinalfile.h5', 'a') as hf:
    hf["data"] = data
    hf["time"] = t