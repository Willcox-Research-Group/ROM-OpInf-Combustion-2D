# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:28:51 2021

@author: Admin
"""


import os
import re
import glob
import h5py
import shutil
import logging
import tarfile
import numpy as np
import multiprocessing as mp
# change the current directory 
# to specified directory 
folder_name_timesnapshots = '200000 to 209999'
os.chdir(r'C:\\Users\\Admin\\Desktop\\storage\\combustion\\'+folder_name_timesnapshots) 

data_to_extract = 'Data_200000to209999.tar'
import tarfile
tar = tarfile.open("C:\\Users\\Admin\\Desktop\\storage\\combustion\\"+data_to_extract)
tar.extractall()