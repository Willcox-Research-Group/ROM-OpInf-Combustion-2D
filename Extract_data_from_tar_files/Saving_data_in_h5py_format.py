import numpy as np
import h5py
import os


workingdir = 'C:\\Users\\Admin\\Desktop\\combustion'
datafile = os.path.join(workingdir, "gems60kfinalfile.h5")

os.chdir(workingdir)
data = np.load('data60k_final.npy')

t = np.arange(0.015,0.021,10**-7)   # np.linspace(0.015,0.017,30000)
t = t[0:int(np.shape(data)[1])]

with h5py.File(datafile, 'a') as hf:
    hf["data"] = data
    hf["time"] = t
