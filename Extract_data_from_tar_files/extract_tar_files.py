# extract_tar_files.py
import os
import tarfile

basefolder = r'C:\\Users\\Admin\\Desktop\\storage\\combustion\\'

# change the current directory to specified directory
folder_name_timesnapshots = '200000 to 209999'
os.chdir(os.path.join(basefolder, folder_name_timesnapshots))

data_to_extract = 'Data_200000to209999.tar'
tar = tarfile.open(os.path.join(basefolder, data_to_extract))
tar.extractall()
tar.close()
