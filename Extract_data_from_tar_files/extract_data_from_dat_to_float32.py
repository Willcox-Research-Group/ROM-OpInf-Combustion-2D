import itertools
import numpy as np


gems_data = np.zeros((308184,10000), dtype='float32')


def get_data(timestep, gems_data_col_no):

    with open('C:\\Users\\Admin\\Desktop\\storage\\combustion\\200000 to 209999\\test_file_ncons_'+str(timestep)+'.dat') as infile:
        lst = [line.strip.split() for line in infile]

    Line_data = list(itertools.chain.from_iterable(lst))
    kkk = np.array([float(i) for i in Line_data[32::]])

    start = 0
    check = 308184

    gems_data[:,gems_data_col_no] = kkk[start:check]
    return gems_data


for i in np.arange(200000, 209999, 1):
    print(i)
    data2 = get_data(i, i - 200000)

np.save('gems_extra_data20.npy', data2)

data2 = np.load('gems_extra_data20.npy')
data1 = np.load('gems50k.npy')

a2 = np.concatenate((data1, data2), axis=1)

np.save('gems60k.npy',a2)
