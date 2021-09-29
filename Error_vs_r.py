# Error_vs_r.py
"""Save the data of traces_rom with a for loop between step 3 and step 4
to get the traces_rom at differnt r, for instance the saved data for r= 30
is 'traces_rom_cubic_r30'.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of snapshots used as training data.
trainsize = 20002

# Regularization parameters for Operator Inference.
regs = 9.248289e+03, 2.371337e+05, 5.048738e+06
# 10**4,10**5,10**6    # 100, 3*10**6, 10**7

_MAXFUN = 100               # Artificial ceiling for optimization routine.


Worked = [17,20,22,23,25,36,38,42,43,44]

FOM_data = np.load('traces_gems_used_in_for_loop.npy')

T_st_series = [12,13,14,15]


def get_data(Worked, FOM_data, T_st_series,
             datanumber=0, last_training_data=20000,
             ylabl='T Error', savefilename='naamdallo'):
    Y2 = []
    for r in Worked:
        val_FOM = []
        val_ROM = []

        for T_st in T_st_series:
            ROM_data = np.load('traces_rom_cubic_r' + str(r)+'.npy')
            T_ROM = pd.DataFrame(ROM_data).loc[T_st + datanumber][last_training_data]
            val_ROM.append(T_ROM)
            T_FOM = pd.DataFrame(FOM_data).loc[T_st + datanumber][last_training_data]
            val_FOM.append(T_FOM)

            avg_FOM = np.mean(val_FOM)
            avg_ROM = np.mean(val_ROM)

            error2 = abs(avg_FOM - avg_ROM)/abs(avg_FOM)

        Y2.append(error2)

    x_axis = np.arange(16,45,1)
    y_axis = np.ones(len(x_axis))*-10

    dict_ = dict(zip(Worked, Y2))

    for a,b in enumerate(x_axis):
        if b in Worked:
            y_axis[a] = dict_[b]

    plt.scatter(x_axis,y_axis, s=70)
    plt.xlabel('Basis Size', fontsize=30)
    plt.ylabel(ylabl, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=50)
    plt.locator_params(axis="x", nbins=7)
    plt.locator_params(axis="y", nbins=3)
    # plt.title("L2 norm Error Plot")
    plt.legend()
    plt.ylim(0,np.max(Y2)+0.1)
    plt.savefig(savefilename + ".pdf", bbox_inches="tight", dpi=600)
    plt.show()


def get_data1(Worked, FOM_data, T_st_series,
              datanumber=0, last_training_data=20000,
              ylabl='T Error', savefilename='naamdallo'):
    Y2 = []
    for r in Worked:
        val_FOM = []
        val_ROM = []

        for T_st in T_st_series:
            ROM_data = np.load('traces_rom_cubic_r' + str(r)+'.npy')
            T_ROM = pd.DataFrame(ROM_data).loc[T_st + datanumber][last_training_data]
            val_ROM.append(T_ROM)
            T_FOM = pd.DataFrame(FOM_data).loc[T_st + datanumber][last_training_data]
            val_FOM.append(T_FOM)

            avg_FOM = np.mean(val_FOM)
            avg_ROM = np.mean(val_ROM)

            error2 = abs(avg_FOM - avg_ROM)/max(abs(np.array(val_FOM)))

        Y2.append(error2)

    x_axis = np.arange(16,45,1)
    y_axis = np.ones(len(x_axis))*-10

    dict_ = dict(zip(Worked, Y2))

    for a,b in enumerate(x_axis):
        if b in Worked:
            y_axis[a] = dict_[b]

    plt.scatter(x_axis,y_axis, s=70)
    plt.xlabel('Basis Size', fontsize=30)
    plt.ylabel(ylabl, fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=50)
    plt.locator_params(axis="x", nbins=7)
    plt.locator_params(axis="y", nbins=3)
    # plt.title("L2 norm Error Plot")
    plt.legend()
    plt.ylim(0,np.max(Y2)+0.1)
    plt.savefig(savefilename + ".pdf", bbox_inches="tight", dpi=600)
    plt.show()


if __name__ == "__main__":

    d_number = [0, -12, 8, 12, 16, 20, -12 + 4, -12 + 8]
    yl = ['T Error', 'P Error',
          '[CH4] Error', '[O2] Error', '[H2O] Error', '[CO2] Error',
          'Vx Error', 'Vy Error']
    savename = ['T_Error', 'P_Error',
                'CH4_Error', 'O2_Error', 'H2O_Error', 'CO2_Error',
                'Vx_Error','Vy_Error']

    for i in np.arange(0, len(d_number), 1):
        if i == 0 or i == 1:
            get_data(Worked, FOM_data, T_st_series,
                     datanumber=d_number[i], last_training_data=19999,
                     ylabl=yl[i], savefilename=savename[i])
            # get_data(Worked, FOM_data, T_st,
            #          datanumber=d_number[i], ylabl=yl[i])
        else:
            get_data1(Worked, FOM_data, T_st_series,
                      datanumber=d_number[i], last_training_data=19999,
                      ylabl=yl[i],savefilename=savename[i])
