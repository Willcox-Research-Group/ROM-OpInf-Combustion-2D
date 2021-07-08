# Spectral_Analysis_Amp_and_Phase.py
import os
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt


# Import time from the data or define it
t = np.arange(0.015, 0.021, 10**-7)
dt = 10**-7


# Define trainsize and number of modes
trainsize = 20000    # Number of snapshots used as training data.
num_modes = 44       # Number of POD modes.
reg = 0              # Just an input in case we regularize DMDc.


# Locate the full data of snapshots FOM and ROMs (INPUT)
Folder_name_data = 'C:\\Users\\Admin\\Desktop\\combustion\\'
file_name_FOM = 'traces_gems_60k_final.npy'
file_name_ROM_DMDc = 'traces_rom_DMDc_rsvd.npy'
file_name_ROM_cubic_r25 = 'traces_rom_cubic_tripple_reg_r25.npy'
file_name_ROM_cubic_r44 = 'traces_rom_cubic_r44.npy'
file_name_ROM_Quad_r44 = 'traces_rom_60k_100_30000.npy'


# Define output file location and file names to identify phase and amplitudes (OUTPUT)
folder_name = "C:\\Users\\Admin\\Desktop\\combustion\\spectral\\Final_plots\\"
Amp_name = folder_name + "\\" + "Amp"       # Amplitude plots
Phase_name = folder_name + "\\" + "Phase"   # Phase plots


# Load the data
FOM_ = np.load(Folder_name_data + file_name_FOM)
ROM_DMDc = np.load(Folder_name_data + file_name_ROM_DMDc)
ROM_cubic_r25 = np.load(Folder_name_data + file_name_ROM_cubic_r25)
ROM_cubic_r44 = np.load(Folder_name_data + file_name_ROM_cubic_r44)
ROM_Quad_r44 = np.load(Folder_name_data + file_name_ROM_Quad_r44)


# Plotting adjustments
End_plot_at = 60000  # 59990  # 40000
freq_limit_to_plot = 15000


# =============================================================================

def lineplots_timeseries(FOM_,
                         ROM_Quad_r44, ROM_cubic_r25, ROM_cubic_r44, ROM_DMDc,
                         datanumber, unit, savefile):
    """Plots for comparision of data in time. Check the saved data in
    folder_name.

    Parameters
    ----------
    FOM_
        Full order model data input
    ROM_Quad_r44
        Q-OPINF at r = 44
    ROM_cubic_r25
        C-OPINF at r = 25
    ROM_cubic_r44
        C-OPINF at r = 44
    ROM_DMDc
        DMDc results
    datanumber
        Defines the state parameter
        * -12 = Pressure
        * -8 = Vx
        * -4 = Vy
        *  0 = Temperature
        *  8 = [CH4]
        *  12 = [O2]
        *  16 = [H2O]
        *  20 = [CO2]
    unit
        Unit for each variable (Pa, Kelvin...)
    savefile
        Suffix to save the file name
    """
    print("Time series plots")
    plt.xlim([0.015, 0.021])  # set axis limits

    plt.plot(t[0:End_plot_at],
             pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at],
             label='FOM', linestyle='solid', c='k')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_Quad_r44).loc[T_st + datanumber][0:End_plot_at],
             label='Q-OPINF', linestyle='dashed', c='#ff7f0e')
    # plt.plot(t[0:End_plot_at],
    #        pd.DataFrame(ROM_cubic_r25).loc[T_st + datanumber][0:End_plot_at],
    #          label='C-OPINF_r25', linestyle='dashed')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_cubic_r44).loc[T_st + datanumber][0:End_plot_at],
             label='C-OPINF', linestyle='dashed', c='b')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_DMDc).loc[T_st + datanumber][0:End_plot_at],
             label='DMDc', linestyle='dashdot', c='r')
    plt.xlabel('time')
    plt.ylabel(unit)
    plt.axvline(x=t[0] + trainsize*dt, color='black')

    plt.legend()
    fname = f"{T_st}_ts_{trainsize}_r_{num_modes}_reg_{reg}{savefile}.pdf"
    plt.savefig(os.path.join(folder_name, fname),
                bbox_inches="tight", dpi=200)
    plt.show()


def L2errorplots(FOM_, ROM_Quad_r44, ROM_cubic_r25, ROM_cubic_r44, ROM_DMDc,
                 datanumber, unit):
    """Plot L2 norm error comparision between all the ROMs.

    Parameters
    ----------
    FOM_
        Full order model data input
    ROM_Quad_r44
        Q-OPINF at r = 44
    ROM_cubic_r25
        C-OPINF at r = 25
    ROM_cubic_r44
        C-OPINF at r = 44
    ROM_DMDc
        DMDc results
    datanumber
        Defines the state parameter
        * -12 = Pressure
        * -8 = Vx
        * -4 = Vy
        *  0 = Temperature
        *  8 = [CH4]
        *  12 = [O2]
        *  16 = [H2O]
        *  20 = [CO2]
    unit
        Unit for each variable (Pa, Kelvin...)
    """
    print("L2 norm error plot")
    e_ROM_Quad_r44 = (la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at] - pd.DataFrame(ROM_Quad_r44).loc[T_st + datanumber][0:End_plot_at]))/la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at])
    e_ROM_cubic_r25 = (la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at] - pd.DataFrame(ROM_cubic_r25).loc[T_st + datanumber][0:End_plot_at]))/la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at])
    e_ROM_cubic_r44 = (la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at] - pd.DataFrame(ROM_cubic_r44).loc[T_st + datanumber][0:End_plot_at]))/la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at])
    e_ROM_DMDc = (la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at] - pd.DataFrame(ROM_DMDc).loc[T_st + datanumber][0:End_plot_at]))/la.norm(pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at])

    plt.plot(t[0:End_plot_at],
             pd.DataFrame(FOM_).loc[T_st + datanumber][0:End_plot_at],
             label='FOM', linestyle='solid')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_Quad_r44).loc[T_st + datanumber][0:End_plot_at],
             label='Q-OPINF', linestyle='dashed')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_cubic_r25).loc[T_st + datanumber][0:End_plot_at],
             label='C-OPINF_r25', linestyle='dashed')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_cubic_r44).loc[T_st + datanumber][0:End_plot_at],
             label='C-OPINF_r44', linestyle='dashed')
    plt.plot(t[0:End_plot_at],
             pd.DataFrame(ROM_DMDc).loc[T_st + datanumber][0:End_plot_at],
             label='DMDc', linestyle='dashdot')

    x_axis = ['ROM_Quad_r44', 'ROM_cubic_r25', 'ROM_cubic_r44', 'ROM_DMDc']
    y_axis = [e_ROM_Quad_r44, e_ROM_cubic_r25, e_ROM_cubic_r44, e_ROM_DMDc]
    plt.scatter(x_axis,y_axis, s=100)
    plt.xlabel('time')
    plt.ylabel(unit)

    plt.title("L2 norm Error Plot")
    plt.legend()
    fnm = f"Error_plot_{T_st}_ts_{trainsize}_r_{num_modes}_reg_{reg}{unit}.pdf"
    plt.savefig(os.path.join(folder_name, fnm), bbox_inches="tight", dpi=200)
    plt.show()


def get_freq_and_amplitude(T_ROM):
    """
    Parameters
    ----------
    T_ROM = any input signal

    Returns
    -------
    frequency and amplitude transformation of the signal
    """
    fft1 = np.fft.fft(T_ROM)
    fftfreq1 = np.fft.fftfreq(len(T_ROM), d=dt)
    amplitude_DMD = abs(fft1)
    return fftfreq1, amplitude_DMD, fft1


def amplitude_plots(fftfreq,
                    fftfreq_Quad_r44, fftfreq_cubic_r25,
                    fftfreq_cubic_r44, fftfreq_DMDc,
                    amplitude,
                    amplitude_Quad_r44, amplitude_cubic_r25,
                    amplitude_cubic_r44, amplitude_DMDc,
                    unit, savefile,
                    title_test_or_train="Training results plotted in the frequency domain",
                    save_id="_ts_"):
    """Amplitude plot comparision and save files in the Amp name folder

    Eg. for the test data filename will be : Amp_test_12_ts_20000_r_44_reg_0CO2
    For the training data filename will be : Amp12_ts_20000_r_44_reg_0CO2

    Parameters
    ----------
    fftfreq
        frequency content of the FOM
    fftfreq_Quad_r44
        frequency content of the Q-OPINF at r = 44
    fftfreq_cubic_r25
        frequency content of the C-OPINF at r = 25
    fftfreq_cubic_r44
        frequency content of the C-OPINF at r = 44
    fftfreq_DMDc
        frequency content of the DMDc at r = 44
    amplitude
        Amplitude content of the FOM
    amplitude_Quad_r44
        Amplitude content of the Q-OPINF at r = 44
    amplitude_cubic_r25
        Amplitude content of the C-OPINF at r = 25
    amplitude_cubic_r44
        Amplitude content of the C-OPINF at r = 44
    amplitude_DMDc
        Amplitude content of the DMDc at r = 44
    unit
        unit for each variable (Pa, Kelvin...)
    savefile
        Filename to be saved
    title_test_or_train
        "Training results plotted in the frequency domain"
    save_id
        '_ts_' for traindata, '_test_' for testing data
    """
    st = 1
    end = 60
    plt.xlim([0,freq_limit_to_plot])
    plt.scatter(fftfreq[st:end], amplitude[st:end],
                s=50, label='FOM', marker='o', alpha=0.5, c='k')
    plt.scatter(fftfreq_Quad_r44[st:end], amplitude_Quad_r44[st:end],
                s=50, label='Q-OPINF', marker='s', alpha=0.5, c='#ff7f0e')
    # plt.scatter(fftfreq_cubic_r25[st:end], amplitude_cubic_r25[st:end],
    #             s=50, label='C-OPINF_r25', marker='p', alpha=0.5)
    plt.scatter(fftfreq_cubic_r44[st:end], amplitude_cubic_r44[st:end],
                s=50, label='C-OPINF', marker='*', alpha=0.5, c='b')
    plt.scatter(fftfreq_DMDc[st:end], amplitude_DMDc[st:end],
                s=50, label='DMDc', marker='+', alpha=0.5, c='r')

    plt.plot(fftfreq[st:end], amplitude[st:end],
             linestyle='solid', c='k')
    plt.plot(fftfreq_Quad_r44[st:end], amplitude_Quad_r44[st:end],
             linestyle='dashed', c='#ff7f0e')
    # plt.plot(fftfreq_cubic_r25[st:end], amplitude_cubic_r25[st:end],
    #          linestyle='dashed')
    plt.plot(fftfreq_cubic_r44[st:end], amplitude_cubic_r44[st:end],
             linestyle='dashed', c='b')
    plt.plot(fftfreq_DMDc[st:end], amplitude_DMDc[st:end],
             linestyle='dashdot', c='r')

    plt.xlabel('freq')
    plt.ylabel('Amplitude')
    plt.legend()
    # plt.title(title_test_or_train)

    if save_id == "_ts_":
        fname = f"{Amp_name}{T_st}{save_id}{trainsize}_r_{num_modes}"
        fname += f"_reg_{reg}{savefile}.pdf"
    elif save_id == "_test_":
        fname = f"{Amp_name}{save_id}{T_st}_ts_{trainsize}_r_{num_modes}"
        fname += f"_reg_{reg}{savefile}.pdf"
    else:
        raise ValueError(f"invalid save_id '{save_id}'")
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()


def get_min(X):
    """
    Parameters
    ----------
    X
        Phase angle array

    Returns
    -------
    min(X, 360-X)
    """
    b = abs(X)
    a = abs(360-b)
    return np.minimum(a,b)


def phase_plots(fftfreq,
                fftfreq_Quad_r44, fftfreq_cubic_r25,
                fftfreq_cubic_r44, fftfreq_DMDc,
                Phase_FOM,
                Phase_Quad_r44, Phase_cubic_r25,
                Phase_cubic_r44, Phase_DMDc,
                unit, savefile,
                title_test_or_train="Training results plotted in the frequency domain",
                save_id="_ts_"):
    """Phase plot comparision and save files in the Amp name folder.

    For the test data filename will be : Phase_test_12_ts_20000_r_44_reg_0CO2
    For the training data filename will be : Phase12_ts_20000_r_44_reg_0CO2

    Parameters
    ----------
    fftfreq
        frequency content of the FOM
    fftfreq_Quad_r44
        frequency content of the Q-OPINF at r = 44
    fftfreq_cubic_r25
        frequency content of the C-OPINF at r = 25
    fftfreq_cubic_r44
        frequency content of the C-OPINF at r = 44
    fftfreq_DMDc
        frequency content of the DMDc at r = 44
    Phase_FOM
        Phase content of the FOM
    Phase_Quad_r44
        Phase content of the Q-OPINF at r = 44
    Phase_cubic_r25
        Phase content of the C-OPINF at r = 25
    Phase_cubic_r44
        Phase content of the C-OPINF at r = 44
    Phase_DMDc
        Phase content of the DMDc at r = 44
    unit
        unit for each variable (Pa, Kelvin...)
    savefile
        Filename to be saved
    title_test_or_train
        "Training results plotted in the frequency domain"
    save_id
        '_ts_' for traindata, '_test_' for testing data
    """
    st = 1
    end = 60
    plt.xlim([0, freq_limit_to_plot])
    # plt.scatter(fftfreq[st:end], Phase_FOM[st:end],
    #             s=50, label='FOM', marker='o', alpha=0.5, c='k')
    plt.scatter(fftfreq_Quad_r44[st:end],
                get_min(Phase_FOM[st:end] - Phase_Quad_r44[st:end]),
                s=50, label='Q-OPINF', marker='s', alpha=0.5, c='#ff7f0e')
    # plt.scatter(fftfreq_cubic_r25[st:end], amplitude_cubic_r25[st:end],
    #             s=50, label='C-OPINF_r25', marker='p', alpha=0.5)
    plt.scatter(fftfreq_cubic_r44[st:end],
                get_min(Phase_FOM[st:end] - Phase_cubic_r44[st:end]),
                s=50, label='C-OPINF', marker='*', alpha=0.5, c='b')
    plt.scatter(fftfreq_DMDc[st:end],
                get_min(Phase_FOM[st:end] - Phase_DMDc[st:end]),
                s=50, label='DMDc', marker='+', alpha=0.5, c='r')

    # plt.plot(fftfreq[st:end],Phase_FOM[st:end], linestyle='solid', c='k')
    plt.plot(fftfreq_Quad_r44[st:end],
             get_min(Phase_FOM[st:end] - Phase_Quad_r44[st:end]),
             linestyle='dashed', c='#ff7f0e')
    # plt.plot(fftfreq_cubic_r25[st:end], amplitude_cubic_r25[st:end],
    #          linestyle='dashed')
    plt.plot(fftfreq_cubic_r44[st:end],
             get_min(Phase_FOM[st:end] - Phase_cubic_r44[st:end]),
             linestyle='dashed', c='b')
    plt.plot(fftfreq_DMDc[st:end],
             get_min(Phase_FOM[st:end] - Phase_DMDc[st:end]),
             linestyle='dashdot', c='r')

    plt.xlabel('freq')
    plt.ylabel('Phase angle difference FOM-ROM (degree)')
    plt.legend()
    # plt.title(title_test_or_train)

    if save_id == "_ts_":
        fname = f"{Phase_name}{T_st}{save_id}{trainsize}_r_{num_modes}"
        fname += f"_reg_{reg}{savefile}.pdf"
    if save_id == "_test_":
        fname = f"{Phase_name}{save_id}{T_st}_ts_{trainsize}_r_{num_modes}"
        fname += f"_reg_{reg}{savefile}.pdf"
    else:
        raise ValueError(f"invalid save_id '{save_id}'")
    plt.savefig(fname, bbox_inches="tight", dpi=200)
    plt.show()


def fftoutput_train(T_st, t, trainsize, num_modes, reg,
                    unit='Temperature in Kelvin', datanumber=0,
                    savefile='filename'):
    """Amplitude and phase plots for training dataset.

    Parameters
    ----------
    T_st
        monitor location code
        * 12: Monitor location 1
        * 13: Monitor location 2
        * 14: Monitor location 3
        * 15: Monitor location 4
    t
        as defined in input
    trainsize
        as defined in input
    num_modes
        as defined in input
    reg
        as defined in input
    unit
        unit for each variable (Pa, Kelvin...)
    datanumber
        defines the state parameter
        * -12: Pressure
        * -8: Vx
        * -4: Vy
        *  0: Temperature
        *  8: [CH4]
        * 12: [O2]
        * 16: [H2O]
        * 20: [CO2]
    savefile
        Suffix to save the file name
    """
    # fmax = 1/dt
    ROM_S = trainsize  # 20000
    FOM_S = trainsize  # 20000
    T = pd.DataFrame(FOM_).loc[13][0:FOM_S]
    # T_ROM = pd.DataFrame(ROM_DMDc).loc[13][0:ROM_S]
    # df = 1/dt/trainsize
    # fdomain = np.arange(0,fmax,df)

    T = pd.DataFrame(FOM_).loc[T_st + datanumber][0:FOM_S]
    T_ROM_Quad_r44 = pd.DataFrame(ROM_Quad_r44).loc[T_st + datanumber][0:ROM_S]
    T_ROM_DMDc = pd.DataFrame(ROM_DMDc).loc[T_st + datanumber][0:ROM_S]
    T_ROM_cubic_r25 = pd.DataFrame(ROM_cubic_r25).loc[T_st + datanumber][0:ROM_S]
    T_ROM_cubic_r44 = pd.DataFrame(ROM_cubic_r44).loc[T_st + datanumber][0:ROM_S]

    lineplots_timeseries(FOM_,
                         ROM_Quad_r44, ROM_cubic_r25, ROM_cubic_r44, ROM_DMDc,
                         datanumber,unit,savefile)
    # L2errorplots(FOM_, ROM_Quad_r44, ROM_cubic_r25, ROM_cubic_r44, ROM_DMDc,
    #              datanumber, unit)

    # fftfreq1, amplitude_DMD, fft1 = get_freq_and_amplitude(T_ROM_DMD)
    fftfreq_DMDc, amplitude_DMDc, fft_DMDc = get_freq_and_amplitude(T_ROM_DMDc)
    fftfreq_Quad_r44, amplitude_Quad_r44, fft_Quad_r44 = get_freq_and_amplitude(T_ROM_Quad_r44)
    fftfreq_cubic_r25, amplitude_cubic_r25, fft_cubic_r25 = get_freq_and_amplitude(T_ROM_cubic_r25)
    fftfreq_cubic_r44, amplitude_cubic_r44, fft_cubic_r44 = get_freq_and_amplitude(T_ROM_cubic_r44)
    fftfreq, amplitude, fft = get_freq_and_amplitude(T)

    amplitude_plots(fftfreq,
                    fftfreq_Quad_r44, fftfreq_cubic_r25,
                    fftfreq_cubic_r44, fftfreq_DMDc,
                    amplitude,
                    amplitude_Quad_r44, amplitude_cubic_r25,
                    amplitude_cubic_r44, amplitude_DMDc,
                    unit, savefile,
                    title_test_or_train="Training results plotted in the frequency domain",
                    save_id="_ts_")

    Phase_FOM = np.angle(fft, deg=True)
    Phase_Quad_r44 = np.angle(fft_Quad_r44, deg=True)
    Phase_cubic_r25 = np.angle(fft_cubic_r25, deg=True)
    Phase_cubic_r44 = np.angle(fft_cubic_r44, deg=True)
    Phase_DMDc = np.angle(fft_DMDc, deg=True)

    phase_plots(fftfreq,
                fftfreq_Quad_r44, fftfreq_cubic_r25,
                fftfreq_cubic_r44, fftfreq_DMDc,
                Phase_FOM,
                Phase_Quad_r44, Phase_cubic_r25,
                Phase_cubic_r44, Phase_DMDc,
                unit, savefile,
                title_test_or_train="Training results plotted in the frequency domain",
                save_id="_ts_")


def fftoutput_test(T_st, t, trainsize, num_modes, reg,
                   unit='Temperature in Kelvin',
                   datanumber=0, savefile='filename'):
    """
    T_st = monitor location code
    code number for each location:
    12 - Monitor location 1
    13 - Monitor location 2
    14 - Monitor location 3
    15 - Monitor location 4

    t = as defined in input
    trainsize = as defined in input
    num_modes = as defined in input
    reg = as defined in input
    unit = unit for each variable (Pa, Kelvin...)

    datanumber = to define the state parameter
    -12 = Pressure
    -8 = Vx
    -4 = Vy
     0 = Temperature
     8 = [CH4]
     12 = [O2]
     16 = [H2O]
     20 = [CO2]


    savefile = Suffix to save the file name

    Returns
    -------
    The calculation of amplitude and phase plots for testing dataset
    """
    # fmax = 1/dt
    # ROM_S = len(t[0:End_plot_at]) - trainsize
    FOM_S = len(t[0:End_plot_at]) - trainsize
    T = pd.DataFrame(FOM_).loc[13][FOM_S::]
    # T_ROM = pd.DataFrame(ROM_DMDc).loc[13][ROM_S::]
    # df = 1/dt/(len(t[0:End_plot_at]) - trainsize)
    # fdomain = np.arange(0,fmax,df)

    T = pd.DataFrame(FOM_).loc[T_st + datanumber][trainsize:len(t[0:End_plot_at])]
    # T_ROM_DMD = pd.DataFrame(ROM_DMDc).loc[T_st + datanumber][trainsize:len(t[0:End_plot_at])]
    T_ROM_DMDc = pd.DataFrame(ROM_DMDc).loc[T_st + datanumber][trainsize:len(t[0:End_plot_at])]
    T_ROM_Quad_r44 = pd.DataFrame(ROM_Quad_r44).loc[T_st + datanumber]
    T_ROM_cubic_r25 = pd.DataFrame(ROM_cubic_r25).loc[T_st + datanumber][trainsize:len(t[0:End_plot_at])]
    T_ROM_cubic_r44 = pd.DataFrame(ROM_cubic_r44).loc[T_st + datanumber][trainsize:len(t[0:End_plot_at])]

    fftfreq_DMDc, amplitude_DMDc, fft_DMDc = get_freq_and_amplitude(T_ROM_DMDc)
    fftfreq_Quad_r44, amplitude_Quad_r44, fft_Quad_r44 = get_freq_and_amplitude(T_ROM_Quad_r44)
    fftfreq_cubic_r25, amplitude_cubic_r25, fft_cubic_r25 = get_freq_and_amplitude(T_ROM_cubic_r25)
    fftfreq_cubic_r44, amplitude_cubic_r44, fft_cubic_r44 = get_freq_and_amplitude(T_ROM_cubic_r44)
    fftfreq, amplitude, fft = get_freq_and_amplitude(T)

    amplitude_plots(fftfreq,
                    fftfreq_Quad_r44, fftfreq_cubic_r25,
                    fftfreq_cubic_r44, fftfreq_DMDc,
                    amplitude,
                    amplitude_Quad_r44, amplitude_cubic_r25,
                    amplitude_cubic_r44, amplitude_DMDc,
                    unit, savefile,
                    title_test_or_train="Testing results plotted in the frequency domain",
                    save_id="_test_")

    # Phase

    Phase_FOM = np.angle(fft, deg=True)
    Phase_Quad_r44 = np.angle(fft_Quad_r44, deg=True)
    Phase_cubic_r25 = np.angle(fft_cubic_r25, deg=True)
    Phase_cubic_r44 = np.angle(fft_cubic_r44, deg=True)
    Phase_DMDc = np.angle(fft_DMDc, deg=True)

    phase_plots(fftfreq,
                fftfreq_Quad_r44, fftfreq_cubic_r25,
                fftfreq_cubic_r44, fftfreq_DMDc,
                Phase_FOM,
                Phase_Quad_r44, Phase_cubic_r25,
                Phase_cubic_r44, Phase_DMDc,
                unit, savefile,
                title_test_or_train="Testing results plotted in the frequency domain",
                save_id="_test_")


for T_st in np.arange(12,16,1):
    """
    T_st = monitor location code
    code number for each location:
    12 - Monitor location 1
    13 - Monitor location 2
    14 - Monitor location 3
    15 - Monitor location 4
    """

    fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='Temperature in Kelvin', datanumber=0, savefile='Temperature')
    fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='Pressure in Pa', datanumber=-12, savefile='Pressure')
    fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='CH4 in kmolm$^-3$', datanumber=8, savefile='CH4')
    # fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='O2 in kmolm$^-3$', datanumber=12)
    # fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='H2O in kmolm$^-3$', datanumber=16)
    fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='CO2 in kmolm$^-3$', datanumber=20, savefile='CO2')
    # fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='vx in ms-1', datanumber=-12+4)
    # fftoutput_train(T_st, t, trainsize, num_modes, reg, unit='vy in ms-1', datanumber=-12+8)

    fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='Temperature in Kelvin', datanumber=0, savefile='Temperature')
    fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='Pressure in Pa', datanumber=-12, savefile='Pressure')
    fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='CH4 in kmolm$^-3$', datanumber=8, savefile='CH4')
    # fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='O2 in kmolm$^-3$', datanumber=12)
    # fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='H2O in kmolm$^-3$', datanumber=16)
    fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='CO2 in kmolm$^-3$', datanumber=20, savefile='CO2')
    # fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='vx in ms-1', datanumber=-12+4)
    # fftoutput_test(T_st, t, trainsize, num_modes, reg, unit='vy in ms-1', datanumber=-12+8)
