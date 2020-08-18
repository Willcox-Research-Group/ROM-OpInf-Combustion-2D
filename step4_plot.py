# step4_plot.py
"""Simulate learned ROMs and visualize results. Figures are saved in the folder
given by config.figures_path().

Examples
--------
# Plot time traces of each variable at the monitor locations for the ROM
# trained from 10,000 snapshots with 22 POD modes and a regularization
# parameter 4e4.
$ python3 step4_plot.py 10000 --time-traces --modes 22 --regularization 4e4

# Plot spatial averages and species integrals for the ROM trained from
# 20,000 snapshots with 44 POD modes and a regularization parameter 5e4.
$ python3 step4_plot.py 20000 --species-integral -modes 44 --regularization 5e4

Loading Results
---------------
>>> import config
>>> print("figures are saved to", config.figures_path())

Command Line Arguments
----------------------
"""
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc


# Helper functions ============================================================

def simulate_rom(trainsize, r, reg, steps=None):
    """Load everything needed to simulate a given ROM, simulate the ROM,
    and return the simulation results and everything needed to reconstruct
    the results in the original high-dimensional space.
    Raise an Exception if any of the ingredients are missing.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    reg : float
        Regularization parameter used to train the ROM.

    steps : int or None
        Number of time steps to simulate the ROM.

    Returns
    -------
    t : (nt,) ndarray
        Time domain corresponding to the ROM outputs.

    V : (config*NUM_ROMVARS*config.DOF,r) ndarray
        POD basis used to project the training data (and for reconstructing
        the full-order scaled predictions).

    scales : (NUM_ROMVARS,4) ndarray
        Information for how the data was scaled. See data_processing.scale().

    x_rom : (nt,r) ndarray
        Prediction results from the ROM.
    """
    # Load the time domain, basis, initial conditions, and trained ROM.
    t = utils.load_time_domain(steps)
    V, _ = utils.load_basis(trainsize, r)
    X_, _, _, scales = utils.load_projected_data(trainsize, r)
    rom = utils.load_rom(trainsize, r, reg)

    # Simulate the ROM over the full time domain.
    with utils.timed_block(f"Simulating ROM with r={r:d}, "
                           f"reg={reg:e} over full time domain"):
        x_rom = rom.predict(X_[:,0], t, config.U, method="RK45")

    return t, V, scales, x_rom


def get_traces(locs, data, V=None, scales=None):
    """Reconstruct time traces from data, unprojecting and unscaling if needed.

    Parameters
    ----------
    locs : (l,nt) ndarray
        Index locations for the time traces to extract.

    data : (r,nt) or (config.DOF*config.NUM_ROMVARS,nt) ndarray
        Data from which to extract the time traces, either the output of a ROM
        or a high-dimensional data set.

    V : (config.DOF*config.NUM_ROMVARS,r) ndarray or None
        Rank-r POD basis. Only needed if `data` is low-dimensional ROM output.

    scales : (config.NUM_ROMVARS,4) ndarray or None
        Information for how the data was scaled (see data_processing.scale()).
        Only needed if `data` is low-dimensional ROM output.

    Returns
    -------
    traces : (l,nt) ndarray
        The specified time traces.
    """
    # TODO: input shape checking
    if V is not None and scales is not None:
        return dproc.unscale(V[locs] @ data, scales)
    else:
        return data[locs]


def get_feature(key, data, V=None, scales=None):
    """Reconstruct a statistical feature from data, unprojecting and
    unscaling if needed.

    Parameters
    ----------
    key : str
        Which statistical feature to calculate (T_mean, CH4_sum, etc.)

    data : (r,nt) or (config.DOF*config.NUM_ROMVARS,nt) ndarray
        Data from which to extract the features, either the output of a ROM
        or a high-dimensional data set.

    V : (config.DOF*config.NUM_ROMVARS,r) ndarray or None
        Rank-r POD basis. Only needed if data is low-dimensional ROM output.

    scales : (config.NUM_ROMVARS,4) ndarray or None
        Information for how the data was scaled (see data_processing.scale()).
        Only needed if `data` is low-dimensional ROM output.

    Returns
    -------
    feature : (nt,) ndarray
        The specified statistical feature.
    """
    # TODO: input shape checking
    var, action = key.split('_')
    print(f"{action}({var})", end='..', flush=True)
    if V is not None and scales is not None:
        variable = dproc.unscale(dproc.getvar(var, V) @ data, scales, var)
    else:
        variable = dproc.getvar(var, data)
    return eval(f"variable.{action}(axis=0)")


# Plot routines ===============================================================

def time_traces(trainsize, r, reg, elems):
    """Plot the time trace of each variable in the original data at the monitor
    location, and the time trace of each variable of the ROM reconstruction at
    the same locations. One figure is generated per variable.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    reg : float
        Regularization parameter used to train the ROM.

    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute the time traces.
    """
    # Get the indicies for each variable.
    elems = np.atleast_1d(elems)
    nelems = elems.size
    nrows = (nelems // 2) + (1 if nelems % 2 != 0 else 0)
    elems = np.concatenate([elems + i*config.DOF
                            for i in range(config.NUM_ROMVARS)])

    # Load and lift the true results.
    data, _ = utils.load_gems_data(rows=elems[:nelems*config.NUM_GEMSVARS])
    with utils.timed_block("Lifting GEMS time trace data"):
        traces_gems = dproc.lift(data)

    # Load and simulate the ROM.
    t, V, scales, x_rom = simulate_rom(trainsize, r, reg)

    # Reconstruct and rescale the simulation results.
    simend = x_rom.shape[1]
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom = dproc.unscale(V[elems] @ x_rom, scales)

    # Save a figure for each variable.
    xticks = np.arange(t[0], t[-1]+.001, .002)
    for i,var in enumerate(config.ROM_VARIABLES):
        fig, axes = plt.subplots(nrows, 2 if nelems > 1 else 1,
                                 figsize=(9,3*nrows), sharex=True)
        axes = np.atleast_2d(axes)
        for j, ax in enumerate(axes.flat):
            idx = j + i*nelems
            ax.plot(t, traces_gems[idx,:], lw=1, **config.GEMS_STYLE)
            ax.plot(t[:simend], traces_rom[idx,:], lw=1, **config.ROM_STYLE)
            ax.axvline(t[trainsize], color='k', lw=1)
            ax.set_xlim(t[0], t[-1])
            ax.set_xticks(xticks)
            ax.set_title(f"Location ${j+1}$", fontsize=12)
            ax.locator_params(axis='y', nbins=2)
        for ax in axes[-1,:]:
            ax.set_xlabel("Time [s]", fontsize=12)
        for ax in axes[:,0]:
            ax.set_ylabel(config.VARLABELS[var], fontsize=12)

        # Single legend to the right of the subplots.
        fig.tight_layout(rect=[0, 0, .85, 1])
        leg = axes[0,0].legend(loc="center right", fontsize=14,
                               bbox_to_anchor=(1,.5),
                               bbox_transform=fig.transFigure)
        for line in leg.get_lines():
            line.set_linewidth(2)

        # Save the figure.
        utils.save_figure("timetrace"
                          f"_{config.TRNFMT(trainsize)}"
                          f"_{config.DIMFMT(r)}"
                          f"_{config.REGFMT(reg)}_{var}.pdf")


def save_statistical_features():
    """Compute the (spatial) mean temperatures on the full time domain and
    save them for later. This only needs to be done once.
    """
    # Load the full data set.
    gems_data, t = utils.load_gems_data()

    # Lift the data (convert to molar concentrations).
    with utils.timed_block("Lifting GEMS data"):
        lifted_data = dproc.lift(gems_data)

    # Compute statistical features.
    with utils.timed_block("Computing statistical features of variables"):
        mins, maxs, sums, stds, means = {}, {}, {}, {}, {}
        for var in config.ROM_VARIABLES:
            val = dproc.getvar(var, lifted_data)
            mins[var] = val.min(axis=0)
            maxs[var] = val.max(axis=0)
            sums[var] = val.sum(axis=0)
            stds[var] = val.std(axis=0)
            means[var] = sums[var] / val.shape[0]

    # Save the data.
    data_path = config.statistical_features_path()
    with utils.timed_block("Saving statistical features"):
        with h5py.File(data_path, 'w') as hf:
            for var in config.ROM_VARIABLES:
                hf.create_dataset(f"{var}_min", data=mins[var])
                hf.create_dataset(f"{var}_max", data=maxs[var])
                hf.create_dataset(f"{var}_sum", data=sums[var])
                hf.create_dataset(f"{var}_std", data=stds[var])
                hf.create_dataset(f"{var}_mean", data=means[var])
            hf.create_dataset("time", data=t)
    logging.info(f"Statistical features saved to {data_path}")


def statistical_features(trainsize, r, reg):
    """Plot spatially averaged temperature and spacially itegrated (summed)
    species concentrations over the full time domain.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    reg : float
        Regularization parameter used to train the ROM.
    """
    # Load the true results.
    keys = [f"{var}_mean" for var in config.ROM_VARIABLES[:4]]
    keys += [f"{var}_sum" for var in config.SPECIES]
    feature_gems, t = utils.load_statistical_features(keys)
    keys = np.reshape(keys, (4,2), order='F')

    # Load and simulate the ROM.
    t, V, scales, x_rom = simulate_rom(trainsize, r, reg)

    # Initialize the figure.
    fig, axes = plt.subplots(keys.shape[0], keys.shape[1],
                             figsize=(9,6), sharex=True)

    # Calculate and plot the results.
    for ax,key in zip(axes.flat, keys.flat):
        with utils.timed_block(f"Reconstructing"):
            feature_rom = get_feature(key, x_rom, V, scales)
        ax.plot(t, feature_gems[key], lw=1, **config.GEMS_STYLE)
        ax.plot(t[:x_rom.shape[1]], feature_rom, lw=1, **config.ROM_STYLE)
        ax.axvline(t[trainsize], color='k')
        ax.set_ylabel(config.VARLABELS[var])
        ax.locator_params(axis='y', nbins=2)

    # Set titles, labels, ticks, and draw a single legend.
    for ax in axes[-1,:]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=12)
    axes[0,0].set_title("Spatial Averages", fontsize=14)
    axes[0,1].set_title("Spatial Integrals", fontsize=14)

    # Legend on the right.
    fig.tight_layout(rect=[0, 0, .85, 1])
    leg = axes[0,0].legend(loc="center right", fontsize=14,
                           bbox_to_anchor=(1,.5),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)

    utils.save_figure(f"statfeatures"
                      f"_{config.TRNFMT(trainsize)}"
                      f"_{config.DIMFMT(r)}"
                      f"_{config.REGFMT(reg)}.pdf")

# =============================================================================

def main(trainsize, r, reg, elems,
         plotTimeTrace=False, plotStatisticalFeatures=False):
    """Make the indicated visualization.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    reg : float
        The regularization parameters used to train each ROM.

    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute time traces.
    """
    utils.reset_logger(trainsize)

    # Time traces (single ROM, several monitoring locations).
    if plotTimeTrace:
        logging.info("TIME TRACES")
        time_traces(trainsize, r, reg, elems)

    # Statistical features (single ROM, several features).
    if plotStatisticalFeatures:
        logging.info("STATISTICAL FEATURES")
        # Compute GEMS features if needed (only done once).
        if not os.path.isfile(config.statistical_features_path()):
            save_statistical_features()
        statistical_features(trainsize, r, reg)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE --time-traces
                                  --modes R --regularization REG
                                  --location M [...]
        python3 {__file__} TRAINSIZE --statistical-features
                                  --modes R --regularization REG"""
    # Positional arguments
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")

    # Routine indicators
    parser.add_argument("-tt", "--time-traces", action="store_true",
                       help="plot time traces for the given "
                            "basis sizes and regularization parameters "
                            "at the specified monitoring locations")
    parser.add_argument("-sf", "--statistical-features", action="store_true",
                        help="plot spatial averages and species integrals "
                             "for the ROM with the given basis size and "
                             "regularization parameters")

    # Other keyword arguments
    parser.add_argument("-r", "--modes", type=int, required=True,
                        help="number of POD modes used to project data")
    parser.add_argument("-reg", "--regularization", type=float, required=True,
                        help="regularization parameter used in ROM training")
    parser.add_argument("-loc", "--location", type=int, nargs='+',
                        default=config.MONITOR_LOCATIONS,
                        help="monitor locations for time trace plots")

    # Do the main routine.
    args = parser.parse_args()
    main(trainsize=args.trainsize, r=args.modes, reg=args.regularization,
         plotTimeTrace=args.time_traces, elems=args.location,
         plotStatisticalFeatures=args.statistical_features)
