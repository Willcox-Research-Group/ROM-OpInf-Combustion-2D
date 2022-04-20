# step4_plot.py
"""Simulate learned ROMs and visualize results. Figures are saved in the folder
given by config.figures_path().

Examples
--------
## --point-traces: plot results in time at fixed spatial locations.

# Plot time traces of each variable at the monitor locations for the ROM
# trained from 10,000 snapshots with 22 POD modes and regularization
# hyperparameters λ1 = 300, λ2 = 21000.
$ python3 step4_plot.py --point-traces 10000 22 300 21000

## --spatial-statistics: plot results in time averaged over the spatial domain.

# Plot spatial averages and species integrals for the ROM trained from 20,000
# snapshots with 40 POD modes and regularization hyperparameters
# λ1 = 9e3, λ2 = 1e4.
$ python3 step4_plot.py --spatial-statistics 20000 40 9e3 1e4

## --relative-errors: plot relative projection and prediction errors in time,
                      averaged over the spatial domain.

# Plot errors for the ROM trained from 20,000 snapshots with 43 POD modes and
# regularization parameters λ1 = 350, λ2 = 18500.
$ python3 step4_plot.py --errors 20000 43 350 18500

Loading Results
---------------
>>> import config
>>> print("figures are saved to", config.figures_path())

Command Line Arguments
----------------------
"""
import os
import h5py
import logging
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
import data_processing as dproc


# Helper functions ============================================================

def simulate_rom(trainsize, r, regs, steps=None):
    """Load everything needed to simulate a given ROM, run the simulation,
    and return the simulation results and everything needed to reconstruct
    the results in the original high-dimensional space.
    Raise an Exception if any of the ingredients are missing.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Dimension of the ROM.
    regs : 2 or 3 positive floats
        Regularization hyperparameters used to train the ROM.
    steps : int or None
        Number of time steps to simulate the ROM.

    Returns
    -------
    t : (nt,) ndarray
        Time domain corresponding to the ROM outputs.
    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis used to project the training data (and for reconstructing
        the full-order scaled predictions).
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot that the training data was shifted by before scaling.
    scales : (NUM_ROMVARS,4) ndarray
        Information for how the data was scaled. See data_processing.scale().
    q_rom : (nt,r) ndarray
        Prediction results from the ROM.
    """
    # Load the time domain, basis, initial conditions, and trained ROM.
    t = utils.load_time_domain(steps)
    V, qbar, scales = utils.load_basis(trainsize, r)
    Q_, _, _ = utils.load_projected_data(trainsize, r)
    rom = utils.load_rom(trainsize, r, regs)

    # Simulate the ROM over the full time domain.
    with utils.timed_block(f"Simulating ROM with k={trainsize:d}, r={r:d}, "
                           f"{config.REGSTR(regs)} over full time domain"):
        q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

    return t, V, qbar, scales, q_rom


def get_traces(locs, data, V=None, qbar=None, scales=None):
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
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot that the training data was shifted by before scaling.
        Only needed if `data` is low-dimensional ROM output.
    scales : (config.NUM_ROMVARS,4) ndarray or None
        Information for how the data was scaled (see data_processing.scale()).
        Only needed if `data` is low-dimensional ROM output.

    Returns
    -------
    traces : (l,nt) ndarray
        The specified time traces.
    """
    if V is not None and qbar is not None and scales is not None:
        qbar = qbar.reshape((-1,1))
        return dproc.unscale((V[locs] @ data), scales) + qbar[locs,:]
    else:
        return data[locs]


def get_feature(key, data, V=None, qbar=None, scales=None):
    """Reconstruct a spatial statistical feature from data, unprojecting and
    unscaling if needed.

    Parameters
    ----------
    key : str
        Which statistical feature to calculate (T_mean, CH4_sum, etc.)
    data : (r,nt) or (DOF*NUM_ROMVARS,nt) ndarray
        Data from which to extract the features, either the output of a ROM
        or a high-dimensional data set.
    V : (DOF*NUM_ROMVARS,r) ndarray or None
        Rank-r POD basis. Only needed if data is low-dimensional ROM output.
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot that the training data was shifted by before scaling.
        Only needed if `data` is low-dimensional ROM output.
    scales : (NUM_ROMVARS,) ndarray or None
        Information for how the data was scaled (see data_processing.scale()).
        Only needed if `data` is low-dimensional ROM output.

    Returns
    -------
    feature : (nt,) ndarray
        The specified statistical feature.
    """
    var, action = key.split('_')
    print(f"{action}({var})", end='..', flush=True)
    if V is not None and qbar is not None and scales is not None:
        qbarvar = dproc.getvar(var, qbar).reshape((-1,1))
        data_scaled = dproc.getvar(var, V) @ data
        variable = dproc.unscale(data_scaled, scales, var) + qbarvar
    else:
        variable = dproc.getvar(var, data)
    return eval(f"variable.{action}(axis=0)")


# Plot routines ===============================================================

def point_traces(trainsize, r, regs, elems, cutoff=60000):
    """Plot the time trace of each variable in the original data at the monitor
    location, and the time trace of each variable of the ROM reconstruction at
    the same locations. One figure is generated per variable.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Dimension of the ROM.
    regs : 2 or 3 positive floats
        Regularization hyperparameters used to train the ROM.
    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute the point traces.
    cutoff : int
        Numer of time steps to plot.
    """
    if elems is None:
        elems = config.MONITOR_LOCATIONS

    # Get the indicies for each variable.
    elems = np.atleast_1d(elems)
    nelems = elems.size
    nrows = (nelems // 2) + (1 if nelems % 2 != 0 else 0)
    elems = np.concatenate([elems + i*config.DOF
                            for i in range(config.NUM_ROMVARS)])

    # Load and lift the true results.
    data, _ = utils.load_gems_data(rows=elems[:nelems*config.NUM_GEMSVARS])
    with utils.timed_block("Lifting GEMS time trace data"):
        traces_gems = dproc.lift(data[:,:cutoff])

    # Load and simulate the ROM.
    t, V, qbar, scales, q_rom = simulate_rom(trainsize, r, regs, cutoff)

    # Reconstruct and rescale the simulation results.
    simend = q_rom.shape[1]
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom = get_traces(elems, q_rom, V, qbar, scales)

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
        utils.save_figure(f"pointtrace_{var}.pdf")


def errors_in_time(trainsize, r, regs, cutoff=60000):
    """Plot spatially averaged errors, and the projection error, in time.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Dimension of the ROM.
    regs : 2 or 3 positive floats
        Regularization hyperparameters used to train the ROM.
    cutoff : int
        Numer of time steps to plot.
    """
    # Load and simulate the ROM.
    t, V, qbar, scales, q_rom = simulate_rom(trainsize, r, regs, cutoff)
    qbar = qbar.reshape((-1,1))

    # Load and lift the true results.
    data, _ = utils.load_gems_data(cols=cutoff)
    with utils.timed_block("Lifting GEMS data"):
        data_gems = dproc.lift(data[:,:cutoff])
    del data

    # Shift and project the data (unscaling done later by chunk).
    with utils.timed_block("Projecting GEMS data to POD subspace"):
        data_scaled, _ = dproc.scale(data_gems - qbar, scales)
        data_proj = V.T @ data_scaled
        del data_scaled

    # Initialize the figure.
    fig, axes = plt.subplots(3, 3, figsize=(12,6), sharex=True)

    # Compute and plot errors in each variable.
    for var, ax in zip(config.ROM_VARIABLES, axes.flat):

        with utils.timed_block(f"Reconstructing results for {var}"):
            Vvar = dproc.getvar(var, V)
            gems_var = dproc.getvar(var, data_gems)
            qbarvar = dproc.getvar(var, qbar)
            proj_var = dproc.unscale(Vvar @ data_proj, scales, var) + qbarvar
            pred_var = dproc.unscale(Vvar @ q_rom, scales, var) + qbarvar

        with utils.timed_block(f"Calculating error in {var}"):
            denom = np.abs(gems_var).max(axis=0)
            proj_error = np.mean(np.abs(proj_var - gems_var), axis=0) / denom
            pred_error = np.mean(np.abs(pred_var - gems_var), axis=0) / denom

        # Plot results.
        ax.plot(t, proj_error, '-', lw=1, label="Projection Error",
                c=config.GEMS_STYLE['color'])
        ax.plot(t, pred_error, '-', lw=1, label="ROM Error",
                c=config.ROM_STYLE['color'])
        ax.axvline(t[trainsize], color='k')
        ax.set_ylabel(config.VARTITLES[var])

    # Format the figure.
    for ax in axes[-1,:]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=12)

    # Make legend centered below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    leg = axes[0,0].legend(ncol=2, fontsize=14, loc="lower center",
                           bbox_to_anchor=(.5, 0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linestyle('-')
        line.set_linewidth(5)

    # Save the figure.
    utils.save_figure("errors.pdf")


def save_statistical_features():
    """Compute the spatial and temporal statistics (min, max, mean, etc.)
    for each variable and save them for later. This only needs to be done once.
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
            for axis, label in enumerate(["space", "time"]):
                name = f"{label}/{var}"
                print(f"\n\tmin_{label}({var})", end='..', flush=True)
                mins[name] = val.min(axis=axis)
                print(f"max_{label}({var})", end='..', flush=True)
                maxs[name] = val.max(axis=axis)
                print(f"sum_{label}({var})", end='..', flush=True)
                sums[name] = val.sum(axis=axis)
                print(f"std_{label}({var})", end='..', flush=True)
                stds[name] = val.std(axis=axis)
            means[f"space/{var}"] = sums[f"space/{var}"] / val.shape[0]
            means[f"time/{var}"] = sums[f"time/{var}"] / t.size

    # Save the data.
    data_path = config.statistical_features_path()
    with utils.timed_block("Saving statistical features"):
        with h5py.File(data_path, 'w') as hf:
            for var in config.ROM_VARIABLES:
                for prefix in ["space", "time"]:
                    name = f"{prefix}/{var}"
                    hf.create_dataset(f"{name}_min", data=mins[name])
                    hf.create_dataset(f"{name}_max", data=maxs[name])
                    hf.create_dataset(f"{name}_sum", data=sums[name])
                    hf.create_dataset(f"{name}_std", data=stds[name])
                    hf.create_dataset(f"{name}_mean", data=means[name])
            hf.create_dataset("t", data=t)
    logging.info(f"Statistical features saved to {data_path}")


def spatial_statistics(trainsize, r, regs):
    """Plot spatially averaged temperature and spacially itegrated (summed)
    species concentrations over the full time domain.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Dimension of the ROM.
    regs : 2 or 3 positive floats
        Regularization hyperparameters used to train the ROM.
    """
    # Load the true results.
    keys = [f"{var}_mean" for var in config.ROM_VARIABLES[:4]]
    keys += [f"{var}_sum" for var in config.SPECIES]
    feature_gems, t = utils.load_spatial_statistics(keys)
    keys = np.reshape(keys, (4,2), order='F')

    # Load and simulate the ROM.
    t, V, qbar, scales, q_rom = simulate_rom(trainsize, r, regs)

    # Initialize the figure.
    fig, axes = plt.subplots(keys.shape[0], keys.shape[1],
                             figsize=(9,6), sharex=True)

    # Calculate and plot the results.
    for ax,key in zip(axes.flat, keys.flat):
        with utils.timed_block("Reconstructing"):
            feature_rom = get_feature(key, q_rom, V, qbar, scales)
        ax.plot(t, feature_gems[key], lw=1, **config.GEMS_STYLE)
        ax.plot(t[:q_rom.shape[1]], feature_rom, lw=1, **config.ROM_STYLE)
        ax.axvline(t[trainsize], color='k')
        ax.set_ylabel(config.VARLABELS[key.split('_')[0]])
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

    utils.save_figure("statfeatures.pdf")


# Main routine ================================================================

def main(trainsize, r, regs, elems=None, plotPointTrace=False,
         plotRelativeErrors=False, plotSpatialStatistics=False):
    """Make the indicated visualization.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Dimension of the ROM.
    regs : two positive floats
        Regularization hyperparameters used to train the ROM.
    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute time traces.
    """
    utils.reset_logger(trainsize)

    # Point traces in time.
    if plotPointTrace:
        logging.info("POINT TRACES")
        point_traces(trainsize, r, regs, elems)

    # Relative projection / prediction errors in time.
    if plotRelativeErrors:
        logging.info("ERRORS IN TIME")
        errors_in_time(trainsize, r, regs)

    # Spatial statistic in time.
    if plotSpatialStatistics:
        logging.info("SPATIAL STATISTICS")
        # Compute GEMS features if needed (only done once).
        if not os.path.isfile(config.statistical_features_path()):
            save_statistical_features()
        spatial_statistics(trainsize, r, regs)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} --point-traces TRAINSIZE MODES REG1 REG2
                           [--location L [...]]
        python3 {__file__} --relative-errors TRAINSIZE MODES REG1 REG2
        python3 {__file__} --spatial-statistics TRAINSIZE MODES REG1 REG2"""

    # Positional arguments
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")
    parser.add_argument("modes", type=int,
                        help="number of POD modes used to project the data"
                             " (dimension of the learned ROM)")
    parser.add_argument("regularization", type=float, nargs='+',
                        help="regularization hyperparameters used in the "
                             "Operator Inference problem for learning the ROM")

    # Routine indicators
    parser.add_argument("--point-traces", action="store_true",
                        help="plot point traces in time at the specified "
                             "monitoring locations")
    parser.add_argument("--relative-errors", action="store_true",
                        help="plot relative errors in time, averaged over "
                             "the spatial domain")
    parser.add_argument("--spatial-statistics", action="store_true",
                        help="plot spatial averages and species integrals")

    # Other keyword arguments
    parser.add_argument("--location", type=int, nargs='+',
                        default=config.MONITOR_LOCATIONS,
                        help="monitor locations for time trace plots")

    # Parse the arguments and do the main routine.
    args = parser.parse_args()
    main(trainsize=args.trainsize,
         r=args.modes, regs=args.regularization,
         plotPointTrace=args.point_traces, elems=args.location,
         plotRelativeErrors=args.relative_errors,
         plotSpatialStatistics=args.spatial_statistics)
