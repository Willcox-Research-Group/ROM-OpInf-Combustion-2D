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
$ python3 step4_plot.py 20000 --spatial-statistics --modes 44 --regularization 5e4

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

def simulate_rom(trainsize, r1, r2, reg, steps=None):
    """Load everything needed to simulate a given ROM, simulate the ROM,
    and return the simulation results and everything needed to reconstruct
    the results in the original high-dimensional space.
    Raise an Exception if any of the ingredients are missing.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        The number of retained POD modes used in the NON-T projection.

    r2 : int
        The number of retained POD modes used in the T-only projection.

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

    q_rom : (nt,r) ndarray
        Prediction results from the ROM.
    """
    # Load the time domain, basis, initial conditions, and trained ROM.
    t = utils.load_time_domain(steps)
    V, scales = utils.load_basis(trainsize, r1, r2)
    Q_, _, _ = utils.load_projected_data(trainsize, r1, r2)
    rom = utils.load_rom(trainsize, r1, r2, reg)

    # Simulate the ROM over the full time domain.
    with utils.timed_block(f"Simulating ROM with r1={r1:d}, r2={r2:d}, "
                           f"reg={reg:e} over full time domain"):
        q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

    return t, V, scales, q_rom


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


def veccorrcoef(X, Y):
    """Calculate the (vectorized) linear correlation coefficent,

                     sum_i[(X_i - Xbar)Y_i - Ybar)]
        r = -------------------------------------------------.
            sqrt(sum_i[(X_i - Xbar)^2] sum_i[(Y_i - Ybar)^2])

    This function is equivalent to (but much faster than)
    >>> r = [np.corrcoef(X[:,j], Y[:,j])[0,1] for j in range(X.shape[1])].

    Parameters
    ----------
    X : (n,k) ndarray
        First array of data, e.g., ROM reconstructions of one variable.

    Y : (n,k) ndarray
        Second array of data, e.g., GEMS data of one variable.

    Returns
    -------
    r : (k,) ndarray
        Linear correlation coefficient of X[:,j], Y[:,j] for j = 0, ..., k-1.
    """
    dX = X - np.mean(X, axis=0)
    dY = Y - np.mean(Y, axis=0)
    numer = np.sum(dX*dY, axis=0)
    denom2 = np.sum(dX**2, axis=0) * np.sum(dY**2, axis=0)
    return numer / np.sqrt(denom2)


# Plot routines ===============================================================

def time_traces(trainsize, r1, r2, reg, elems):
    """Plot the time trace of each variable in the original data at the monitor
    location, and the time trace of each variable of the ROM reconstruction at
    the same locations. One figure is generated per variable.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        The number of retained POD modes used in the NON-T projection.

    r2 : int
        The number of retained POD modes used in the T-only projection.

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
        traces_gems = dproc.lift(data[:,:60000])

    # Load and simulate the ROM.
    t, V, scales, q_rom = simulate_rom(trainsize, r1, r2, reg, 60000)

    # Reconstruct and rescale the simulation results.
    simend = q_rom.shape[1]
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom = dproc.unscale(V[elems] @ q_rom, scales)

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
        utils.save_figure("pointtrace"
                          f"_{config.TRNFMT(trainsize)}"
                          f"_{config.DIMFMT([r1,r2])}"
                          f"_{config.REGFMT(reg)}_{var}.pdf")


def errors_in_time(trainsize, r1, r2, reg):
    """Plot spatially averaged errors, and the projection error, in time.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        The number of retained POD modes used in the NON-T projection.

    r2 : int
        The number of retained POD modes used in the T-only projection.

    reg : float
        Regularization parameter used to train the ROM.
    """
    # Load and simulate the ROM.
    t, V, scales, q_rom = simulate_rom(trainsize, r1, r2, reg, 60000)

    # Load and lift the true results.
    data, _ = utils.load_gems_data(cols=60000)
    with utils.timed_block("Lifting GEMS data"):
        data_gems = dproc.lift(data[:,:60000])
    del data

    # Shift and project the data (unscaling done later by chunk).
    with utils.timed_block("Projecting GEMS data to POD subspace"):
        data_shifted, _ = dproc.scale(data_gems.copy(), scales)
        data_proj = V.T @ data_shifted
        del data_shifted

    # Initialize the figure.
    fig, axes = plt.subplots(3, 3, figsize=(12,6), sharex=True)

    # Compute and plot errors in each variable.
    for var, ax in zip(config.ROM_VARIABLES, axes.flat):

        with utils.timed_block(f"Reconstructing results for {var}"):
            Vvar = dproc.getvar(var, V)
            gems_var = dproc.getvar(var, data_gems)
            proj_var = dproc.unscale(Vvar @ data_proj, scales, var)
            pred_var = dproc.unscale(Vvar @ q_rom, scales, var)

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
    utils.save_figure(f"errors"
                      f"_{config.TRNFMT(trainsize)}"
                      f"_{config.DIMFMT([r1,r2])}"
                      f"_{config.REGFMT(reg)}.pdf")
    return


def corrcoef(trainsize, r1, r2, reg):
    """Plot correlation coefficients in time between GEMS and ROM solutions.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        The number of retained POD modes used in the NON-T projection.

    r2 : int
        The number of retained POD modes used in the T-only projection.

    reg : float
        Regularization parameter used to train the ROM.
    """
    # Load and simulate the ROM.
    t, V, scales, q_rom = simulate_rom(trainsize, r1, r2, reg, 60000)

    # Load and lift the true results.
    data, _ = utils.load_gems_data(cols=60000)
    with utils.timed_block("Lifting GEMS data"):
        data_gems = dproc.lift(data[:,:60000])

    # Initialize the figure.
    fig, axes = plt.subplots(3, 3, figsize=(12,6), sharex=True, sharey=True)

    # Compute and plot errors in each variable.
    for var, ax in zip(config.ROM_VARIABLES, axes.flat):

        with utils.timed_block(f"Reconstructing results for {var}"):
            Vvar = dproc.getvar(var, V)
            gems_var = dproc.getvar(var, data_gems)
            pred_var = dproc.unscale(Vvar @ q_rom, scales, var)

        with utils.timed_block(f"Calculating correlation in {var}"):
            corr = veccorrcoef(gems_var, pred_var)

        # Plot results.
        ax.plot(t, corr, '-', lw=1, color='C2')
        ax.axvline(t[trainsize], color='k')
        ax.axhline(.8, ls='--', lw=1, color='k', alpha=.25)
        ax.set_ylim(0, 1)
        ax.set_ylabel(config.VARTITLES[var])

    # Format the figure.
    for ax in axes[-1,:]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=14)
    fig.suptitle("Linear Correlation Coefficient", fontsize=16)

    # Save the figure.
    utils.save_figure(f"corrcoef"
                      f"_{config.TRNFMT(trainsize)}"
                      f"_{config.DIMFMT([r1,r2])}"
                      f"_{config.REGFMT(reg)}.pdf")


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


def spatial_statistics(trainsize, r1, r2, reg):
    """Plot spatially averaged temperature and spacially itegrated (summed)
    species concentrations over the full time domain.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        The number of retained POD modes used in the NON-T projection.

    r2 : int
        The number of retained POD modes used in the T-only projection.

    reg : float
        Regularization parameter used to train the ROM.
    """
    # Load the true results.
    keys = [f"{var}_mean" for var in config.ROM_VARIABLES[:4]]
    keys += [f"{var}_sum" for var in config.SPECIES]
    feature_gems, t = utils.load_spatial_statistics(keys)
    keys = np.reshape(keys, (4,2), order='F')

    # Load and simulate the ROM.
    t, V, scales, q_rom = simulate_rom(trainsize, r1, r2, reg)

    # Initialize the figure.
    fig, axes = plt.subplots(keys.shape[0], keys.shape[1],
                             figsize=(9,6), sharex=True)

    # Calculate and plot the results.
    for ax,key in zip(axes.flat, keys.flat):
        with utils.timed_block(f"Reconstructing"):
            feature_rom = get_feature(key, q_rom, V, scales)
        ax.plot(t, feature_gems[key], lw=1, **config.GEMS_STYLE)
        ax.plot(t[:q_rom.shape[1]], feature_rom, lw=1, **config.ROM_STYLE)
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
                      f"_{config.DIMFMT([r1,r2])}"
                      f"_{config.REGFMT(reg)}.pdf")

# =============================================================================

def main(trainsize, r1, r2, reg, elems,
         plotTimeTrace=False, plotStatisticalFeatures=False,
         plotErrors=False, plotCorrelation=False):
    """Make the indicated visualization.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        The number of retained POD modes used in the NON-T projection.

    r2 : int
        The number of retained POD modes used in the T-only projection.

    reg : float
        The regularization parameters used to train each ROM.

    elems : list(int) or ndarray(int)
        Indices in the spatial domain at which to compute time traces.
    """
    utils.reset_logger(trainsize)

    # Time traces (single ROM, several monitoring locations).
    if plotTimeTrace:
        logging.info("POINT TRACES")
        time_traces(trainsize, r1, r2, reg, elems)

    # Statistical features (single ROM, several features).
    if plotStatisticalFeatures:
        logging.info("STATISTICAL FEATURES")
        # Compute GEMS features if needed (only done once).
        if not os.path.isfile(config.statistical_features_path()):
            save_statistical_features()
        spatial_statistics(trainsize, r1, r2, reg)

    if plotErrors:
        logging.info("ERRORS IN TIME")
        errors_in_time(trainsize, r1, r2, reg)

    if plotCorrelation:
        logging.info("CORRELATIONS IN TIME")
        corrcoef(trainsize, r1, r2, reg)

    return


def projection_errors(trainsize, rs):
    """Plot spatially averaged projection errors in time.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    rs : list(int)
        Basis sizes to test
    """
    # Load and lift the true results.
    data, t = utils.load_gems_data()
    with utils.timed_block("Lifting GEMS data"):
        data_gems = dproc.lift(data)
    del data

    # Initialize the figure.
    fig, axes = plt.subplots(3, 3, figsize=(12,6), sharex=True)

    # Get projection errors for each r.
    for r in rs:

        # Load the POD basis of rank r.
        V, scales = utils.load_basis(trainsize, r)

        # Shift the data (unscaling done later by chunk).
        if r == rs[0]:
            with utils.timed_block(f"Shifting GEMS data"):
                data_shifted, _ = dproc.scale(data_gems.copy(), scales)

        # Project the shifted data.
        with utils.timed_block(f"Projecting GEMS data to rank-{r} subspace"):
            data_proj = V.T @ data_shifted

        # Compute and plot errors in each variable.
        for var, ax in zip(config.ROM_VARIABLES, axes.flat):

            with utils.timed_block(f"Reconstructing results for {var}"):
                Vvar = dproc.getvar(var, V)
                gems_var = dproc.getvar(var, data_gems)
                proj_var = dproc.unscale(Vvar @ data_proj, scales, var)

            with utils.timed_block(f"Calculating error in {var}"):
                denom = np.abs(gems_var).max(axis=0)
                proj_error = np.mean(np.abs(proj_var-gems_var), axis=0) / denom

            # Plot results.
            ax.plot(t, proj_error, '-', lw=1, label=fr"$r = {r}$")

    # Format the figure.
    for ax in axes[-1,:]:
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_xlabel("Time [s]", fontsize=12)
    for var, ax in zip(config.ROM_VARIABLES, axes.flat):
        ax.axvline(t[trainsize], color='k')
        ax.set_ylabel(config.VARTITLES[var])

    # Make legend centered below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    leg = axes[0,0].legend(ncol=3, fontsize=14, loc="lower center",
                           bbox_to_anchor=(.5, 0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linestyle('-')
        line.set_linewidth(5)

    # Save the figure.
    utils.save_figure(f"projerrors_{config.TRNFMT(trainsize)}.pdf")


# =============================================================================
if __name__ == "__main__":

    # projection_errors( 5000, [11, 14,  19,  24,  32,  55])
    # projection_errors(10000, [22, 28,  37,  47,  62]) #, 108])
    # projection_errors(20000, [44, 54,  73,  94, 123]) #, 214])
    # projection_errors(30000, [68, 83, 112, 144, 188]) #, 326])
    ## projection_errors(40000, [87, 108, 147, 188, 247, 426])
    # projection_errors(30000, [68, 326])
    # import sys; sys.exit(0)

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
    parser.add_argument("-sf", "--spatial-statistics", action="store_true",
                        help="plot spatial averages and species integrals "
                             "for the ROM with the given basis size and "
                             "regularization parameters")
    parser.add_argument("--errors", action="store_true",
                       help="plot normalized absolute errors, averaged over "
                            "the spatial domain, as a function of time")
    parser.add_argument("--correlation", action="store_true",
                       help="plot correlation coefficients in time for each "
                            "variable")

    # Other keyword arguments
    parser.add_argument("-r1", type=int, required=True,
                        help="number of POD modes used to project "
                             "non-temperature data")
    parser.add_argument("-r2", type=int, required=True,
                        help="number of POD modes used to project "
                             "temperature data")
    parser.add_argument("-reg", "--regularization", type=float, required=True,
                        help="regularization parameter used in ROM training")
    parser.add_argument("-loc", "--location", type=int, nargs='+',
                        default=config.MONITOR_LOCATIONS,
                        help="monitor locations for time trace plots")

    # Do the main routine.
    args = parser.parse_args()
    main(trainsize=args.trainsize,
         r1=args.r1, r2=args.r2, reg=args.regularization,
         plotTimeTrace=args.time_traces, elems=args.location,
         plotStatisticalFeatures=args.spatial_statistics,
         plotErrors=args.errors, plotCorrelation=args.correlation)
