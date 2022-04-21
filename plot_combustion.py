# plot_combustion.py
"""Plots for the combustion BayesOpInf example."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import rom_operator_inference as opinf

import config
import utils
import step3_train as step3
import data_processing as dproc
import bayes


DATAFILENAME = "bayes_data.h5"
black = (.2, .2, .2)


def init_settings():
    """Turn on custom matplotlib settings."""
    plt.rc("figure", figsize=(12,4))
    plt.rc("axes", titlesize="xx-large", labelsize="xx-large", linewidth=.5)
    plt.rc("xtick", labelsize="large")  # "x-large" ?
    plt.rc("ytick", labelsize="large")  # "x-large" ?
    plt.rc("legend", fontsize="xx-large", frameon=False, edgecolor="none")


def _plotdatafile(trainsize, r):
    """Get the name of the file to save data at."""
    savefolder = config._makefolder(config.BASE_FOLDER,
                                    config.TRNFMT(trainsize),
                                    config.DIMFMT(r))
    return os.path.join(savefolder, DATAFILENAME)


def generate_plot_data(trainsize, r, reg, ndraws, steps, overwrite=False):
    """Generate data to be visualized.

    Parameters
    ----------
    trainsize : int
        Number of training snapshots.
    r : int
        Number of POD modes to retain (size of the ROM).
    reg : (float, float)
        Regularization hyperparameters for non-quadratic / quadratic terms.
    ndraws : int
        Number of simulation samples to draw.
    steps : int
        Number of time steps to simulate the posterior data.
    overwrite : bool
        If True, write data even if the outfile already exists.
    """
    # Construct the name of the file to export data to.
    savefile = _plotdatafile(trainsize, r)
    if os.path.isfile(savefile) and not overwrite:
        raise FileExistsError(f"{savefile} (set overwrite=True to ignore)")

    # Run numerical experiment.
    if isinstance(reg, tuple) and len(reg) == 2:
        reg = step3.regularizer(r, reg[0], reg[1])
    Q_, R_, t = utils.load_projected_data(trainsize, r)
    U = config.U(t).reshape((1,-1))
    rom = opinf.InferredContinuousROM("cAHB").fit(None, Q_, R_, U, reg)

    post = bayes.construct_posterior(rom, reg, case=-1)
    t = utils.load_time_domain(steps)
    mean, draws = bayes.simulate_posterior(post, Q_[:,0], t, config.U, ndraws)

    # Save the data.
    with utils.timed_block(f"saving data to {savefile}"):
        with h5py.File(savefile, 'w') as hf:
            gp = hf.create_group("meta")
            gp.create_dataset("trainsize", data=[trainsize])
            gp.create_dataset("r", data=[r])
            gp.create_dataset("regularization", data=reg)

            gp = hf.create_group("data")
            gp.create_dataset("mean", data=mean)
            gp.create_dataset("draws", data=draws)


# Plotting routines ===========================================================

def _plot_rom_draws(ax, t, rom_mean, rom_sample_mean, rom_deviation, ndevs=3,
                    gems=None, tf=None):
    """Plot ROM mean ± `ndevs` standard deviations.

    Parameters
    ----------
    ax :
        Axes on which to draw the plot.
    t : (k,) ndarray
        Time domain.
    rom_mean :
        Results from the mean posterior ROM (mean operators).
    rom_sample_mean :
        Mean of the ROM posterior (computed from samples).
    rom_deviation :
        Standard deviations of the ROM posterior.
    ndevs : int
        Number of standard deviations from the mean to shade in.
    gems : TODO or None
        GEMS data (truth).
    tf : float or None
        Final time. If given, plot a vertical line here.
    """
    if gems is not None:
        ax.plot(t, gems, ls="-", color=black, lw=1)
    if rom_mean is not None:
        ax.plot(t, rom_mean, ls="--", color="C5", lw=1)

    if rom_sample_mean is not None:
        ax.plot(t, rom_sample_mean, ls='-.', c="C0", lw=1)
        ax.fill_between(t,
                        rom_sample_mean - ndevs*rom_deviation,
                        rom_sample_mean + ndevs*rom_deviation,
                        color="C0", alpha=.5, lw=0, zorder=0)
    if tf is not None:
        ax.axvline(tf, color="k", lw=1)

    ax.set_xlim(t[0], t[-1])
    ax.locator_params(axis='y', nbins=2)
    ax.set_rasterization_zorder(1)


def plot_mode_uncertainty(trainsize, mean, draws, modes=4):
    """Plot reduced-order coordinate in time ± 3 standard deviations.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    mean : (r,k) ndarray
        Results of integrating the mean OpInf ROM.
    draws : (s,r,k) ndarray
        Results of integrating several posterior OpInf ROMs.
    modes : int
        Number of modes to plot.
    """
    steps = mean.shape[1]
    t = utils.load_time_domain(steps)

    if len(draws) > 0:
        with utils.timed_block("Processing sample draws"):
            deviations = np.std(draws, axis=0)
            sample_means = np.mean(draws, axis=0)
    else:
        deviations = [None]*modes

    fig, axes = plt.subplots(modes//2, 2, figsize=(12,6), sharex=True)
    for i, ax in zip(range(modes), axes.flat):
        _plot_rom_draws(ax, t,
                        mean[i], sample_means[i], deviations[i],
                        ndevs=3, tf=t[trainsize])
        ax.set_ylabel(fr"$\hat{{q}}_{{{i+1}}}(t)$")

    # for ax in axes[0]:
    #     ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
    #     ax.set_xticklabels(" "*(len(t)//10000 + 1))
    for ax in axes[-1]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xlabel("Time [s]")
    for j in range(axes.shape[1]):
        fig.align_ylabels(axes[:,j])
    for ax in axes[0]:
        ax.text(1/5, 1.025, "Training", fontsize="xx-large",
                transform=ax.transAxes, ha="center", va="bottom")
        ax.text(7/10, 1.025, "Prediction", fontsize="xx-large",
                transform=ax.transAxes, ha="center", va="bottom")

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    fig.subplots_adjust(hspace=.1, wspace=.2)
    patch = mpatches.Patch(facecolor=axes[0,0].lines[1].get_color(),
                           alpha=.5, linewidth=0)

    leg = axes[0,0].legend([axes[0,0].lines[0], (patch, axes[0,0].lines[1])],
                           ["BayesOpInf ROM mean",
                            "BayesOpInf solution "
                            r"sampling mean $\pm$ 3 stdevs"],
                           loc="lower center", ncol=2,
                           bbox_to_anchor=(.525,0),
                           bbox_transform=fig.transFigure)
    leg.get_lines()[0].set_linewidth(3)


def _get_pointtrace_data(trainsize, mean, draws, var="p"):
    if var not in ["p", "vx", "vy", "T"]:
        raise NotImplementedError(f"var='{var}'")

    # Get the indicies for each variable.
    elems = np.atleast_1d(config.MONITOR_LOCATIONS)
    elems = elems + config.ROM_VARIABLES.index(var)*config.DOF

    # Load the true traces and the time domain.
    traces_gems, t = utils.load_gems_data(rows=elems)
    steps = mean.shape[1]
    t = t[:steps]
    traces_gems = traces_gems[:,:steps]

    # Load the basis rows corresponding to the pressure traces.
    V, qbar, scales = utils.load_basis(trainsize, mean.shape[0])
    qbar = qbar[elems].reshape((-1,1))
    Velems = V[elems]

    # Reconstruct and rescale the simulation results.
    with utils.timed_block("Reconstructing simulation results"):
        # traces_rom_mean = dproc.unscale(Velems @ mean, scales, var) + qbar
        traces_rom_draws = [dproc.unscale(Velems @ draw, scales, var) + qbar
                            for draw in draws]

    with utils.timed_block("Processing samples"):
        sample_means = np.mean(traces_rom_draws, axis=0)
        deviations = np.std(traces_rom_draws, axis=0)

    return t, traces_gems, sample_means, deviations


def plot_pointtrace_uncertainty(trainsize, mean, draws, var="p"):
    """Plot mean point trace ± 3 standard deviations for a single variable
    at the monitoring locations.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    mean : (r,k) ndarray
        Results of integrating the mean OpInf ROM.
    draws : (s,r,k) ndarray
        Results of integrating several posterior OpInf ROMs.
    """
    (t, traces_gems,
     sample_means, deviations) = _get_pointtrace_data(trainsize,
                                                      mean, draws, var=var)

    fig, axes = plt.subplots(traces_gems.shape[0]//2, 2, figsize=(12,6))
    for i, ax in enumerate(axes.flat):
        _plot_rom_draws(ax, t,
                        None,  # traces_rom_mean[i],
                        sample_means[i], deviations[i],
                        ndevs=3, gems=traces_gems[i], tf=t[trainsize])
        ax.set_title(f"Location ${i+1}$")

    # Time label below lowest axis.
    for ax in axes[0]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xticklabels(" "*(len(t)//10000 + 1))
    for ax in axes[-1]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xlabel("Time [s]")

    # Set specific y limits.
    if var == "p":
        axes[0,0].set_ylim(8.5e5, 1.4e6)
        axes[0,1].set_ylim(9.5e5, 1.3e6)
        axes[1,0].set_ylim(8e5, 1.4e6)
        axes[1,1].set_ylim(9e5, 1.4e6)
    elif var == "vx":
        axes[0,0].set_ylim(-.04, .03)
        axes[0,1].set_ylim(-150, 200)
        axes[1,0].set_ylim(-6, 10)
        axes[1,1].set_ylim(-100, 200)

    # Single variable label on the left.
    ax_invis = fig.add_subplot(1, 1, 1, frameon=False)
    ax_invis.tick_params(labelcolor="none", bottom=False, left=False)
    ax_invis.set_ylabel(config.VARLABELS[var], labelpad=20)

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.subplots_adjust(hspace=.25, wspace=.15)
    gline, rline2 = axes[0,0].lines[:2]
    patch = mpatches.Patch(facecolor=rline2.get_color(), alpha=.5, linewidth=0)
    leg = axes[0,0].legend([gline, (patch, rline2)],
                           ["GEMS",
                            # "BayesOpInf ROM mean",
                            "BayesOpInf solution sampling mean "
                            r"$\pm$ 3 stdevs"],
                           loc="lower center", ncol=3,
                           bbox_to_anchor=(.525,0),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines()[:2]:
        line.set_linewidth(3)


def plot_pointtrace_timespread(trainsize, mean, draws, var="p"):
    """Plot absolute sample mean error and 3 standard deviations in time.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    mean : (r,k) ndarray
        Results of integrating the mean OpInf ROM.
    draws : (s,r,k) ndarray
        Results of integrating several posterior OpInf ROMs.
    """
    (t, traces_gems,
     sample_means, deviations) = _get_pointtrace_data(trainsize,
                                                      mean, draws, var=var)
    sample_errors = np.abs(traces_gems - sample_means)

    fig, axes = plt.subplots(traces_gems.shape[0]//2, 2, figsize=(12,6))
    for i, ax in enumerate(axes.flat):
        ax.semilogy(t, sample_errors[i], "C1", lw=1,
                    label="BayesOpInf ROM sample mean error")
        ax.semilogy(t, 3*deviations[i], "C0--", lw=1, ms=4, markevery=5000,
                    label="BayesOpInf ROM 3 sample standard deviations")
        ax.axvline(t[trainsize], color="k", lw=1)
        ax.set_title(f"Location ${i+1}$")
        ax.set_xlim(t[0], t[-1])
        if var == "p":
            ax.set_ylim(1e-2, 1e6)
        elif var == "vx":
            ax.set_ylim(5e-7, 5e2)

    # Time label below lowest axis.
    for ax in axes[0]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xticklabels(" "*(len(t)//10000 + 1))
    for ax in axes[-1]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xlabel("Time [s]")

    # Single variable label on the left.
    ax_invis = fig.add_subplot(1, 1, 1, frameon=False)
    ax_invis.tick_params(labelcolor="none", bottom=False, left=False)
    ax_invis.set_ylabel(config.VARLABELS[var], labelpad=20)

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.subplots_adjust(hspace=.25, wspace=.15)
    axes[0,0].legend(loc="lower center", ncol=3,
                     bbox_to_anchor=(.525,0), bbox_transform=fig.transFigure)


# =============================================================================

def main(trainsize, r, nmodes=4):
    """Construct posterior, make draws, and visualize results."""
    init_settings()
    utils.reset_logger(trainsize)

    # Load data.
    savefile = _plotdatafile(trainsize, r)
    if not os.path.isfile(savefile):
        raise FileNotFoundError(f"{savefile} (call generate_plot_data())")
    with h5py.File(savefile, 'r') as hf:
        if hf["meta/trainsize"][0] != trainsize:
            raise RuntimeError("inconsistent trainsize")
        if hf["meta/r"][0] != r:
            raise RuntimeError("inconsistent ROM dimension r")
        mean = hf["data/mean"][:]
        draws = hf["data/draws"][:]

    # Posterior modes with uncertainty.
    plot_mode_uncertainty(trainsize, mean, draws, nmodes)
    utils.save_figure(f"bayes/combustion_first{nmodes}modes.pdf")

    # Point traces with uncertainty.
    for var, label in [
        ("p", "pressure"),
        ("vx", "xvelocity"),
        ("vy", "yvelocity"),
        ("T", "temperature"),
    ]:
        plot_pointtrace_uncertainty(trainsize, mean, draws, var=var)
        utils.save_figure(f"bayes/combustion_traces_{label}.pdf")
        plot_pointtrace_timespread(trainsize, mean, draws, var=var)
        utils.save_figure(f"bayes/combustion_timeVspread_{label}.pdf")


# =============================================================================
if __name__ == "__main__":
    # Experiment parameters.
    trainsize = 20000
    r = 38
    regs = (7845.530230529863, 17575.683334267986)
    ndraws = 100
    nsteps = 50000

    # Generate data (do this only once).
    # generate_plot_data(trainsize, r, regs, ndraws, nsteps)

    # Load and visualize data.
    main(trainsize, r, nmodes=6)
