# bayes.py
"""Bayesian Operator Inference for this problem."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
import utils
import step4_plot as step4
import data_processing as dproc
import bayes


DATAFILENAME = "bayes_data.h5"


def init_settings():
    """Turn on custom matplotlib settings."""
    plt.rc("figure", figsize=(12,4))
    plt.rc("axes", titlesize="xx-large", labelsize="xx-large", linewidth=.5)
    plt.rc("xtick", labelsize="large")  # "x-large" ?
    plt.rc("ytick", labelsize="large")  # "x-large" ?
    plt.rc("legend", fontsize="xx-large", frameon=False, edgecolor="none")


def _plotdatafile(trainsize, r):
    savefolder = config._makefolder(config.BASE_FOLDER,
                                    config.TRNFMT(trainsize),
                                    config.DIMFMT(r))
    return os.path.join(savefolder, DATAFILENAME)


def generate_plot_data(trainsize, r, reg, ndraws, steps, overwrite=False):
    """Generate data to be visualized.

    Parameters
    ----------
    TODO
    """
    # Construct the name of the file to export data to.
    savefile = _plotdatafile(trainsize, r)
    if os.path.isfile(savefile) and not overwrite:
        raise FileExistsError(f"{savefile} (set overwrite=True to ignore)")

    # Run numerical experiment.
    post = bayes.construct_posterior(trainsize, r, reg, case=-1)
    mean, draws = bayes.simulate_posterior(trainsize, post, ndraws, steps)

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

def _plot_rom_draws(ax, t, rom_mean, rom_deviation, ndevs=3,
                    gems=None, tf=None):
    """Plot ROM mean ± `ndevs` standard deviations.

    Parameters
    ----------
    ax :
        Axes on which to draw the plot.
    t : (k,) ndarray
        Time domain.
    rom_mean :
        Mean of the ROM posterior.
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
        ax.plot(t, gems, "C1-", lw=1.5, label="GEMS")
    ax.plot(t, rom_mean, 'C0--', lw=1.5, label="BayesOpInf ROM (mean)")
    # for draw in traces_rom_draws:
    #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.25)
    if rom_deviation is not None:
        ax.fill_between(t,
                        rom_mean - ndevs*rom_deviation,
                        rom_mean + ndevs*rom_deviation,
                        alpha=.5, zorder=0,
                        label=("BayesOpInf ROM "
                               fr"(mean $\pm$ {ndevs:d} standard deviations)"))
    if tf is not None:
        ax.axvline(tf, color='k', lw=1)

    ax.set_xlim(t[0], t[-1])
    ax.locator_params(axis='y', nbins=2)
    ax.set_rasterization_zorder(1)


def plot_mode_uncertainty(trainsize, mean, draws, modes=4):
    """Plot reduced-order coordinate in time ± 3 standard deviations.

    Parameters
    ----------
    trainsize : int
        TODO
    mean :
        TODO
    draws :
        TODO
    modes : int
        Number of modes to plot.
    """
    steps = mean.shape[1]
    t = utils.load_time_domain(steps)

    if len(draws) > 0:
        with utils.timed_block("Calculating sample deviations"):
            deviations = np.std(draws, axis=0)
    else:
        deviations = [None]*modes

    fig, axes = plt.subplots(modes//2, 2, figsize=(12,6))
    for i, ax in zip(range(modes), axes.flat):
        _plot_rom_draws(ax, t, mean[i], deviations[i], 3, tf=t[trainsize])
        ax.set_ylabel(fr"$\hat{{q}}_{{{i+1}}}(t)$")

    for ax in axes[0]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xticklabels(" "*(len(t)//10000 + 1))
    for ax in axes[-1]:
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_xlabel(r"Time [s]")
    for j in range(axes.shape[1]):
        fig.align_ylabels(axes[:,j])

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    fig.subplots_adjust(hspace=.1, wspace=.2)
    patch = mpatches.Patch(facecolor="C0", alpha=.5, linewidth=0)
    axes[0,0].legend([(patch, axes[0,0].lines[0])],
                     [r"BayesOpInf ROM (mean $\pm$ 3 standard deviations)"],
                     loc="lower center", ncol=2,
                     bbox_to_anchor=(.5,0), bbox_transform=fig.transFigure)


def plot_pointtrace_uncertainty(trainsize, mean, draws, var="p"):
    """Plot mean point trace ± 3 standard deviations for a single variable
    at the monitoring locations.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    mean :
        TODO
    draws :
        TODO
    """
    if var not in ["p", "vx", "vy", "T"]:
        raise NotImplementedError(f"var='{var}'")

    # Get the indicies for each variable.
    elems = np.atleast_1d(config.MONITOR_LOCATIONS)
    nelems = elems.size
    elems = elems + config.ROM_VARIABLES.index(var)*config.DOF

    # Load the true pressure traces and the time domain.
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
        traces_rom_mean = dproc.unscale(Velems @ mean, scales, var) + qbar
        traces_rom_draws = [dproc.unscale(Velems @ draw, scales, var) + qbar
                            for draw in draws]

    with utils.timed_block("Calculating sample deviations"):
        deviations = np.std(traces_rom_draws, axis=0)

    fig, axes = plt.subplots(nelems//2, 2, figsize=(12,6))
    for i, ax in enumerate(axes.flat):
        _plot_rom_draws(ax, t, traces_rom_mean[i], deviations[i], ndevs=3,
                        gems=traces_gems[i], tf=t[trainsize])
        ax.set_title(f"Location ${i+1}$")

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
    fig.subplots_adjust(hspace=.2, wspace=.1)
    patch = mpatches.Patch(facecolor="C0", alpha=.5, linewidth=0)
    gline, rline = axes[0,0].lines[:2]
    axes[0,0].legend([gline, (patch, rline)],
                     ["GEMS",
                      r"BayesOpInf ROM (mean $\pm$ 3 standard deviations)"],
                     loc="lower center", ncol=3,
                     bbox_to_anchor=(.525,0), bbox_transform=fig.transFigure)


def plot_speciesintegral_uncertainty(trainsize, mean, draws):
    """Plot species integral ± standard deviations.

    Parameters
    ----------
    trainsize : int
        TODO
    mean : (r,k) ndarray
        TODO
    draws : (s,r,k) ndarray?
        TODO
    """
    # Load the true spatial statistics and the time domain.
    steps = mean.shape[1]
    keys = ["CH4_sum", "O2_sum", "H2O_sum", "CO2_sum"]
    integrals_gems, t = utils.load_spatial_statistics(keys, steps)

    # Load the basis.
    V, qbar, scales = utils.load_basis(trainsize, mean.shape[0])

    fig, axes = plt.subplots(len(keys), 1, figsize=(9,9), sharex=True)
    for key, ax in zip(keys, axes.flat):
        var, action = key.split('_')
        with utils.timed_block(f"Reconstructing {key}"):
            integral_rom_mean = step4.get_feature(key, mean, V, qbar, scales)
            integral_rom_draws = [step4.get_feature(key, draw, V, qbar, scales)
                                  for draw in draws]
            deviation = np.std(integral_rom_draws, axis=0)
        _plot_rom_draws(ax, t, integral_rom_mean, deviation, ndevs=3,
                        gems=integrals_gems[key], tf=t[trainsize])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .001))
        ax.set_title(config.VARTITLES[var])

    # Time label below lowest axis.
    axes[-1].set_xlabel("Time [s]")

    # Single variable label on the left.
    ax_invis = fig.add_subplot(111, frameon=False)
    ax_invis.tick_params(labelcolor="none", bottom=False, left=False)
    ax_invis.set_ylabel(f"Specied Concentration Integrals "
                        f"[{config.VARUNITS[var]}]",
                        labelpad=20)

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.subplots_adjust(hspace=.225)
    leg = axes[0].legend(loc="lower center", ncol=3,
                         bbox_to_anchor=(.525,0),
                         bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)


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
    utils.save_figure(f"bayes/bayes_first{nmodes}modes.pdf")

    # Point traces with uncertainty.
    for var, label in [
        ("p", "pressure"),
        ("vx", "xvelocity"),
        ("vy", "yvelocity"),
        ("T", "temperature"),
    ]:
        plot_pointtrace_uncertainty(trainsize, mean, draws, var=var)
        utils.save_figure(f"bayes/bayes_traces_{label}.pdf")

    # Species integrals with uncertainty (expensive!).
    # plot_speciesintegral_uncertainty(trainsize, mean, draws)
    # utils.save_figure("bayes/bayes_species_integrals.pdf")


# TODO: see if this can be salvaged.
def iterate(trainsize, r, reg, niter, case=2):
    """Do the iteration several times, plotting the evolution thereof."""
    utils.reset_logger(trainsize)
    print(f"Initialization: reg = {reg}")
    means = np.empty(niter+1, dtype=float)
    stds = means.copy()
    means[0], stds[0] = reg**2, 0
    iterations = np.arange(niter+1)
    for n in iterations[1:]:
        post, reg = bayes.construct_posterior(trainsize, r, reg, case=case)
        print(f"Iteration {n}: reg = {reg}")
        means[n], stds[n] = np.mean(reg**2), np.std(reg**2)
    print("Relative change in mean at final update:",
          f"{abs(means[-1] - means[-2]) / abs(means[-1]):%}")

    # Plot progression of regularization statistics.
    plt.semilogy(iterations, means, 'C0.-', ms=10)
    plt.fill_between(iterations, means-stds, means+stds,
                     color="C0", alpha=.5)
    plt.xlabel("Bayes Iteration")
    plt.ylabel(r"Regularization $\lambda$ ($\mu \pm \sigma$)")
    plt.title("Iterative Bayesian Regularization Update: Combustion")
    plt.xlim(right=niter)
    utils.save_figure(f"bayes/iteration_case{case}.pdf")

    # Try simulating the final model.
    mean, draws = bayes.simulate_posterior(trainsize, post, 0, trainsize)
    plot_mode_uncertainty(trainsize, mean, draws, 4)
    utils.save_figure(f"bayes/iter{case}_first4modes.pdf")


# =============================================================================
if __name__ == "__main__":
    # Experiment parameters.
    trainsize = 20000
    r = 38
    regs = (7845.530230529863, 17575.683334267986)
    ndraws = 100
    nsteps = 50000

    # Generate data (do this once).
    # generate_plot_data(trainsize, r, regs, ndraws, nsteps, overwrite=True)

    # Load and visualize data.
    main(trainsize, r, nmodes=4)

    # iterate(20000, 40, 36382, 15, case=2)
