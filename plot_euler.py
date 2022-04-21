# plot_euler.py
"""Plots for the noisy 1D Euler BayesOpInf example."""
import os
import h5py
import itertools
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patches as mpatches

import rom_operator_inference as opinf

import config
import utils
import euler1D
import bayes


DATAFILENAME = "bayes_data.h5"
black = (.2, .2, .2)


def init_settings():
    """Turn on custom matplotlib settings."""
    plt.rc("figure", figsize=(12,4))
    plt.rc("axes", titlesize="xx-large", labelsize="xx-large", linewidth=.5)
    plt.rc("xtick", labelsize="large")
    plt.rc("ytick", labelsize="large")
    plt.rc("legend", fontsize="xx-large", frameon=False, edgecolor="none")


# Data generation =============================================================

def _resultsgroup(level, r):
    return f"{config.NOSFMT(level)}/{config.DIMFMT(r)}"


def _train_predict_error(solver, ktrain, rom_approx, fom_truth):
    """Calculate the error of a low-dimensional ROM trajectory
    as the average relative error in each learning variable.

    Parameters
    ----------
    solver : euler1D.EulerROMSolver
    Vr : (3n, r) ndarray
        Reduced-order model basis matrix (POD).
    ktrain : int
        Number of snapshots in the training data (size of training set).
    rom_approx : (3n, k) ndarray
        Full-order reconstruction of time-integrated ROM state.
    fom_truth : (3n, k) ndarray
        True full-order state.
    """
    t_train, t_pred = np.split(solver.t, [ktrain], axis=0)
    rom_train, rom_predict = np.split(rom_approx, [ktrain], axis=1)
    fom_train, fom_predict = np.split(fom_truth, [ktrain], axis=1)
    train_error, pred_error = 0, 0
    for fomvar, romvar in zip(np.split(fom_train, 3, axis=0),
                              np.split(rom_train, 3, axis=0)):
        train_error += opinf.post.Lp_error(fomvar, romvar, t_train)[1]
    for fomvar, romvar in zip(np.split(fom_predict, 3, axis=0),
                              np.split(rom_predict, 3, axis=0)):
        pred_error += opinf.post.Lp_error(fomvar, romvar, t_pred)[1]
    return train_error/3, pred_error/3


def _error_foreach(solver, rom, ktrain, level):
    """Calculate the relative ROM error for each solution trajectory,
    separated into training/prediction phases.

    Parameters
    ----------
    solver : euler1D.EulerROMSolver
    rom : opinf.InferredContinuousROM
    ktrain : int
        Number of snapshots in the training data (size of training set).
    level : float
        Ratio of variable range to the noise standard deviation,
        e.g., level=.01 means stdev = 1% of max(variable) - min(variable).

    Returns
    -------
    training_errors : list of length s
        Relative ROM errors over the training regime.
    prediction_errors : list of length s
        Relative ROM errors over the prediction regime.
    """
    training_errors, prediction_errors = [], []
    inits = [U[:,0] for U in solver.apply_noise(level)]   # Noisy ICs.
    for icn, u0 in enumerate(inits):
        print(f"Initial condition {icn+1}/{len(solver)}")
        init_ = rom.Vr.T @ solver.nondimensionalize(u0)
        rom_out = euler1D.rom_predict(rom, init_, solver.t)
        rom_approx = solver.redimensionalize(rom.Vr @ rom_out)

        # Calculate error relative to the true full-order solution.
        err1, err2 = _train_predict_error(solver, ktrain,
                                          rom_approx, solver.snapshots[icn])
        training_errors.append(err1)
        prediction_errors.append(err2)

    return training_errors, prediction_errors


def generate_training_data():
    """Generate the full-order snapshot set that uses the corners of the
    initial condition space (six degrees of freedom).

    Existing data files must be deleted first, or an error will be raised.
    """
    solver = euler1D.EulerROMSolver(**config.EULER_DOMAIN)
    rho, u = [20, 24], [95, 105]
    for i,init_params in enumerate(itertools.product(rho, rho, rho, u, u, u)):
        print(f"Trial {i+1:02d}/64", end=' ')
        solver.add_snapshot_set(init_params, plot_init=False)
    solver.save(config.EULER_SNAPSHOT_DATA, overwrite=False)

    return solver


def generate_regopinf_results(solver, trainsize, level, r):
    """Generate results for OpInf with error-based regularization selection.

    Parameters
    ----------
    solver : euler1D.EulerROMSolver
        Object for doing full-order solves and handling noising.
        Must already have one set of snapshots / noise.
    trainsize : int
        Number of training snapshots.
    level : float
        Ratio of variable range to the noise standard deviation,
        e.g., level=.01 means stdev = 1% of max(variable) - min(variable).
    r : int
        Number of POD modes to retain (size of the ROM).
    """
    if os.path.isfile(config.EULER_RESULTS):
        gp = _resultsgroup(level, r)
        with h5py.File(config.EULER_RESULTS) as hf:
            if gp in hf and "bayesopinf_meanrom_train" in hf[gp]:
                raise ValueError("experiment result exists")
    print(f"\n\n*** REGOPINF EXPERIMENT: noise = {level}, r = {r} ***\n")

    # Get regOpInf ROM and evaluate it at the particular initial condition.
    rom = solver.train_rom(r, noise_level=level, margin=1.1, ktrain=trainsize)

    # Get relative errors of the ROM at each initial condition.
    training, prediction = _error_foreach(solver, rom, trainsize, level)

    # Save relative errors.
    with h5py.File(config.EULER_RESULTS, 'a') as hf:
        gp = hf.require_group(config.NOSFMT(level))
        gp = gp.require_group(config.DIMFMT(r))
        gp.create_dataset("regopinf_train", data=training)
        gp.create_dataset("regopinf_predict", data=prediction)

    return training, prediction


def generate_bayesopinf_results(solver, trainsize, level, r,
                                reg0=50, tol=.001, testloc=None, ndraws=100):
    """Generate results for BayesOpInf.

    Parameters
    ----------
    solver : euler1D.EulerROMSolver
        Object for doing full-order solves and handling noising.
        Must already have one set of snapshots / noise.
    trainsize : int
        Number of training snapshots.
    level : float
        Ratio of variable range to the noise standard deviation,
        e.g., level=.01 means stdev = 1% of max(variable) - min(variable).
    r : int
        Number of POD modes to retain (size of the ROM).
    reg0 : float > 0
        Initial regularization hyperparameter (IC for FPI).
    tol : float > 0
        Convergence tolerance for the Bayesian regularization update.
    testloc : int
        Index of the initial condition to do MC sampling for UQ.
    ndraws : int
        Number of simulation samples to draw.
    """
    if os.path.isfile(config.EULER_RESULTS):
        gp = _resultsgroup(level, r)
        with h5py.File(config.EULER_RESULTS) as hf:
            if gp in hf and "bayesopinf_meanrom_train" in hf[gp]:
                raise ValueError("experiment result exists")
            if testloc is not None and gp in hf and "particular" in hf[gp]:
                raise ValueError("experiment result exists")
    print(f"\n\n*** BAYESOPINF EXPERIMENT: noise = {level}, r = {r} ***\n")

    # Get initial ROM at fixed initial reg.
    rom = solver.train_rom(r, noise_level=level, ktrain=trainsize, reg=reg0)

    # Do Bayesian fixed-point iteration to determine the regularization.
    post, reg = bayes.construct_posterior(rom, rom.reg, case=2)
    Q_ = rom._training_states_
    R_ = rom.solver_.B.T
    D = rom.solver_.A
    gs = la.eigvalsh(D.T @ D)
    rom.fit(rom.Vr, Q_, R_, P=reg)  # Refit so new regularization type works.

    for _ in range(50):
        # Update posterior.
        post, regnew = bayes.construct_posterior(rom, reg, case=2, gs=gs)

        # Compute relative change in the regularization hyperparameters.
        diff = la.norm(regnew - reg) / la.norm(reg)
        print(f"Regularization iteration: diff = {diff:.2e}")
        if diff < tol:
            break
        reg = regnew

    # Get relative errors of the mean BayesOpInf ROM at each initial condition.
    meanrom = post.mean_rom
    meanrom.Vr = rom.Vr
    training, prediction = _error_foreach(solver, meanrom, trainsize, level)

    # Get Bayesian ROM draws and FOM solutions for one initial condition.
    if testloc is not None:
        snaps_noised = solver.apply_noise(level)[testloc]
        init_ = rom.Vr.T @ solver.nondimensionalize(snaps_noised[:,0])
        _, draws_ = bayes.simulate_posterior(post, init_,
                                             solver.t, ndraws=ndraws)
        draws = [solver.redimensionalize(rom.Vr @ U) for U in draws_]

    # Save relative errors.
    with h5py.File(config.EULER_RESULTS, 'a') as hf:
        gp = hf.require_group(config.NOSFMT(level))
        gp = gp.require_group(config.DIMFMT(r))
        gp.create_dataset("bayesopinf_meanrom_train", data=training)
        gp.create_dataset("bayesopinf_meanrom_predict", data=prediction)
        if testloc is not None:
            gp = gp.create_group("particular")
            gp.create_dataset("draws", data=draws)
            gp.create_dataset("snapshots", data=solver.snapshots[testloc])
            gp.create_dataset("snapshots_noised", data=snaps_noised)

    return training, prediction


# Plotting ====================================================================

def plot_training_data(solver, trainsize=1000, testloc=2, nlocs=10,
                       levels=(.001, .01, .1), jmax=25):
    """Plot training data and the singular value decay of the training data."""
    # Get full-order and noised snapshots.
    upzeta = solver.snapshots[testloc]

    # Initialize figure.
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(12,4),
                             gridspec_kw=dict(width_ratios=[3, 2]))
    toerase = axes[:,1]
    axes = axes[:,0]
    gs = toerase[0].get_gridspec()
    for ax in toerase:
        ax.remove()
    axbig = fig.add_subplot(gs[:,1])

    # Set up colors.
    xlocs = np.linspace(0, solver.n, nlocs+1, dtype=np.int)[:-1]
    xlocs += xlocs[1]//2
    cmap = plt.cm.twilight
    colors = cmap(np.linspace(0, .95, solver.n)[:-1][xlocs])

    # Plot snapshot variables.
    u, p, zeta = np.split(upzeta, solver.NUM_VARIABLES)
    for j, c in zip(xlocs, colors):
        for i, var in enumerate([u, p, 1/zeta]):
            axes[i].plot(solver.t, var[j], color=c, lw=1)

    # Plot singular values.
    indices = np.arange(1, jmax+2)
    snapshots = np.hstack([solver.nondimensionalize(U[:,:trainsize])
                           for U in solver.snapshots])
    svals = la.svdvals(snapshots)
    svals /= svals[0]
    axbig.semilogy(indices, svals[:jmax+1],
                   "*-", color=black, ms=7, mew=0, lw=.5, label="Noiseless")
    axbig.text(jmax-10, 1.5*svals[jmax-10],
               "noiseless", color=black, ha="left")
    axbig.set_ylim(svals[jmax]/2, 2*svals[0])

    # Plot singular values for each noise level.
    for i, level in enumerate(levels):
        snapshots_noised = np.hstack([solver.nondimensionalize(U[:,:trainsize])
                                      for U in solver.apply_noise(level)])
        svals = la.svdvals(snapshots_noised)
        svals /= svals[0]
        axbig.semilogy(indices, svals[:jmax+1],
                       f".-C{i:d}", ms=7, mew=0, lw=.5,
                       label=fr"${level*100:.0f}\%$ noise")
        axbig.text(jmax-10, 1.25*svals[jmax-10],
                   fr"${level*100:.0f}\%$ noise",
                   color=f"C{i:d}", ha="left", va="bottom")

    # Format axes.
    axes[0].set_ylabel(r"Velocity [m/s]", fontsize="medium")
    axes[1].set_ylabel(r"Pressure [Pa]", fontsize="medium")
    axes[2].set_ylabel(r"Density [kg/m$^3$]", fontsize="medium")
    for ax in axes:
        ax.set_xlim(solver.t[0], solver.t[-1])
        ax.set_xticks([0, .010, .020, .030])
        ax.axvline(solver.t[1000], color="k", lw=1)
    axes[-1].set_xlabel("Time [s]", fontsize="large")
    # axes[0].set_title("Training Snapshots", fontsize="large")
    axes[0].text(1/6, 1.025, "Training", fontsize="large",
                 transform=axes[0].transAxes, ha="center", va="bottom")
    axes[0].text(2/3, 1.025, "Prediction", fontsize="large",
                 transform=axes[0].transAxes, ha="center", va="bottom")

    axbig.set_xlim(0, jmax+.5)
    axbig.set_xlabel(r"Singular value index", fontsize="large")
    axbig.set_ylabel(r"Singular value (normalized)", fontsize="large")
    # axbig.legend(loc="upper right", ncol=2)

    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=.15, wspace=.25)

    # Colorbar.
    lsc = cmap(np.linspace(0, .95, 400))
    scale = mplcolors.Normalize(vmin=0, vmax=1)
    lscmap = mplcolors.LinearSegmentedColormap.from_list("euler", lsc, N=nlocs)
    mappable = plt.cm.ScalarMappable(norm=scale, cmap=lscmap)
    cbar = fig.colorbar(mappable, ax=axes, pad=0.02)
    cbar.set_ticks(solver.x[xlocs] / solver._L)
    cbar.set_ticklabels([f"{x:.2f}" for x in solver.x[xlocs]])
    cbar.set_label(r"Spatial coordinate $x$", fontsize="large")

    utils.save_figure("bayes/euler_training.pdf")


def plot_projected_training_data(solver, level, ktrain=1000, loc=42):
    """Plot the first few projected (noisy) snapshots."""
    Us = [solver.nondimensionalize(U[:,:ktrain])
          for U in solver.apply_noise(level)]
    Vr = la.svd(np.hstack(Us), full_matrices=False)[0][:,:15]
    U_ = Vr.T @ Us[loc]
    tt = solver.t[:ktrain]

    fig, axes = plt.subplots(4, 3, figsize=(12,4), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(tt, U_[i] / np.abs(U_[i]).max(), lw=1)
        ax.set_ylabel(fr"$\hat{{q}}_{{{i+1}}}(t)$")

    # Format axes.
    for ax in axes.flat:
        ax.set_xticks([0, .003, .006, .009])
        ax.set_xlim(tt[0], tt[-1])
        ax.set_ylim(-1.1, 1.1)
    fig.subplots_adjust(wspace=.3)
    for j in range(axes.shape[1]):
        axes[-1,j].set_xlabel("Time [s]")
        fig.align_ylabels(axes[:,j])

    utils.save_figure("bayes/euler_projected_training_data.pdf")


def plot_point_traces(solver, level, r, trainsize=1000, nlocs=3, case=2):
    """Plot pointwise Bayes results."""
    # Load results.
    with h5py.File(config.EULER_RESULTS, 'r') as hf:
        gp = hf[f"{_resultsgroup(level, r)}/particular"]
        snapshots = gp["snapshots"][:]
        snapshots_noised = gp["snapshots_noised"][:]
        draws = gp["draws"][:]

    # Process uncertainty band.
    post_sample_mean = np.mean(draws, axis=0)
    post_sample_stds = np.std(draws, axis=0)
    post_low = post_sample_mean - 3*post_sample_stds
    post_high = post_sample_mean + 3*post_sample_stds

    # Initialize figure and colors.
    fig, axes = plt.subplots(3, nlocs, sharex=True, sharey="row",
                             figsize=(12, 5))
    xlocs = np.linspace(0, solver.n, nlocs+1, dtype=np.int)[:-1]
    xlocs += xlocs[1]//2

    # Plot variables.
    for j, loc in enumerate(xlocs):
        for i in range(solver.NUM_VARIABLES):
            idx = loc + i*solver.n
            if i == 2:
                axes[i,j].plot(solver.t, 1/snapshots[idx],
                               ls="-", color=black, lw=1)
                axes[i,j].plot(solver.t[:trainsize],
                               1/snapshots_noised[idx,:trainsize],
                               ".", color=black, mew=0, ms=1.5)
                # axes[i,j].plot(solver.t, 1/post_mean[idx], "C9-.", lw=2)
                axes[i,j].plot(solver.t, 1/post_sample_mean[idx], "C0-.", lw=1)
                axes[i,j].fill_between(solver.t,
                                       1/post_high[idx], 1/post_low[idx],
                                       color="C0", alpha=.5, lw=0, zorder=0)
            else:
                axes[i,j].plot(solver.t, snapshots[idx],
                               ls="-", color=black, lw=1)
                axes[i,j].plot(solver.t[:trainsize],
                               snapshots_noised[idx,:trainsize],
                               ".", color=black, mew=0, ms=1.5)
                # axes[i,j].plot(solver.t, post_mean[idx], "C9-.", lw=2)
                axes[i,j].plot(solver.t, post_sample_mean[idx], "C0-.", lw=1)
                axes[i,j].fill_between(solver.t, post_low[idx], post_high[idx],
                                       color="C0", alpha=.5, lw=0, zorder=0)
        xval = round(solver.x[loc],2)
        if abs(xval - 1) < .015:
            xval = 1
        axes[0,j].set_title(fr"$x = {xval:.2f}$")

    # Format axes.
    for ax in axes.flat:
        ax.axvline(solver.t[trainsize], color="k", lw=1)
        ax.set_xlim(solver.t[0], solver.t[-1])
        if nlocs < 3:
            ax.set_xticks([0, .004, .008, .012, .016, .020, .024, .028])
        else:
            ax.set_xticks([0, .008, .016, .024])
        # ax.locator_params(axis='y', nbins=2)
        ax.set_rasterization_zorder(1)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    axes[0,0].set_ylabel("Velocity\n[m/s]")
    axes[1,0].set_ylabel("Pressure\n[Pa]")
    axes[2,0].set_ylabel("Density\n[kg/m$^3$]")
    fig.align_ylabels(axes[:,0])

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .1, 1, 1])
    fig.subplots_adjust(hspace=.15, wspace=.05)
    dline, pts, rline2 = axes[0,0].lines[:3]
    patch = mpatches.Patch(facecolor=rline2.get_color(), alpha=.5, linewidth=0)
    leg = axes[0,0].legend([dline,
                            axes[0,0].plot([], [], 'o', color=black, ms=3)[0],
                            (patch, rline2)],
                           ["Truth",
                            "Observations",
                            "BayesOpInf solution "
                            r"sampling mean $\pm$ 3 stdevs"],
                           loc="lower center", ncol=4,
                           bbox_to_anchor=(.525,0),
                           bbox_transform=fig.transFigure)
    for i, line in enumerate(leg.get_lines()):
        if i != 1:
            line.set_linewidth(2)

    outfile = '_'.join(["bayes/euler", "traces", config.NOSFMT(level),
                        config.DIMFMT(r), f"case{case:d}"]) + ".pdf"
    utils.save_figure(outfile)


def plot_error_vs_basis_size(solver, level, rs):
    """Basis size versus ROM error."""
    containers = {
        "meanrom_train": [],
        "meanrom_predict": [],
        "regopinf_train": [],
        "regopinf_predict": [],
    }

    # Load results.
    with h5py.File(config.EULER_RESULTS, 'r') as hf:
        for r in rs:
            gp = hf[_resultsgroup(level, r)]
            containers["meanrom_train"].append(
                np.mean(gp["bayesopinf_meanrom_train"][:], axis=0))
            containers["meanrom_predict"].append(
                np.mean(gp["bayesopinf_meanrom_predict"][:], axis=0))
            containers["regopinf_train"].append(
                np.mean(gp["regopinf_train"][:], axis=0))
            containers["regopinf_predict"].append(
                np.mean(gp["regopinf_predict"][:], axis=0))

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharex=True, sharey=True)
    for j, tp in enumerate(["train", "predict"]):
        for name, ls, c, label in [
            ("meanrom", ".-", "C9", "BayesOpInf ROM mean"),
            # ("samplemean", ".-.", "C0", "BayesOpInf solution sampling mean"),
            ("regopinf", ".--", "C5", "OpInf ROM")
        ]:
            axes[j].plot(rs, containers[f"{name}_{tp}"], ls,
                         color=c, mew=0, ms=10, lw=1, label=label)
            # axes[j].fill_between(rs, dmin, dmax, color=c, lw=0, alpha=.5)

    axes[0].set_yticks([.01, .02, .03, .04, .05])
    axes[0].set_yticklabels([r"1\%", r"2\%", r"3\%", r"4\%", r"5\%"])
    for ax in axes:
        ax.set_xlim(rs[0] - .5, rs[-1] + .5)
        ax.set_xlabel(r"ROM dimension $r$", fontsize="x-large")
        ax.set_ylim(0, .055)
        ax.grid(axis='y', lw=.5, color="gray")
    axes[0].set_ylabel("Relative error", fontsize="x-large")
    axes[0].set_title("Training")
    axes[1].set_title("Prediction")
    axes[0].legend(loc="upper right")

    # Single legend below the subplots.
    fig.tight_layout(rect=[0, .15, 1, 1])
    fig.subplots_adjust(wspace=.05)
    leg = axes[0].legend(loc="lower center", ncol=3,
                         bbox_to_anchor=(.525,0),
                         bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)
        line.set_markersize(20)

    savefile = "_".join(["bayes/euler", "basisVerror",
                         config.NOSFMT(level), "case2"]) + ".pdf"
    utils.save_figure(savefile)


# =============================================================================

def main():
    """Make all plots."""
    # Generate data -----------------------------------------------------------

    # Full-order training data.
    if not os.path.isfile(config.EULER_SNAPSHOT_DATA):
        generate_training_data()
    solver = euler1D.EulerROMSolver.load(config.EULER_SNAPSHOT_DATA)

    # BayesOpInf / RegOpInf results.
    if not os.path.isfile(config.EULER_RESULTS):
        for r in range(3, 21):
            generate_regopinf_results(solver, trainsize=1000, level=.05, r=r)
            if r == 9:
                generate_bayesopinf_results(solver, trainsize=1000, level=.05,
                                            r=r, reg0=50, tol=.001,
                                            testloc=42, ndraws=100)
            else:
                generate_bayesopinf_results(solver, trainsize=1000, level=.05,
                                            r=r, reg0=50, tol=.001)

    # Make plots --------------------------------------------------------------
    init_settings()

    # Figure 2: Training data + singular value decay.
    plot_training_data(solver, trainsize=1000,
                       testloc=42, nlocs=10, levels=(.01, .05), jmax=30)

    # Figure 3: Projected noisy training data.
    plot_projected_training_data(solver, level=.05, ktrain=1000, loc=42)

    # Figure 4: Point traces.
    plot_point_traces(solver, level=.05, r=9, nlocs=3)

    # Figure 5: Error vs. basis size.
    plot_error_vs_basis_size(solver, .05, rs=list(range(3, 21)))


# =============================================================================
if __name__ == "__main__":
    utils.reset_logger("euler")
    main()
