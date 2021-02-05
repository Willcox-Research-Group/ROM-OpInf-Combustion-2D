# plots.py
"""Generate the non-tecplot-based plots from the paper
"Data-driven reduced-order models via regularised Operator Inference
for a single-injector combustion process"
by Shane A. McQuarrrie, Cheng Huang, and Karen E. Willcox.
© 2021
"""

import re
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
import data_processing as dproc
import step4_plot as step4
import poddeim


# Font sizes for time traces, mean temperatures, and species integrals.
BIGFONT = 36
MEDFONT = 28
SMLFONT = 20
TNYFONT = 18


# Computational domain ========================================================

def domain(fill="C0", point="C1", note="C3", extension="pdf"):
    # Read domain info from the grid file.
    grid_path = config.grid_data_path()
    with open(grid_path, 'r') as infile:
        grid = infile.read()
    if int(re.findall(r"Elements=(\d+)", grid)[0]) != config.DOF:
        raise RuntimeError(f"{grid_path} DOF and config.DOF do not match")
    num_nodes = int(re.findall(r"Nodes=(\d+)", grid)[0])
    end_of_header = re.findall(r"DT=.*?\n", grid)[0]
    headersize = grid.find(end_of_header) + len(end_of_header)

    # Extract geometry information.
    grid_data = grid[headersize:].split()
    x = np.array(grid_data[:num_nodes], dtype=np.float)
    y = np.array(grid_data[num_nodes:2*num_nodes], dtype=np.float)

    # Figure out the boundary.
    xmin = x.min()
    ymid1 = y[x == xmin].max()
    ymid = y[np.argmin(np.abs(x + .005))]
    xmid = x[y == ymid].min()
    ymax = y.max()

    fig, ax = plt.subplots(1, 1, figsize=(9,5), dpi=1200)

    ax.fill_between([0, .1], y.max(), y.min(), color=fill)
    ax.fill_between([xmin, 0], ymid1, color=fill)
    ax.fill_between([-.01, 0], ymid, color=fill)
    ax.fill_between([xmid, 0], ymid, ymid-.0004, color=fill)

    ax.set_aspect("equal")
    ax.set_xlim(-.08, .1)
    ax.set_ylim(0, .025)
    ax.set_xticks(np.linspace(-.06, .1, 9))
    ax.set_yticks([0, .01, .02])
    ax.set_xlabel(r"$x$ [m]", fontsize=14)
    ax.set_ylabel(r"$y$ [m]", fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(.5)
    ax.spines["bottom"].set_linewidth(.5)
    ax.spines["bottom"].set_position("zero")

    plt.text(-.0725, .004, "Oxidizer", color=note, fontsize=14)
    plt.arrow(-.055, .005, .01, 0, color=note,
              head_width=0.004, head_length=0.004, linewidth=.0005)
    plt.text(-.065, .011, "Fuel", color=note, fontsize=14)
    plt.arrow(-.055, ymid, .01, 0, color=note,
              head_width=0.004, head_length=0.004, linewidth=.0005)

    plt.plot([0], [ymax], marker='.', ms=10, c=point)
    plt.text(0+.002, ymax+.001, "Location 1", fontsize=12)
    plt.plot([.06], [ymax], marker='.', ms=10, c=point)
    plt.text(.06+.002, ymax+.001, "Location 2", fontsize=12)
    plt.plot([0], [.01225], marker='.', ms=10, c=point)
    plt.text(0+.002, .013+.001, "Location 3", fontsize=12, color='w')
    plt.plot([-.02], [ymid1], marker='.', ms=10, c=point)
    plt.text(-0.02+.002, ymid1-.004, "Location 4", fontsize=12, color='w')

    fig.tight_layout()
    utils.save_figure(f"domain.{extension}")


# Singular value decay and cumulative energy ==================================

def svdval_decay(trainsize, marker):
    """Plot the singular values of the first `trainsize` lifted snapshots.
    This requires the file svdvals.h5.
    """
    svdvals = utils.load_basis(trainsize, -1)
    j = np.arange(1, svdvals.size + 1)
    plt.semilogy(j, svdvals, marker+'-',
                 lw=2, ms=8, mew=0, markevery=trainsize//5,
                 label=f"$k = {trainsize:d}$")

def svdval_decay_all(ks=(10000, 20000, 30000), extension="pdf"):
    """Singular Value Decay plot"""
    fig, ax = plt.subplots(1, 1, figsize=(8,4)) # (12,4)

    for k, marker in zip(ks, "odspv"):
        svdval_decay(k, marker)

    ax.set_xlabel("Singular Value Index", fontsize=MEDFONT)
    ax.set_xlim(-250, ks[-1]+5000)
    ax.set_xticks([0] + list(ks))
    ax.set_ylabel("Singular Values", fontsize=MEDFONT)
    ax.set_ylim(1e-6, 1e5)
    ax.set_yticks([1e-4, 1e0, 1e4])
    ax.tick_params(axis="both", which="major", labelsize=SMLFONT)
    # ax.legend(loc="upper right", fontsize=8)
    for k in ks:
        ax.text(k, 1e-5, fr"$k = {k:,d}$".replace(',', "{,}"),
                ha="center", fontsize=SMLFONT)

    fig.tight_layout()
    utils.save_figure(f"svdvals_decay.{extension}")


def svdval_cumulative_energy(trainsize, marker='.'):
    """Plot the cumulative energy of the first `trainsize` lifted snapshots.
    This requires the file svdvals.h5.
    """
    data_path = os.path.join(config.BASE_FOLDER,
                             config.TRNFMT(trainsize), "svdvals.h5")
    with h5py.File(data_path, 'r') as hf:
        svdvals = hf["svdvals"][:]

    svdvals2 = svdvals**2
    cumulative_energy = np.cumsum(svdvals2) / np.sum(svdvals2)
    j = np.arange(1, svdvals.size + 1)
    plt.plot(j, cumulative_energy, marker+'-', lw=.5, ms=3, mew=0,
             label=f"$k = {trainsize:,d}$".replace(',', '{,}'))


def svdval_cumulative_energy_all(extension="pdf"):
    """Singular Value Cumulative Energy plot"""
    fig, ax = plt.subplots(1, 1, figsize=(8,4))

    ax.axhline(.985, color='k', lw=.5, label=r"$98.5\%$ energy")
    for i, r in enumerate([22, 43, 66]):
        ax.axvline(r, color=f"C{i}", lw=.5)
    svdval_cumulative_energy(10000, 'o')
    svdval_cumulative_energy(20000, 'd')
    svdval_cumulative_energy(30000, 's')

    ax.set_xlim(0, 80)
    ax.set_xlabel("Singular Value Index", fontsize=MEDFONT)
    ax.set_ylabel("Cumulative Energy", fontsize=MEDFONT)
    ax.set_yticks([.8, .85, .9, .95, 1])
    ax.tick_params(axis="both", which="major", labelsize=SMLFONT)
    ax.legend(fontsize=MEDFONT//2, loc=(.56, 0))

    fig.tight_layout()
    utils.save_figure(f"svdvals_cumulative_energy.{extension}")

# Results in time =============================================================

def _init_subplots(nrows, ncols):
    _size = max(6*ncols, 18), max(3*nrows, 6)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=_size, sharex=True, sharey="col")
    if nrows == 1:
        axes = np.atleast_2d(axes)
    elif ncols == 1:
        axes = axes.reshape((-1,1))
    return fig, axes


def _format_y_axis(ax):
    """Only use a few y ticks, set in scientific notation."""
    ax.tick_params(axis='y', which="major", labelsize=SMLFONT)
    ax.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(TNYFONT)
    ax.locator_params(axis='y', nbins=2)


def _format_x_axis(ax):
    """Set x ticks and label (time)."""
    ax.set_xlim(.015, .021)
    ax.set_xticks([.015, .017, .019, .021])
    ax.tick_params(axis='x', which="major", labelsize=SMLFONT)
    ax.set_xlabel("Time [s]", fontsize=MEDFONT)


def _large_y_labels(fig, axes, ylabels, ps=False):
    """Write ylabels spanning the subplot columns"""
    for j,label in enumerate(ylabels):
        if ps:                  # No transparency for postscript type
            ax = axes[axes.shape[0]//2,j]
            pad = 2
        else:                   # Transparent axes over top.
            ax = fig.add_subplot(1, axes.shape[1], j+1, frameon=False)
            ax.tick_params(labelcolor="none", bottom=False, left=False)
            pad = 20
        ax.set_ylabel(label, labelpad=pad, fontsize=MEDFONT)


def _small_numbers(axes, num_modes, deim=False):
    """Write r value in bottom-left corner of each subplot."""
    for i, (axrow, r) in enumerate(zip(axes, num_modes)):
        if deim and i == 0:
            c = poddeim.STYLE["color"]
        else:
            c = config.ROM_STYLE["color"]
        for ax in axrow:
            ax.text(.01, .01, fr"$r = {r}$", color=c, fontsize=TNYFONT,
                    ha="left", va="bottom", transform=ax.transAxes)
            a, b = ax.get_ylim()
            lower_bound = .9*(a - b) + b
            for line in ax.get_lines()[:2]:
                if np.any(line.get_data()[:10000] < lower_bound):
                    ax.set_ylim(b - (b - a)/.9, b)


def _format_legend(fig, ax, labels):
    """Make legend centered below the subplots."""

    # Make legend centered below the subplots.
    leg = ax.legend(labels, ncol=len(labels),
                    fontsize=BIGFONT, loc="lower center",
                    bbox_to_anchor=(.5, 0), bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(5)
    return leg


def _format_subplots(fig, axes, num_modes, ylabels,
                     deim=False, numbers=True, ps=False):
    """Do all of the formatting except for y-labels and titles."""
    for ax in axes.flat:
        _format_y_axis(ax)

    for ax in axes[-1,:]:
        _format_x_axis(ax)

    _large_y_labels(fig, axes, ylabels, ps)

    if numbers:
        _small_numbers(axes, num_modes, deim)

    # Fix subplots layout.
    fig.tight_layout(rect=[0, .15 if axes.shape[0] == 1 else .1, 1, 1])
    if ylabels[-1] is None:
        fig.subplots_adjust(hspace=.15, wspace=.15)
    else:
        fig.subplots_adjust(hspace=.15, wspace=.25)

    # Choose legend labels.
    labels = ["GEMS", "ROM"]
    if deim and not numbers:
        r1, r2 = num_modes
        labels = ["GEMS", fr"POD-DEIM, $r={r1}$", fr"OpInf, $r={r2}$"]
    elif deim and numbers:
        labels = ["GEMS", "POD-DEIM", "OpInf"]

    leg = _format_legend(fig, axes[0,0], labels)
    if deim:
        line = leg.get_lines()[-1]
        line.set_color(config.ROM_STYLE["color"])
        line.set_linestyle(config.ROM_STYLE["linestyle"])        


def custom_traces(trainsizes, num_modes, regs,
                  variables=(), locs=(), loclabels=None, keys=(),
                  cutoff=None, filename="trace.pdf"):
    """Draw a grid of subplots where each row corresponds to a different
    ROM (trainsizes[i], num_modes[i], regs[i]) and each column is for a
    specific time trace (variables, locs) or statistical features (keys).
    """
    # Load GEMS time trace data for the given indices and variables.
    if variables:
        assert len(variables) == len(locs)
        locs = np.array(locs)
        nlocs = locs.size
        locs = np.concatenate([locs + i*config.DOF
                               for i in range(config.NUM_ROMVARS)])
        if loclabels is None:
            loclabels = [i+1 for i in range(nlocs)]

        # Load and lift the GEMS time trace results.
        gems_locs = locs[:nlocs*config.NUM_GEMSVARS]
        data, t = utils.load_gems_data(rows=gems_locs, cols=cutoff)
        with utils.timed_block("Lifting GEMS time trace data"):
            traces_gems = dproc.lift(data)
    else:
        nlocs = 0

    # Load GEMS statistical features.
    if keys:
        if isinstance(keys, str):
            keys = [keys]
        features_gems, t = utils.load_spatial_statistics(keys, cutoff)
        if len(keys) == 1:
            features_gems = {keys[0]: features_gems}

    # Initialize the figure.
    nrows, ncols = len(trainsizes), nlocs + len(keys)
    fig, axes = _init_subplots(nrows, ncols)
    axes = np.atleast_2d(axes)

    for i, (trainsize,r,reg) in enumerate(zip(trainsizes, num_modes, regs)):

        # Load basis, training data, ROM, then simulate and reconstruct.
        t, V, scales, q_rom = step4.simulate_rom(trainsize, r, reg, cutoff)
        with utils.timed_block(f"Processing k={trainsize:d}, r={r:d}"):

            if variables:   # First len(locs) columns: variable traces.
                traces_pred = step4.get_traces(locs, q_rom, V, scales)
                for j,var in enumerate(variables):
                    romvar = dproc.getvar(var, traces_pred)
                    gemvar = dproc.getvar(var, traces_gems)
                    axes[i,j].plot(t, gemvar[j,:], **config.GEMS_STYLE)
                    axes[i,j].plot(t[:romvar.shape[1]], romvar[j,:],
                                   **config.ROM_STYLE)
                    axes[i,j].axvline(t[trainsize], lw=2, color='k')

            if keys:        # Last len(keys) columns: statistical features.
                for ax,key in zip(axes[i,nlocs:], keys):
                    features_pred = step4.get_feature(key, q_rom, V, scales)
                    ax.plot(t, features_gems[key], **config.GEMS_STYLE)
                    ax.plot(t, features_pred, **config.ROM_STYLE)
                    ax.axvline(t[trainsize], lw=2, color='k')

    # Format the figure.
    if variables:
        for ax, num in zip(axes[0,:nlocs], loclabels):
            ax.set_title(fr"Location ${num}$", fontsize=MEDFONT)
        ylabels = [config.VARLABELS[var] for var in variables]
    else:
        ylabels = []
    if keys:
        sep = ' ' if nrows > 2 else '\n'
        for key in keys:
            v,action = key.split('_')
            if action == "sum":
                ylabels.append(f"{config.VARTITLES[v]} Concentration{sep}"
                               f"Integral [{config.VARUNITS[v]}]")
            elif action == "mean":
                ylabels.append(f"Spatially Averaged{sep}{config.VARLABELS[v]}")

    _format_subplots(fig, axes, num_modes, ylabels,
                     ps=filename.endswith(".ps"))

    # Set custom y limits for plots in the publication.
    if len(variables) > 0 and variables[0] == "p":
        for ax in axes[:,0]:
            ax.set_ylim(9.5e5, 1.35e6)
    if len(variables) > 0 and variables[0] == "CH4":
        for ax in axes[:,0]:
            ax.set_ylim(4e-3, 1.9e-2)
            ax.set_yticks([1e02, 1.5e-2])
    if len(variables) > 0 and variables[0] == "O2":
        for ax in axes[:,0]:
            ax.set_ylim(-2e-3, 1.2e-2)
    if len(variables) > 1 and variables[1] == "T":
        for ax in axes[:,1]:
            ax.set_ylim(5e2, 3.5e3)
            ax.set_yticks([1e3, 3e3])
    if len(variables) > 1 and variables[1] == "vx" and loclabels[1] == 2:
        for ax in axes[:,1]:
            ax.set_ylim(-240, 190)
    if len(variables) > 1 and variables[1] == "vx" and loclabels[1] == 3:
        for ax in axes[:,1]:
            ax.set_ylim(-5, 10)
    if len(variables) > 1 and variables[1] == "vx" and loclabels[1] == 4:
        for ax in axes[:,1]:
            ax.set_ylim(-50, 180)
    if len(variables) > 2 and variables[2] == "vx":
        for ax in axes[:,2]:
            ax.set_ylim(-5, 10)
    if len(variables) > 3 and variables[3] == "CH4":
        for ax in axes[:,3]:
            ax.set_ylim(-1e-24, 5e-24)
    if len(variables) > 3 and variables[3] == "O2":
        for ax in axes[:,3]:
            ax.set_ylim(.05, .065)

    if len(keys) > 0 and keys[0] == "T_mean":
        for ax in axes[:,nlocs+0]:
            ax.set_ylim(8.25e2, 1.2e3)
    for i in range(len(keys)):
        if keys[i] == "CH4_sum":
            for ax in axes[:,nlocs+i]:
                ax.set_ylim(1.1e3, 1.5e3)
        if keys[i] == "CO2_sum":
            for ax in axes[:,nlocs+i]:
                ax.set_ylim(45, 95)
        if keys[i] == "O2_sum":
            for ax in axes[:,nlocs+i]:
                ax.set_ylim(1.15e3, 1.6e3)
                ax.set_yticks([1.2e3, 1.5e3])

    utils.save_figure(filename)


def all_traces(trainsizes, num_modes, regs, cutoff=None,
               prefix='', extension="pdf"):
    """Plot ALL time traces, as well as spatial averages for pressure,
    velocities, and temperature, and species integrals.
    Save 11 separate figures.
    """
    locs = np.array(config.MONITOR_LOCATIONS)
    nlocs = locs.size
    locs = np.concatenate([locs + i*config.DOF
                           for i in range(config.NUM_ROMVARS)])
    loclabels = [i+1 for i in range(nlocs)]

    # Load and lift the GEMS time trace results.
    gems_locs = locs[:nlocs*config.NUM_GEMSVARS]
    data, t = utils.load_gems_data(rows=gems_locs, cols=cutoff)
    with utils.timed_block("Lifting GEMS time trace data"):
        traces_gems = dproc.lift(data)

    # Load GEMS statistical features.
    keys = [f"{var}_mean" for var in config.ROM_VARIABLES[:4]]
    keys += [f"{var}_sum" for var in config.SPECIES]
    features_gems, t = utils.load_spatial_statistics(keys, cutoff)
    keys = np.reshape(keys, (2,nlocs))

    # Initialize the figures.
    nrows, ncols = len(trainsizes), nlocs
    figs_axes = [_init_subplots(nrows, ncols)
                 for _ in range(config.NUM_ROMVARS + 2)]

    for i, (trainsize,r,reg) in enumerate(zip(trainsizes, num_modes, regs)):
        # Load basis, training data, ROM, then simulate the ROM.
        t, V, scales, q_rom = step4.simulate_rom(trainsize, r, reg, cutoff)

        with utils.timed_block(f"Processing k={trainsize:d}, r={r:d}"):
            # Reconstruct traces from the ROM (all variables, all locations).
            traces_pred = step4.get_traces(locs, q_rom, V, scales)

            for k,var in enumerate(config.ROM_VARIABLES):
                fig, axes = figs_axes[k]
                # Extract a single variable (all locations).
                romvar = dproc.getvar(var, traces_pred)
                gemvar = dproc.getvar(var, traces_gems)

                for j in range(nlocs):
                    # Plot a single variable at a single location.
                    axes[i,j].plot(t, gemvar[j,:], **config.GEMS_STYLE)
                    axes[i,j].plot(t, romvar[j,:], **config.ROM_STYLE)
                    axes[i,j].axvline(t[trainsize], lw=2, color='k')

            for (fig,axes), subkeys in zip(figs_axes[-2:], keys):
                # Reconstruct each feature from the rom.
                for ax,key in zip(axes[i,:], subkeys):
                    features_pred = step4.get_feature(key, q_rom, V, scales)
                    ax.plot(t, features_gems[key], **config.GEMS_STYLE)
                    ax.plot(t, features_pred, **config.ROM_STYLE)
                    ax.axvline(t[trainsize], lw=2, color='k')

    # Format and save each figure.
    for k,var in enumerate(config.ROM_VARIABLES + ["means", "sums"]):
        fig, axes = figs_axes[k]
        if var in config.ROM_VARIABLES:
            for num,ax in enumerate(axes[0,:]):
                ax.set_title(fr"Location ${num+1}$", fontsize=MEDFONT)
            ylabels = [config.VARLABELS[var]] + [None]*3
        else:
            sep = ' ' if nrows > 2 else '\n'
            if var == "means":
                ylabels = [f"Spatially Averaged{sep}{config.VARLABELS[v]}"
                           for v in config.ROM_VARIABLES[:4]]
            elif var == "sums":
                ylabels = [f"{config.VARTITLES[v]} Concentration{sep}"
                           f"Integral [{config.VARUNITS[v]}]"
                           for v in config.SPECIES]

        _format_subplots(fig, axes, num_modes, ylabels,
                         ps=(extension == "ps"))
        plt.figure(fig.number)
        utils.save_figure(f"{prefix}pointtrace_{var}.{extension}")


def projection_errors(trainsize, r, regs, variables, cutoff=70000,
                      filename="projerrors_grid.pdf"):
    """Plot spatially averaged projection errors in time.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    rs : list(int)
        Basis sizes to test.
    """
    # Load the POD basis and simulate the ROM.
    t, V, scales, q_rom = step4.simulate_rom(trainsize, r, regs, cutoff)

    # Load, lift, and project the GEMS data.
    gems_data, _ = utils.load_gems_data(cols=cutoff)
    with utils.timed_block("Lifting and projecting GEMS data"):
        gems_data = dproc.lift(gems_data)
        q_gems = V.T @ dproc.scale(gems_data.copy(), scales)[0]

    # Initialize the figure.
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9,7))

    # Get projection errors for each variable.
    for var, ax in zip(variables, axes):
        gems_var = dproc.getvar(var, gems_data)
        denom = np.abs(gems_var)
        if var in ["vx", "vy"]:
            denom = np.max(denom, axis=0)
        
        # Compute the projection error of the ROM results.
        with utils.timed_block(f"Computing ROM projection error ({var})"):
            Vvar = dproc.getvar(var, V)
            rom_var = dproc.unscale(Vvar @ q_rom, scales, var)
            rom_error = np.mean(np.abs(rom_var - gems_var) / denom, axis=0)

        # Scale, project, and unproject the GEMS data for this variable.
        with utils.timed_block(f"Computing GEMS projection error ({var})"):
            gems_var2 = dproc.unscale(Vvar @ q_gems, scales, var)
            gems_error = np.mean(np.abs(gems_var2 - gems_var) / denom, axis=0)
            
        # Plot the errors.
        ax.plot(t, gems_error, c='C3', label="Projection error")
        ax.plot(t, rom_error, **config.ROM_STYLE)
        ax.axvline(t[trainsize], color='k', lw=2)

        # Format the y axis.
        ax.set_ylim(0, .45)
        ax.set_yticks([0, .2, .4])
        ax.tick_params(axis='y', which="major", labelsize=SMLFONT)
        ax.set_ylabel(config.VARTITLES[var], fontsize=MEDFONT, labelpad=2)

    # Format the figure.
    _format_x_axis(axes[1])
    fig.tight_layout(rect=[0, .15, 1, 1])
    fig.subplots_adjust(hspace=.15)
    _format_legend(fig, axes[0], ["Projection error", "ROM error"])
#     for ax in axes:
#         ax.text(.01, .97, fr"$r = {r}$",
#                 color=config.ROM_STYLE["color"], fontsize=TNYFONT,
#                 ha="left", va="top", transform=ax.transAxes)

    # Save the figure.
    utils.save_figure(filename)


def compare_poddeim(trainsize, r, regs,
                    var, location, keys, cutoff=70000,
                    filename="poddeim_comparison_grid.pdf"):
    """Plot a comparison between OpInf and POD-DEIM.
    
    |       Point Trace       |
    |-------------------------|
    | Feature 1 | | Feature 2 |

    """
    # Load the relevant GEMS data.
    assert var in ["p", "vx", "vy", "T"]    # no lifting.
    assert len(keys) == 2                   # two spatial features.
    loc = config.MONITOR_LOCATIONS[location - 1]
    trace_gems, t1 = utils.load_gems_data(rows=np.array([loc]), cols=cutoff)
    trace_gems = np.ravel(trace_gems)
    specs_gems, t2 = utils.load_spatial_statistics(keys, k=cutoff)

    # Get OpInf simulation results and extract relevant data.
    t_rom, V, scales, q_rom = step4.simulate_rom(trainsize, r, regs, cutoff)
    with utils.timed_block("Extracting OpInf features"):
        trace_rom = dproc.unscale(V[loc] @ q_rom, scales, var)
        specs_rom = {k: step4.get_feature(k, q_rom, V, scales) for k in keys}

    # Load POD-DEIM data and extract relevant data.
    data_deim, t_deim = poddeim.load_data(cols=cutoff)
    with utils.timed_block("Extracting POD-DEIM features"):
        lifted_deim = dproc.lift(data_deim)
        del data_deim
        trace_deim = lifted_deim[loc]
        specs_deim = {k: step4.get_feature(k, lifted_deim) for k in keys}

    # Initialize the figure.
    fig = plt.figure(figsize=(18,9), constrained_layout=False)
    gsp = plt.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gsp[0,:])
    ax2 = fig.add_subplot(gsp[1,0])
    ax3 = fig.add_subplot(gsp[1,1])

    # Plot results.
    with utils.timed_block("Plotting results"):
        # Time trace: top plot.
        ax1.plot(t1, trace_gems, **config.GEMS_STYLE)
        ax1.plot(t_deim, trace_deim, **poddeim.STYLE)
        ax1.plot(t_rom, trace_rom, **config.ROM_STYLE)
        ax1.axvline(t1[trainsize], lw=2, color='k')

        # Spatial features: bottom plots.
        for key, ax in zip(keys, [ax2, ax3]):
            ax.plot(t2, specs_gems[key], **config.GEMS_STYLE)
            ax.plot(t_deim, specs_deim[key], **poddeim.STYLE)
            ax.plot(t_rom, specs_rom[key], **config.ROM_STYLE)
            ax.axvline(t2[trainsize], lw=2, color='k')
    
    # Format axes.
    for ax in [ax1, ax2, ax3]:
        _format_y_axis(ax)
        _format_x_axis(ax)

    fig.tight_layout(rect=[0, .15, 1, 1])
    fig.subplots_adjust(hspace=.6, wspace=.25)

    # Create the legend, centered at the bottom of the plot.
    labels = ["GEMS", fr"POD-DEIM, $r=70$", fr"OpInf, $r={r:d}$"]    
    _format_legend(fig, ax1, labels)

    # Set titles and labels.
    ax1.set_title(fr"Location ${location:d}$", fontsize=MEDFONT)
    ax1.set_ylabel(config.VARLABELS[var],
                   fontsize=MEDFONT, labelpad=2)
    for key, ax in zip(keys, [ax2, ax3]):
        v,action = key.split('_')
        if action == "sum":
            ax.set_ylabel(f"{config.VARTITLES[v]} Concentration\n"
                           f"Integral [{config.VARUNITS[v]}]",
                           fontsize=MEDFONT, labelpad=2)
        elif action == "mean":
            ax.set_ylabel(f"Spatially Averaged\n{config.VARLABELS[v]}",
                           fontsize=MEDFONT, labelpad=2)

    utils.save_figure(filename)



def all_poddeim_comparisons(trainsize, r, reg, cutoff=None,
                            prefix='', extension="pdf"):
    """Plot all comparisons between POD-DEIM and OpInf (one on each row),
    both overlayed with GEMS. Generate one figure for each time trace,
    one for spatial averages, and one for species integrals.
    """
    if prefix:
        prefix += '_'

    locs = np.array(config.MONITOR_LOCATIONS)
    nlocs = locs.size
    locs = np.concatenate([locs + i*config.DOF
                           for i in range(config.NUM_ROMVARS)])
    loclabels = [i+1 for i in range(nlocs)]

    # Load and lift the GEMS time traces (all variables, all locations).
    gems_locs = locs[:nlocs*config.NUM_GEMSVARS]
    data, t = utils.load_gems_data(rows=gems_locs, cols=cutoff)
    with utils.timed_block("Lifting GEMS time trace data"):
        traces_gems = dproc.lift(data)

    # Load GEMS statistical features.
    keys = [f"{var}_mean" for var in config.ROM_VARIABLES[:4]]
    keys += [f"{var}_sum" for var in config.SPECIES]
    features_gems, t = utils.load_spatial_statistics(keys, cutoff)

    # Load and lift the POD-DEIM data.
    data, t_deim = poddeim.load_data(cols=cutoff)
    with utils.timed_block("Lifting POD-DEIM data"):
        lifted_deim = dproc.lift(data)

    # Extract POD-DEIM time traces (all variables, all locations).
    traces_deim = step4.get_traces(locs, lifted_deim)

    # Compute POD-DEIM statistical features.
    with utils.timed_block("Computing POD-DEIM statistical features"):
        features_deim = {key : step4.get_feature(key, lifted_deim)
                              for key in keys}

    # Load basis, training data, ROM, then simulate the ROM (OpInf).
    t, V, scales, q_rom = step4.simulate_rom(trainsize, r, reg, cutoff)

    # Reconstruct Opinf time traces (all variables, all locations).
    traces_opinf = step4.get_traces(locs, q_rom, V, scales)

    # Compute OpInf statistical features.
    with utils.timed_block("Computing OpInf statistical features"):
        features_opinf = {key : step4.get_feature(key, q_rom, V, scales)
                               for key in keys}

    # Initialize the figures.
    nrows, ncols = 2, 4
    figs_axes = [_init_subplots(nrows, ncols)
                 for _ in range(config.NUM_ROMVARS + 2)]
    keys = np.reshape(keys, (2,ncols))

    # Plot results.
    with utils.timed_block("Plotting results"):
        for i in range(2):

            # Select POD-DEIM data or OpInf data.
            tt = t_deim if i == 0 else t
            traces_pred = traces_deim if i == 0 else traces_opinf
            features_pred = features_deim if i == 0 else features_opinf
            ROMSTYLE = poddeim.STYLE if i == 0 else config.ROM_STYLE

            for k,var in enumerate(config.ROM_VARIABLES):
                fig, axes = figs_axes[k]
                # Extract a single variable (all locations).
                romvar = dproc.getvar(var, traces_pred)
                gemvar = dproc.getvar(var, traces_gems)

                for j in range(nlocs):
                    # Plot a single variable at a single location.
                    axes[i,j].plot(t,  gemvar[j,:], **config.GEMS_STYLE)
                    axes[i,j].plot(tt, romvar[j,:], **ROMSTYLE)
                    axes[i,j].axvline(t[trainsize], lw=2, color='k')

            for (fig,axes), subkeys in zip(figs_axes[-2:], keys):
                # Reconstruct each feature from the rom.
                for ax,key in zip(axes[i,:], subkeys):
                    ax.plot(t,  features_gems[key], **config.GEMS_STYLE)
                    ax.plot(tt, features_pred[key], **ROMSTYLE)
                    ax.axvline(t[trainsize], lw=2, color='k')

    # Format and save each figure.
    for k,var in enumerate(config.ROM_VARIABLES + ["means", "sums"]):
        fig, axes = figs_axes[k]
        if var in config.ROM_VARIABLES:
            ylabels = [config.VARLABELS[var]] + [None]*3
            for num,ax in enumerate(axes[0,:]):
                ax.set_title(fr"Location ${num+1}$", fontsize=MEDFONT)
        elif var == "means":
            ylabels = [config.VARLABELS[v] for v in config.ROM_VARIABLES[:4]]
            for ax in axes[0,:]:
                ax.set_title("Spatial Averages", fontsize=MEDFONT)
        elif var == "sums":
            ylabels = [config.VARLABELS[v] for v in config.SPECIES]
            for ax in axes[0,:]:
                ax.set_title("Spatial Integrals", fontsize=MEDFONT)

        _format_subplots(fig, axes, [70, r], ylabels, deim=True,
                         ps=(extension == "ps"))
        plt.figure(fig.number)
        utils.save_figure(f"{prefix}comparepoddeim_{var}.{extension}")


# =============================================================================

def main():
    """Create figures for the publication."""
    # Singular value behavior -------------------------------------------------
    svdval_decay_all()

    # Point traces ------------------------------------------------------------
    custom_traces(trainsizes=[     10000,       20000,       30000],
                   num_modes=[        22,          43,          66],
                        regs=[(91,32251), (316,18199), (105,27906)],
                  variables=["p", "vx"],
                  locs=[config.MONITOR_LOCATIONS[i] for i in [0,2]],
                  loclabels=[1, 3],
                  cutoff=60000,
                  filename="trace_grid.ps")

    # Statistical features ----------------------------------------------------
    custom_traces(trainsizes=[     10000,       20000,       30000],
                   num_modes=[        22,          43,          66],
                        regs=[(91,32251), (316,18199), (105,27906)],
                  keys=["CH4_sum", "CO2_sum"],
                  cutoff=60000,
                  filename="statistics_grid.ps")

    # Projection errors -------------------------------------------------------
    projection_errors(trainsize=20000, r=43, regs=(316,18199),
                      variables=["p", "T"], cutoff=60000,
                      filename="projerrors_grid.ps")

    # POD-DEIM comparison -----------------------------------------------------
    compare_poddeim(trainsize=20000, r=43, regs=(316,18199),
                    var="p", location=3, keys=["O2_sum", "CO2_sum"],
                    cutoff=60000, filename="poddeim_grid.ps")

    
def extra():
    """Create figures that go in the website but not in the publication."""
    # Computational domain ----------------------------------------------------
    domain()

    # Singular value behavior -------------------------------------------------
    svdval_decay_all(extension="svg")
    svdval_cumulative_energy_all(extension="svg")

    # Point traces ------------------------------------------------------------
    all_traces(trainsizes=[     10000,       20000,       30000],
                num_modes=[        22,          43,          66],
                     regs=[(91,32251), (316,18199), (105,27906)],
               cutoff=60000, extension="svg")

    # POD-DEIM comparisons
    all_poddeim_comparisons(trainsize=20000, r=43, reg=(316,18199),
                            cutoff=60000, extension="svg")


# =============================================================================
if __name__ == "__main__":
    utils.reset_logger()
    main()
    extra()
