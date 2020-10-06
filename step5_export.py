# step5_export.py
"""Write Tecplot-friendly ASCII (text) files from simulation data.
The resulting files can be used with Tecplot to visualize snapshots
over the entire computational domain.

Output types
------------
* gems: write full-order GEMS data, converting mass fractions to molar
    concentrations.
* rom: write reconstructed ROM outputs, calculating temperature from the
    results. The specific ROM is selected via command line arguments
    --trainsize, --modes, and --regularization.
* error: write the absolute error between the full-order GEMS data and
    the ROM reconstruction.

Examples
--------
# Export every 100th snapshot (default) of GEMS data (all variables).
$ python3 step5_export.py gems

# Export only snapshot 5000 of GEMS data (all variables).
$ python3 step5_export.py gems --timeindex 5000

# Export only snapshot 4000 of GEMS pressure and temperature data.
$ python3 step5_export.py gems --timeindex 4000 --variables p T

# Export snapshot 4000 of reconstructed pressure, temperature, and methane
# data from the ROM trained from 10,000 snapshots, 22 POD modes, and
# regularization parameter 3e4.
$ python3 step5_export.py rom --timeindex 4000 --variables p T CH4
                              --trainsize 10000 --modes 22 --regularization 3e4

# Export every 100th snapshot of reconstructed ROM data (all variables) and the
# absolute errors, derived from the ROM trained from 20,000 snapshots, 44 POD
# modes, and regularization parameter 3e4.
$ python3 step5_export.py rom error --trainsize 20000 --modes 44
                                    --regularization 3e4

Loading Results
---------------
>>> import config
>>> print("Tecplot-friendly files are exported to", config.tecplot_path())

Command Line Arguments
----------------------
"""
import os
import re
import numpy as np

import config
import utils
import data_processing as dproc


HEADER = """TITLE = "Combustion GEMS 2D Nonintrusive ROM"
VARIABLES="x"
"y"
{:s}
ZONE T="zone 1"
STRANDID={:d}, SOLUTIONTIME={:.7f}
Nodes={:d}, Elements={:d}, ZONETYPE=FEQuadrilateral
DATAPACKING=BLOCK
VARLOCATION=([3-{:d}]=CELLCENTERED)
DT=({:s})
""" # header = HEADER.format(varnames, timeindex, solutiontime,
    #                        num_nodes, DOF, num_vars, datatypes)

NCOLS = 4


def main(timeindices, variables=None, snaptype=["gems", "rom", "error"],
         trainsize=None, r=None, reg=None):
    """Convert a snapshot in .h5 format to a .dat file that matches the format
    of grid.dat. The new file is saved in `config.tecplot_path()` with the same
    filename and the new file extension .dat.

    Parameters
    ----------
    timeindices : ndarray(int) or int
        Indices (one-based) in the full time domain of the snapshots to save.

    variables : str or list(str)
        The variables to scale, a subset of config.ROM_VARIABLES.
        Defaults to all variables.

    snaptype : {"rom", "gems", "error"} or list(str)
        Which kinds of snapshots to save. Options:
        * "gems": snapshots from the full-order GEMS data;
        * "rom": reconstructed snapshots produced by a ROM;
        * "error": absolute error between the full-order data
                   and the reduced-order reconstruction.
        If "rom" or "error" are selected, the ROM is selected by the
        remaining arguments.

    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Number of retained modes in the ROM.

    reg : float
        Regularization factor used to train the ROM.
    """
    utils.reset_logger(trainsize)

    # Parse parameters.
    timeindices = np.sort(np.atleast_1d(timeindices))
    simtime = timeindices.max()
    t = utils.load_time_domain(simtime+1)

    # Parse the variables.
    if variables is None:
        variables = config.ROM_VARIABLES
    elif isinstance(variables, str):
        variables = [variables]
    varnames = '\n'.join(f'"{v}"' for v in variables)

    if isinstance(snaptype, str):
        snaptype = [snaptype]
    for stype in snaptype:
        if stype not in ("gems", "rom", "error"):
            raise ValueError(f"invalid snaptype '{stype}'")

    # Read the grid file.
    with utils.timed_block("Reading Tecplot grid data"):
        # Parse the header.
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
        x = grid_data[:num_nodes]
        y = grid_data[num_nodes:2*num_nodes]
        cell_volume = grid_data[2*num_nodes:3*num_nodes]
        connectivity = grid_data[3*num_nodes:]

    # Extract full-order data if needed.
    if ("gems" in snaptype) or ("error" in snaptype):
        gems_data, _ = utils.load_gems_data(cols=timeindices)
        with utils.timed_block("Lifting selected snapshots of GEMS data"):
            lifted_data = dproc.lift(gems_data)
            true_snaps = np.concatenate([dproc.getvar(v, lifted_data)
                                                        for v in variables])
    # Simulate ROM if needed.
    if ("rom" in snaptype) or ("error" in snaptype):
        # Load the SVD data.
        V, _, scales = utils.load_basis(trainsize, r)

        # Load the initial conditions.
        X_, _, _ = utils.load_projected_data(trainsize, r)

        # Load the appropriate ROM.
        rom = utils.load_rom(trainsize, r, reg)

        # Simulate the ROM over the time domain.
        with utils.timed_block(f"Simulating ROM with r={r:d}, reg={reg:.0e}"):
            x_rom = rom.predict(X_[:,0], t, config.U, method="RK45")
            if np.any(np.isnan(x_rom)) or x_rom.shape[1] < simtime:
                raise ValueError("ROM unstable!")

        # Reconstruct the results (only selected variables / snapshots).
        with utils.timed_block("Reconstructing simulation results"):
            x_rec = dproc.unscale(V[:,:r] @ x_rom[:,timeindices], scales)
            x_rec = np.concatenate([dproc.getvar(v, x_rec) for v in variables])

    dsets = {}
    if "rom" in snaptype:
        dsets["rom"] = x_rec
    if "gems" in snaptype:
        dsets["gems"] = true_snaps
    if "error" in snaptype:
        with utils.timed_block("Computing absolute error of reconstruction"):
            abs_err = np.abs(true_snaps - x_rec)
        dsets["error"] = abs_err

    # Save each of the selected snapshots in Tecplot format matching grid.dat.
    for j,tindex in enumerate(timeindices):

        header = HEADER.format(varnames, tindex, t[tindex],
                               num_nodes, config.DOF,
                               len(variables)+2, "SINGLE "*len(variables))
        for label, dset in dsets.items():

            if label == "gems":
                save_path = config.gems_snapshot_path(tindex)
            if label in ("rom", "error"):
                folder = config.rom_snapshot_path(trainsize, r, reg)
                save_path = os.path.join(folder, f"{label}_{tindex:05d}.dat")
            with utils.timed_block(f"Writing {label} snapshot {tindex:05d}"):
                with open(save_path, 'w') as outfile:
                    # Write the header.
                    outfile.write(header)

                    # Write the geometry data (x,y coordinates).
                    for i in range(0, len(x), NCOLS):
                        outfile.write(' '.join(x[i:i+NCOLS]) + '\n')
                    for i in range(0, len(y), NCOLS):
                        outfile.write(' '.join(y[i:i+NCOLS]) + '\n')

                    # Write the data for each variable.
                    for i in range(0, dset.shape[0], NCOLS):
                        row = ' '.join(f"{v:.9E}"
                                       for v in dset[i:i+NCOLS,j])
                        outfile.write(row + '\n')

                    # Write connectivity information.
                    for i in range(0, len(connectivity), NCOLS):
                        outfile.write(' '.join(connectivity[i:i+NCOLS]) + '\n')


# =============================================================================
if __name__ == '__main__':
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} -h
        python3 {__file__} (gems | rom | error)
                                --timeindex T [...]
                                --variables V [...]
                                --trainsize S [...]
                                --modes R [...]
                                --regularization REG [...]"""
    parser.add_argument("snaptype", type=str, nargs='*',
                        help="which snapshot types to save (gems, rom, error)")
    parser.add_argument("-idx", "--timeindex", type=int, nargs='*',
                        default=list(range(0,60100,100)),
                        help="indices of snapshots to save "
                             "(default every 100th snapshot)")
    parser.add_argument("-vars", "--variables", type=str, nargs='*',
                        default=config.ROM_VARIABLES,
                        help="variables to save, a subset of "
                             "config.ROM_VARIABLES (default all)")

    parser.add_argument("--trainsize", type=int, nargs='?',
                        help="number of snapshots in the ROM training data")
    parser.add_argument("-r", "--modes", type=int, nargs='?',
                        help="ROM dimension (number of retained POD modes)")
    parser.add_argument("-reg", "--regularization", type=float, nargs='?',
                        help="regularization parameter in the ROM training")

    # Do the main routine.
    args = parser.parse_args()
    if ("rom" in args.snaptype) or ("error" in args.snaptype):
        if args.trainsize is None:
            raise TypeError("--trainsize required")
        if args.modes is None:
            raise TypeError("--modes required")
        if args.regularization is None:
            raise TypeError("--regularization required")

    main(args.timeindex, args.variables, args.snaptype,
         args.trainsize, args.modes, args.regularization)
