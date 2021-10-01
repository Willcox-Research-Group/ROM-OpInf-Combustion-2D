# step5_export.py
"""Write Tecplot-readable ASCII (text) files from simulation data.
The resulting files can be used with Tecplot to visualize snapshots
over the entire computational domain.

Output types
------------
* gems: write full-order GEMS data in the ROM learning variables.
* rom: write reconstructed ROM outputs. The specific ROM is selected via
       command line arguments --trainsize, --modes, and --regularization.
* error: write the absolute error between the GEMS data and the ROM outputs.

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
import step4_plot as step4


# header = HEADER.format(varnames, timeindex, solutiontime,
#                        num_nodes, DOF, num_vars, datatypes)
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
"""

NCOLS = 4


def main(timeindices, variables=None, snaptype=("gems", "rom", "error"),
         trainsize=None, r=None, reg=None):
    """Convert a snapshot in .h5 format to a .dat file that matches the format
    of grid.dat. The new file is saved in `config.tecplot_path()` with the same
    filename and the new file extension .dat.

    Parameters
    ----------
    timeindices : ndarray(int) or int
        Indices (one-based) in the full time domain of the snapshots to save.
    variables : str or list(str)
        Variables to save, a subset of config.ROM_VARIABLES.
        Defaults to all variables.
    snaptype : {"rom", "gems", "error"} or list(str)
        Which kinds of snapshots to save. Options:
        * "gems": snapshots from the full-order GEMS data;
        * "rom": reconstructed snapshots produced by a ROM;
        * "error": absolute error between the full-order data
                   and the reduced-order reconstruction.
        If "rom" or "error" are selected, the remaining arguments are required.
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Number of retained modes in the ROM.
    reg : two non-negative floats
        Regularization hyperparameters used to train the ROM.
    """
    utils.reset_logger(trainsize)

    # Parse parameters.
    timeindices = np.sort(np.atleast_1d(timeindices))
    simtime = timeindices.max()
    t = utils.load_time_domain(simtime+1)

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
        # cell_volume = grid_data[2*num_nodes:3*num_nodes]
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
        t, V, scales, q_rom = step4.simulate_rom(trainsize, r, reg,
                                                 steps=simtime+1)

        # Reconstruct the results (only selected variables / snapshots).
        with utils.timed_block("Reconstructing simulation results"):
            q_rec = dproc.unscale(V @ q_rom[:,timeindices], scales)
            q_rec = np.concatenate([dproc.getvar(v, q_rec) for v in variables])

    dsets = {}
    if "rom" in snaptype:
        dsets["rom"] = q_rec
    if "gems" in snaptype:
        dsets["gems"] = true_snaps
    if "error" in snaptype:
        with utils.timed_block("Computing absolute error of reconstruction"):
            abs_err = np.abs(true_snaps - q_rec)
        dsets["error"] = abs_err

    # Save each of the selected snapshots in Tecplot format matching grid.dat.
    for j,tindex in enumerate(timeindices):

        header = HEADER.format(varnames, tindex, t[tindex],
                               num_nodes, config.DOF,
                               len(variables)+2, "DOUBLE "*len(variables))
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


def temperature_average(trainsize, r, reg, cutoff=60000):
    """Get the average-in-time temperature profile for the GEMS data and a
    specific ROM.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.
    r : int
        Dimension of the ROM.
    reg : float
        Regularization hyperparameters used to train the ROM.
    cutoff : int
        Number of time steps to average over.
    """
    utils.reset_logger(trainsize)

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
        # cell_volume = grid_data[2*num_nodes:3*num_nodes]
        connectivity = grid_data[3*num_nodes:]

    # Compute full-order time-averaged temperature from GEMS data.
    _s = config.DOF*config.GEMS_VARIABLES.index("T")
    gems_data, _ = utils.load_gems_data(rows=slice(_s, _s+config.DOF),
                                        cols=cutoff)
    with utils.timed_block("Computing time-averaged GEMS temperature"):
        T_gems = gems_data.mean(axis=1)
        assert T_gems.shape == (config.DOF,)

    # Simulate ROM and compute the time-averaged temperature.
    t, V, scales, q_rom = step4.simulate_rom(trainsize, r, reg, steps=cutoff)
    with utils.timed_block("Reconstructing ROM simulation results"):
        T_rom = dproc.unscale(dproc.getvar("T",V) @ q_rom, scales, "T")
        T_rom = T_rom.mean(axis=1)
        assert T_rom.shape == (config.DOF,)

    header = HEADER.format('"T"', 0, 0, num_nodes, config.DOF,
                           3, "DOUBLE "*3)
    header = header.replace("VARLOCATION=([3-3]", "VARLOCATION=([3]")
    for label, dset in zip(["gems", "rom"], [T_gems, T_rom]):
        if label == "gems":
            save_path = os.path.join(config.tecplot_path(), "gems",
                                     "temperature_average.dat")
        elif label == "rom":
            folder = config.rom_snapshot_path(trainsize, r, reg)
            save_path = os.path.join(folder, "temperature_average.dat")
        with utils.timed_block(f"Writing {label} temperature average"):
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
                    row = ' '.join(f"{v:.9E}" for v in dset[i:i+NCOLS])
                    outfile.write(row + '\n')

                # Write connectivity information.
                for i in range(0, len(connectivity), NCOLS):
                    outfile.write(' '.join(connectivity[i:i+NCOLS]) + '\n')


def basis(trainsize, r, variables=None):
    """Export the POD basis vectors to Tecplot format.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to compute the basis.
    r : int
        Number of basis vectors to save.
    variables : str or list(str)
        Variables to save, a subset of config.ROM_VARIABLES.
        Defaults to all variables.
    """
    utils.reset_logger(trainsize)

    if variables is None:
        variables = config.ROM_VARIABLES
    elif isinstance(variables, str):
        variables = [variables]
    varnames = '\n'.join(f'"{v}"' for v in variables)

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
        # cell_volume = grid_data[2*num_nodes:3*num_nodes]
        connectivity = grid_data[3*num_nodes:]

    # Load the basis and extract desired variables.
    V, _, _ = utils.load_basis(trainsize, r)
    V = np.concatenate([dproc.getvar(var, V) for var in variables])

    # Save each of the basis vectors in Tecplot format matching grid.dat.
    for j in range(r):
        header = HEADER.format(varnames, j, j, num_nodes, config.DOF,
                               len(variables)+2, "DOUBLE "*len(variables))
        save_folder = config._makefolder(config.tecplot_path(),
                                         "basis", config.TRNFMT(trainsize))
        save_path = os.path.join(save_folder, f"vec_{j+1:03d}.dat")
        with utils.timed_block(f"Writing basis vector {j+1:d}"):
            with open(save_path, 'w') as outfile:
                # Write the header.
                outfile.write(header)

                # Write the geometry data (x,y coordinates).
                for i in range(0, len(x), NCOLS):
                    outfile.write(' '.join(x[i:i+NCOLS]) + '\n')
                for i in range(0, len(y), NCOLS):
                    outfile.write(' '.join(y[i:i+NCOLS]) + '\n')

                # Write the data for each variable.
                for i in range(0, V.shape[0], NCOLS):
                    row = ' '.join(f"{v:.9E}" for v in V[i:i+NCOLS,j])
                    outfile.write(row + '\n')

                # Write connectivity information.
                for i in range(0, len(connectivity), NCOLS):
                    outfile.write(' '.join(connectivity[i:i+NCOLS]) + '\n')
    print(f"Basis info exported to {save_folder}/*.dat.")


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
                                [--trainsize TRAINSIZE]
                                [--modes MODES]
                                [--regularization REG1 REG2]"""
    parser.add_argument("snaptype", type=str, nargs='*',
                        help="which snapshot types to save (gems, rom, error)")
    parser.add_argument("--timeindex", type=int, nargs='*',
                        default=list(range(0,60100,100)),
                        help="indices of snapshots to save "
                             "(default every 100th snapshot)")
    parser.add_argument("--variables", type=str, nargs='*',
                        default=config.ROM_VARIABLES,
                        help="variables to save, a subset of "
                             "config.ROM_VARIABLES (default all)")

    parser.add_argument("--trainsize", type=int, nargs='?',
                        help="number of snapshots in the ROM training data")
    parser.add_argument("--modes", type=int, nargs='?',
                        help="ROM dimension (number of retained POD modes)")
    parser.add_argument("--regularization", type=float, nargs='*',
                        help="regularization hyperparameters in the "
                             "ROM training")

    parser.add_argument("--temperature-average", action="store_true",
                        help="compute temperature averages of GEMS / ROM")
    parser.add_argument("--basis", action="store_true",
                        help="save basis vectors for visualization")

    # Do the main routine.
    args = parser.parse_args()
    if ("rom" in args.snaptype) or ("error" in args.snaptype):
        if args.trainsize is None:
            raise TypeError("--trainsize required")
        if args.modes is None:
            raise TypeError("--modes required")
        if args.regularization is None:
            raise TypeError("--regularization required")

    if args.temperature_average:
        temperature_average(args.trainsize, args.modes, args.regularization)
    elif args.basis:
        basis(args.trainsize, args.modes, args.variables)
    else:
        main(args.timeindex, args.variables, args.snaptype,
             args.trainsize, args.modes, args.regularization)
