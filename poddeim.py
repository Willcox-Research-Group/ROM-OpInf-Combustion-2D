# poddeim.py
"""Tools and routines for processing POD-DEIM outputs.

* read_and_save_data(): Read the POD-DEIM output from .dat files, compile it
  into a single data set of size (num_variables * domain_size)x(num_snapshots),
  and save it in HDF5 format.
* export_to_tecplot(): Export the POD-DEIM output in Tecplot-friendly format
"""
import os
import re
import glob
import h5py
import logging
import numpy as np

import config
import utils
import data_processing as dproc
from step1_unpack import _SIMTIME, _HEADEREND, _ELEMENTS  # Regular expressions
from step5_export import HEADER, NCOLS                    # Export globals


# Globals
PODDEIM_DATA_FILE = "pod_deim.h5"
STYLE = dict(linestyle="-.", color="gray", label="POD-DEIM")


# Utilities ===================================================================

def poddeim_data_path():
    """Return the path to the file containing the POD-DEIM data."""
    return os.path.join(config.BASE_FOLDER, PODDEIM_DATA_FILE)


def load_data(rows=None, cols=None):
    """Load the indicated rows and colums of POD-DEIM data.
    Note that the time step is 10x that of the GEMS data.

    Parameters
    ----------
    rows : int, slice, or one-dimensional ndarray of integer indices
        Which rows (spatial locations) to extract from the data (default all).
        If an integer, extract the first `rows` rows.

    cols : int, slice, or one-dimensional ndarray of integer indices
        Which columns (temporal points) to extract from the data (default all).
        If an integer, extract the first `cols` columns.

    Returns
    -------
    data : (nrows,ncols) ndarray
        The indicated rows / columns of the data.

    time_domain : (ncols,) ndarray
        The time (in seconds) associated with each column of extracted data.
    """
    # Locate the data.
    data_path = utils._checkexists(poddeim_data_path())

    # Ensure data is loaded in ascending index order (HDF5 requirement).
    if isinstance(rows, np.ndarray):
        if isinstance(cols, np.ndarray):
            raise ValueError("only one of `rows` and `cols` can be NumPy array")
        roworder = np.argsort(rows)
        rows = rows.copy()[roworder]
        oldroworder = np.argsort(roworder)
    elif np.isscalar(rows) or rows is None:
        rows = slice(None, rows)

    if isinstance(cols, np.ndarray):
        colorder = np.argsort(cols)
        cols = cols.copy()[colorder]
        oldcolorder = np.argsort(colorder)
    elif np.isscalar(cols) or cols is None:
        cols = slice(None, cols)

    # Extract the data.
    NROWS = config.NUM_GEMSVARS * config.DOF
    with utils.timed_block(f"Loading POD-DEIM data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shape.
            if hf["data"].shape[0] != NROWS:
                raise RuntimeError(f"data should have {NROWS} rows")
            data = hf["data"][rows,cols]
            time_domain = hf["time"][cols]

    # Restore the initial ordering if needed.
    if isinstance(rows, np.ndarray):
        data = data[oldroworder,:]
    elif isinstance(cols, np.ndarray):
        data = data[:,oldcolorder]
        time_domain = time_domain[oldcolorder]

    return data, time_domain


def snapshot_path(timeindex):
    folder = config._makefolder(config.tecplot_path(), "poddeim")
    return os.path.join(folder, f"poddeim_{timeindex:05d}.dat")


# Main Routines ===============================================================

def read_and_save_data(data_folder):
    """Extract the data from each of the .dat files in the specified folder."""
    utils.reset_logger()

    # Locate and sort raw .dat files.
    datfiles = sorted(glob.glob(os.path.join(data_folder, "*.dat")))
    num_snapshots = len(datfiles)

    # Allocate space for the snapshots.
    dataset = np.empty((config.DOF*config.NUM_GEMSVARS, num_snapshots),
                        dtype=np.float64)
    times = np.empty(num_snapshots, dtype=np.float64)

    # Extract the data from the .dat files.
    for j,datfile in enumerate(datfiles):

        # Get the contents of a single file.
        with open(datfile, 'r') as infile:
            contents = infile.read()

        # Get the simulation time from the file name.
        simtime = float(_SIMTIME.findall(datfile)[0]) * config.DT

        # Parse and verify the header.
        header_end = _HEADEREND.findall(contents)[0]
        headersize = contents.find(header_end) + len(header_end)
        if int(_ELEMENTS.findall(contents[:headersize])[0]) != config.DOF:
            raise RuntimeError(f"{datfile} DOF != config.DOF")

        # Extract and store the variable data.
        data = contents[headersize:].split()[:dataset.shape[0]],
        dataset[:,j] = np.array(data, dtype=np.float64)
        times[j] = simtime
        print(f"\rProcessed file {j+1:05d}/{num_snapshots}",
              end='', flush=True)

    # Save the data to the appropriate slice.
    save_path = poddeim_data_path()
    with utils.timed_block(f"Saving POD-DEIM data to HDF5"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", data=dataset)
            hf.create_dataset("time", data=times)
    logging.info(f"Data saved to {save_path}.")


def export_to_tecplot(timeindices, variables=None):
    """Convert a snapshot in .h5 format to a .dat file that matches the format
    of grid.dat. The new file is saved in `config.tecplot_path()` with the same
    filename and the new file extension .dat.

    Parameters
    ----------
    timeindices : ndarray(int) or int
        Indices in the full time domain of the snapshots to save.

    variables : str or list(str)
        The variables to scale, a subset of config.ROM_VARIABLES.
        Defaults to all variables.
    """
    utils.reset_logger()

    # Parse parameters.
    if isinstance(timeindices, int):
        timeindices = [timeindices]
    timeindices = np.sort(timeindices)
    t = utils.load_time_domain(timeindices.max()+1)

    # Parse the variables.
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
        num_nodes = int(re.findall(r"Nodes=(\d+)", grid)[0])
        header_end = _HEADEREND.findall(grid)[0]
        headersize = grid.find(header_end) + len(header_end)
        if int(_ELEMENTS.findall(grid[:headersize])[0]) != config.DOF:
            raise RuntimeError(f"{grid_path} DOF and config.DOF do not match")

        # Extract geometry information.
        grid_data = grid[headersize:].split()
        x = grid_data[:num_nodes]
        y = grid_data[num_nodes:2*num_nodes]
        cell_volume = grid_data[2*num_nodes:3*num_nodes]
        connectivity = grid_data[3*num_nodes:]

    # Load the POD-DEIM snapshots that we want.
    data, _ = load_data(cols=timeindices/10)
    with utils.timed_block("Lifting selected POD-DEIM snapshots"):
        lifted_data = dproc.lift(data)
    deim_snaps = np.concatenate([dproc.getvar(v, lifted_data)
                                            for v in variables])

    # Save each of the selected snapshots in Tecplot format matching grid.dat.
    for j,tindex in enumerate(timeindices):

        header = HEADER.format(varnames, tindex, t[tindex],
                               num_nodes, config.DOF,
                               len(variables)+2, "SINGLE "*len(variables))

        save_path = snapshot_path(tindex)
        with utils.timed_block(f"Writing POD-DEIM snapshot {tindex:05d}"):
            with open(save_path, 'w') as outfile:
                # Write the header.
                outfile.write(header)

                # Write the geometry data (x,y coordinates).
                for i in range(0, len(x), NCOLS):
                    outfile.write(' '.join(x[i:i+NCOLS]) + '\n')
                for i in range(0, len(y), NCOLS):
                    outfile.write(' '.join(y[i:i+NCOLS]) + '\n')

                # Write the data for each variable.
                for i in range(0, deim_snaps.shape[0], NCOLS):
                    row = ' '.join(f"{v:.9E}"
                                   for v in deim_snaps[i:i+NCOLS,j])
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
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} --load DATAFOLDER
        python3 {__file__} --export --timeindex I [...] --variables V [...]"""
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--load", type=str, nargs='?',
                       help="read and compile POD-DEIM data from the .dat "
                            "files in the specified folder")
    group.add_argument("--export", action="store_true",
                       help="export POD-DEIM data to tecplot-friendly format")

    parser.add_argument("-snap", "--timeindex", type=int, nargs='*',
                        default=list(range(0,60000,100)),
                        help="snapshot index for basis vs error plots")
    parser.add_argument("-vars", "--variables", type=str, nargs='*',
                        default=config.ROM_VARIABLES,
                        help="variables to save, a subset of "
                             "config.ROM_VARIABLES (default all)")

    # Do the main routine.
    args = parser.parse_args()
    if args.load:
        read_and_save_data(args.datafolder)
    elif args.export:
        export_to_tecplot(args.timeindex, args.variables)
    else:
        print(f"usage: {parser.usage}")
