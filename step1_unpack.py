# step1_unpack.py
"""Read the GEMS output directly from .tar archives, compile it into a single
data set of size (num_variables * domain_size)x(num_snapshots), and save it
in HDF5 format.

The GEMS data for this project collects snapshots of the following variables.

* Pressure
* x-velocity
* y-velocity
* Temperature
* CH4 mass fraction
* O2  mass fraction
* H2O mass fraction
* CO2 mass fraction

Each variable is discretized over a domain with 38,523 degrees of freedom
(DOF), so each snapshot has 8 * 38,523 = 308,184 entries.

Examples
--------
# Process the raw .tar data files that are placed in /storage/combustion/.
$ python3 step1_unpack.py /storage/combustion

# Process the raw .tar data files that are placed in the current directory,
# overwriting the resulting HDF5 file if it already exists.
$ python3 step1_unpack.py . --overwrite

# Process the raw .tar data files in /storage/combustion/ serially
# (not in parallel, which is the default).
$ python3 step1_unpack.py /storage/combustion --serial

Loading Results
---------------
>>> import utils
>>> gems_data, time_domain = utils.load_gems_data()

Command Line Arguments
----------------------
"""
import os
import re
import glob
import h5py
import shutil
import logging
import tarfile
import numpy as np
import multiprocessing as mp

import config
import utils


# Regular expressions
_SIMTIME = re.compile(r"_(\d+).dat")        # Simulation time from file name
_HEADEREND = re.compile(r"DT=.*?\n")        # Last line in .dat headers
_ELEMENTS = re.compile(r"Elements=(\d+)")   # DOF listed in .dat headers


def _read_tar_and_save_data(tfile, start, stop, parallel=True):
    """Read snapshot data directly from a .tar archive (without untar-ing it)
    and copy the data to the snapshot matrix HDF5 file config.GEMS_DATA_FILE.

    Parameters
    ----------
    tfile : str
        Name of a .tar file to read data from.
    start : int
        Index of the first snapshot contained in the .tar file.
    stop : int
        Index of the last snapshot contained in the .tar file.
    parallel : bool
        If True, then only print progress if start == 0 and lock / unlock
        when writing to the HDF5 file.
    """
    # Allocate space for the snapshots in this .tar file.
    num_snapshots = stop - start
    gems_data = np.empty((config.DOF*config.NUM_GEMSVARS, num_snapshots),
                         dtype=np.float64)
    times = np.empty(num_snapshots, dtype=np.float64)

    # Extract the data from the .tar file.
    with tarfile.open(tfile, 'r') as archive:
        for j,tarinfo in enumerate(archive):

            # Read the contents of one file.
            with archive.extractfile(tarinfo) as datfile:
                contents = datfile.read().decode()

            # Get the simulation time from the file name.
            simtime = float(_SIMTIME.findall(tarinfo.name)[0]) * config.DT

            # Parse and verify the header.
            header_end = _HEADEREND.findall(contents)[0]
            headersize = contents.find(header_end) + len(header_end)
            if int(_ELEMENTS.findall(contents[:headersize])[0]) != config.DOF:
                raise RuntimeError(f"{tarinfo.name} DOF != config.DOF")

            # Extract and store the variable data.
            data = contents[headersize:].split()[:gems_data.shape[0]],
            gems_data[:,j] = np.array(data, dtype=np.float64)
            times[j] = simtime
            if start == 0 or not parallel:
                print(f"\rProcessed file {j+1:05d}/{num_snapshots}",
                      end='', flush=True)
    if start == 0 or not parallel:
        print()

    # Save the data to the appropriate slice.
    save_path = config.gems_data_path()
    if parallel:
        lock.acquire()  # Only allow one process to open the file at a time.
    with utils.timed_block(f"Saving snapshots {start}-{stop} to HDF5"):
        with h5py.File(save_path, 'a') as hf:
            hf["data"][:,start:stop] = gems_data
            hf["time"][start:stop] = times
    print(f"Data saved to {save_path}.", flush=True)
    if parallel:
        lock.release()  # Let other processes resume.


def _globalize_lock(L):
    global lock
    lock = L


def main(data_folder, overwrite=False, serial=False):
    """Extract snapshot data, in parallel, from the .tar files in the
    specified folder of the form Data_<first-snapshot>to<last-snapshot>.tar.

    Parameters
    ----------
    data_folder : str
        Path to the folder that contains the raw GEMS .tar data files,
        preferably as an absolute path (e.g., /path/to/folder).
    overwrite : bool
        If False and the snapshot matrix file exists, raise an error.
        If True, overwrite the existing snapshot matrix file if it exists.
    serial : bool
        If True, do the unpacking sequentially in 10,000 snapshot chunks.
        If False, do the unpacking in parallel with 10,000 snapshot chunks.
    """
    utils.reset_logger()

    # If it exists, copy the grid file to the Tecplot data directory.
    source = os.path.join(data_folder, config.GRID_FILE)
    if os.path.isfile(source):
        target = config.grid_data_path()
        with utils.timed_block(f"Copying {source} to {target}"):
            shutil.copy(source, target)
    else:
        logging.warning(f"Grid file {source} not found!")

    # Locate and sort raw .tar files.
    target_pattern = os.path.join(data_folder, "Data_*to*.tar")
    tarfiles = sorted(glob.glob(target_pattern))
    if not tarfiles:
        raise FileNotFoundError(target_pattern)

    # Get the snapshot indices corresponding to each file from the file names.
    starts, stops = [], []
    for i,tfile in enumerate(tarfiles):
        matches = re.findall(r"Data_(\d+)to(\d+).tar", tfile)
        if not matches:
            raise ValueError(f"file {tfile} not named with convention "
                             "Data_<first-snapshot>to<last-snapshot>.tar")
        start, stop = [int(d) for d in matches[0]]
        if i == 0:
            start0 = start  # Offset
        starts.append(start - start0)
        stops.append(stop + 1 - start0)

        if i > 0 and stops[i-1] != starts[i]:
            raise ValueError(f"file {tfile} not continuous from previous set")
    num_snapshots = stops[-1]

    # Create an empty HDF5 file of appropriate size for the data.
    save_path = config.gems_data_path()
    if os.path.isfile(save_path) and not overwrite:
        raise FileExistsError(f"{save_path} (use --overwrite to overwrite)")
    with utils.timed_block("Initializing HDF5 file for data"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", shape=(config.DOF*config.NUM_GEMSVARS,
                                             num_snapshots),
                              dtype=np.float64)
            hf.create_dataset("time", shape=(num_snapshots,), dtype=np.float64)
    logging.info(f"Data file initialized as {save_path}.")

    # Read the files in chunks.
    args = zip(tarfiles, starts, stops)
    if serial:       # Read the files serially (sequentially).
        for tf, start, stop in args:
            _read_tar_and_save_data(tf, start, stop, parallel=False)
    else:            # Read the files in parallel.
        with mp.Pool(initializer=_globalize_lock, initargs=(mp.Lock(),),
                     processes=min([len(tarfiles), mp.cpu_count()])) as pool:
            pool.starmap(_read_tar_and_save_data, args)


# =============================================================================
if __name__ == '__main__':
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} DATAFOLDER [--overwrite] [--serial]"""

    parser.add_argument("datafolder", type=str,
                        help="folder containing the raw GEMS .tar data files")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite the existing HDF5 data file")
    parser.add_argument("--serial", action="store_true",
                        help="do the unpacking sequentially, not in parallel")

    # Do the main routine.
    args = parser.parse_args()
    main(args.datafolder, args.overwrite, args.serial)
