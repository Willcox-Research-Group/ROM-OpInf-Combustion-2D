# utils.py
"""Utilities for logging, timing, loading, and saving."""
import os
import re
import sys
import glob
import h5py
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

try:
    import rom_operator_inference as roi
except ModuleNotFoundError:
    print("\nrom_operator_inference module not installed",
          "(python3 -m pip install --user rom-operator-inference)\n")
    raise

import config


# Logging / timing tools ======================================================

def reset_logger(trainsize=None):
    """Switch to the log file within the folder for experiments with
    `trainsize` snapshots of training data. If `trainsize` is None,
    switch to the log file in the base folder.
    """
    # Remove all old logging handlers.
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    # Get the log filename and append a newline.
    log_filename = config.log_path(trainsize)
    with open(log_filename, 'a') as lf:
        lf.write('\n')

    # Get a new logging handler to the log file.
    handler = logging.FileHandler(log_filename, 'a')
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    print(f"Logging to {log_filename}")

    # Log the session header.
    if hasattr(sys.modules["__main__"], "__file__"):
        _front = f"({os.path.basename(sys.modules['__main__'].__file__)})"
        _end = time.strftime('%Y-%m-%d %H:%M:%S')
        _mid = '-' * (79 - len(_front) - len(_end) - 20)
        header = f"NEW SESSION {_front} {_mid} {_end}"
    else:
        header = f"NEW SESSION {time.strftime(' %Y-%m-%d %H:%M:%S'):->61}"
    logging.info(header)


class timed_block:
    """Context manager for timing a block of code and reporting the timing.

    >>> with timed_operation("This is a test"):
    ...     # Code to be timed
    ...     time.sleep(2)
    ...
    This is a test...done in 2.00 s.
    """
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        """Print the message and record the current time."""
        print(f"{self.message}...", end='', flush=True)
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Calculate and report the elapsed time."""
        self.end = time.time()
        elapsed = self.end - self.start
        if exc_type:    # Report an exception if present.
            print(f"{exc_type.__name__}: {exc_value}")
            logging.info(self.message.strip())
            logging.error(f"({exc_type.__name__}) {exc_value} "
                          f"(raised after {elapsed:.6f} s)")
        else:           # If no exception, report execution time.
            print(f"done in {elapsed:.2f} s.", flush=True)
            logging.info(f"{self.message.strip()}...done in {elapsed:.6f} s.")


# Data loaders ================================================================

class DataNotFoundError(FileNotFoundError):
    """Exception to be raised when attempting to load a missing data file for
    * GEMS simulation data
    * Scaled training data
    * POD basis
    * Projected training data
    * Trained ROMs
    """
    pass


def _checkexists(filename):
    """Check that the file `filename` exists; if not, raise an exception."""
    if not os.path.isfile(filename):
        raise DataNotFoundError(filename)
    return filename


def load_gems_data(rows=None, cols=None):
    """Load the indicated rows and colums of GEMS simulation data.
    This is a large file, so try to only load what is needed at the moment.

    Parameters
    ----------
    rows : int, slice, or (nrows,) ndarray of integer indices
        Which rows (spatial locations) to extract from the data (default all).
        If an integer, extract the first `rows` rows.

    cols : int or slice
        Which columns (temporal points) to extract from the data (default all).
        If an integer, extract the first `cols` columns.

    Returns
    -------
    gems_data : (nrows,ncols) ndarray
        The indicated rows / columns of the data.

    time_domain : (ncols,) ndarray
        The time (in seconds) associated with each column of extracted data.
    """
    # Locate the data.
    data_path = _checkexists(config.gems_data_path())

    # Ensure rows are loaded in ascending index order (HDF5 requirement).
    if isinstance(rows, (np.ndarray, list)):
        row_order = np.argsort(rows)
        rows = np.array(rows, copy=True)[row_order]
        old_row_order = np.argsort(row_order)
    elif np.isscalar(rows) or rows is None:
        rows = slice(None, rows)
    if np.isscalar(cols) or cols is None:
        cols = slice(None, cols)

    # Extract the data.
    NROWS = config.NUM_GEMSVARS * config.DOF
    with timed_block(f"Loading GEMS simulation data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shape.
            if hf["data"].shape[0] != NROWS:
                raise RuntimeError(f"data should have exactly {NROWS} rows")
            gems_data = hf["data"][rows,cols]
            time_domain = hf["time"][cols]

    # Restore row ordering if needed.
    if isinstance(rows, np.ndarray):
        gems_data = gems_data[old_row_order,:]

    return gems_data, time_domain


def load_time_domain(nsteps=None):
    """Load the time domain corresponding to the GEMS simulation data.

    Parameters
    ----------
    nsteps : int or None
        How many entries to extract from the time domain (default all).
    """
    # Locate the data.
    data_path = _checkexists(config.gems_data_path())

    # Extract the data.
    with timed_block(f"Loading time domain data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            time_domain = hf["time"][:nsteps]

    # Check time spacing.
    if not np.allclose(np.diff(time_domain), config.DT):
        raise ValueError("time domain DT != config.DT")

    # If a larger number of time steps is requested, use np.linspace().
    if np.isscalar(nsteps) and time_domain.size < nsteps:
        t0 = time_domain[0]
        return np.linspace(t0, t0 + nsteps*config.DT, nsteps)

    return time_domain


def load_scaled_data(trainsize):
    """Load scaled snapshot data and the associated scaling factors.

    Parameters
    ----------
    trainsize : int
        The number of snapshots of scaled data to load. See step1b.py.

    Returns
    -------
    X : (NUM_ROMVARS*DOF,trainsize) ndarray
        The lifted, scaled data.

    time_domain : (trainsize) ndarray
        The time domain corresponding to the lifted, scaled data.

    scales : (NUM_ROMVARS,4) ndarray
        The min/max factors used to scale the variables.
    """
    # Locate the data.
    data_path = _checkexists(config.scaled_data_path(trainsize))

    # Extract the data.
    with timed_block(f"Loading lifted, scaled snapshot data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shapes.
            if hf["data"].shape != (config.NUM_ROMVARS*config.DOF, trainsize):
                raise RuntimeError(f"data set 'data' has incorrect shape")
            if hf["time"].shape != (trainsize,):
                raise RuntimeError(f"data set 'time' has incorrect shape")
            if hf["scales"].shape != (config.NUM_ROMVARS, 4):
                raise RuntimeError(f"data set 'scales' has incorrect shape")

            # Load and return the data.
            return hf["data"][:,:], hf["time"][:], hf["scales"][:,:]


def load_basis(trainsize, r):
    """Load a POD basis and associated singular values.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used when the SVD was computed.

    r : int
        The number of left singular vectors/values to load.

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r) ndarray
        The POD basis of rank `r` (the first `r` left singular vectors).

    svdvals : (r,) ndarray
        The first `r` singular values.
    """
    # Locate the data.
    try:
        data_path = config.smallest_basis_path(trainsize, r)
    except FileNotFoundError as e:
        raise DataNotFoundError(e) from e

    # Extract the data.
    with timed_block(f"Loading POD basis from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shapes.
            if hf["V"].shape[1] < r:
                raise RuntimeError(f"data set 'V' has fewer than {r} columns")

            # Load and return the data.
            return hf["V"][:,:r], hf["svdvals"][:r]


def load_projected_data(trainsize, num_modes):
    """Load snapshots that have been projected to a low-dimensional subspace.

    Parameters
    ----------
    trainsize : int
        The number of snapshots to load. This is also the number of
        snapshots that were used when the POD basis (SVD) was computed.

    num_modes : int
        The number of retained POD modes used in the projection.

    Returns
    -------
    X_ : (num_modes,trainsize) ndarray
        The lifted, scaled, projected snapshots.

    Xdot_ : (num_modes,trainsize) ndarray
        Velocity snapshots corresponding to X_.

    time_domain : (trainsize) ndarray
        The time domain corresponding to the lifted, scaled data.

    scales : (NUM_ROMVARS,4) ndarray
        The min/max factors used to scale the variables before projecting.
    """
    # Locate the data.
    data_path = _checkexists(config.projected_data_path(trainsize, num_modes))

    # Extract the data.
    _data_shape = (num_modes, trainsize)
    with timed_block(f"Loading projected training data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:

            # Check data shapes.
            if hf["data"].shape != _data_shape:
                raise RuntimeError(f"data set 'data' has incorrect shape")
            if hf["xdot"].shape != _data_shape:
                raise RuntimeError(f"data set 'xdot' has incorrect shape")
            if hf["time"].shape != (trainsize,):
                raise RuntimeError(f"data set 'time' has incorrect shape")
            if hf["scales"].shape != (config.NUM_ROMVARS, 4):
                raise RuntimeError(f"data set 'scales' has incorrect shape")

            return hf["data"][:,:], hf["xdot"][:,:], \
                   hf["time"][:], hf["scales"][:,:]


def load_rom(trainsize, num_modes, reg):
    """Load a single trained ROM.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    num_modes : int
        The dimension of the ROM. This is also the number of retained POD modes
        (left singular vectors) used to project the training data.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROM.

    Returns
    -------
    rom : roi.InferredContinuousROM
        The trained reduced-order model.
    """
    # Locate the data.
    data_path = _checkexists(config.rom_path(trainsize, num_modes, reg))

    # Extract the trained ROM.
    try:
        rom = roi.load_model(data_path)
    except FileNotFoundError as e:
        raise DataNotFoundError(f"could not locate ROM with {trainsize:d} "
                                f"training snapshots, r={num_modes}, and "
                                f"reg={reg:e}") from e
    # Check data shapes.
    if rom.r != num_modes:
        raise RuntimeError(f"rom.r = {rom.r} != {num_modes}")

    return rom


def load_all_roms_r(trainsize, num_modes):
    """Load all trained ROM of the same dimension for a given training size.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROMs. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    num_modes : int
        The dimension of the ROMs. This is also the number of retained POD
        modes (left singular vectors) used to project the training data.

    Returns
    -------
    regs : list(float)
        The regularization factors corresponding to the ROMs.

    roms : list(roi.InferredContinuousROM)
        The trained reduced-order models.
    """
    # Find the ROM files of interest.
    folder = os.path.join(config.BASE_FOLDER,
                          config.TRNFMT(trainsize), config.DIMFMT(num_modes))
    pat = os.path.join(folder, f"{config.ROM_PREFIX}_{config.REG_PREFIX}*.h5")
    romfiles = sorted(glob.glob(pat))
    if not romfiles:
        raise DataNotFoundError(f"no trained ROMs with {trainsize:d} "
                                f"training snapshots and {num_modes:d} "
                                f"retained POD modes")

    # Load the files (in sorted order).
    regs = sorted(float(re.findall(fr"_{config.REG_PREFIX}(.+?)\.h5", s)[0])
                                                          for s in romfiles)
    roms = [load_rom(trainsize, num_modes, reg) for reg in regs]

    return regs, roms


def load_all_roms_reg(trainsize, reg):
    """For a given training size, load all trained ROM that were trained
    with the same regularization parameter.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROMs. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROMs.

    Returns
    -------
    rs : list(int)
        The dimension (number of retained POD modes) of each ROM.

    roms : list(roi.InferredContinuousROM)
        The trained reduced-order models.
    """
    # Find the rom files of interest.
    pat = os.path.join(config.BASE_FOLDER,
                       config.TRNFMT(trainsize),
                       f"r*",
                       f"{config.ROM_PREFIX}_{config.REGFMT(reg)}.h5")
    romfiles = glob.glob(pat)
    if not romfiles:
        raise DataNotFoundError(f"no trained ROMs with {trainsize:d} "
                                f"training snapshots and regularization "
                                f"factor {reg:e}")

    # Load the files (in sorted order).
    rs = sorted(int(re.findall(r"r(\d+)", s)[0]) for s in romfiles)
    roms = [load_rom(trainsize, r, reg) for r in rs]

    return rs, roms


def load_statistical_features(keys, k=None):
    """Load statistical features of the lifted data, computed over the
    spatial domain at each point in time.

    Parameters
    ----------
    keys : list(str)
        Which data set(s) to load. Options:
        * {var}_min : minimum of variable var
        * {var}_max : maximum of variable var
        * {var}_sum : sum (integral) of variable var
        * {var}_std : standard deviation of variable var
        * {var}_mean : mean of variable var
        Here var is a member of config.ROM_VARIABLES. Examples:
        * "T_mean" -> mean temperature
        * "vx_min" -> minimum x-velocity
        * "CH4_sum" -> methane molar concentration integral

    k : int, slice, or one-dimensional ndarray of sorted integer indices
        Number of time steps of data to load (default all).

    Returns
    -------
    features : dict(str -> (k,) ndarray) or (k,) ndarray
        Dictionary of statistical feature arrays with keys `keys`.
        If only one key is given, return the actual array, not a dict.

    t : (k,) ndarray
        Time domain corresponding to the statistical features.
    """
    # Locate the data.
    data_path = _checkexists(config.statistical_features_path())

    # Parse arguments.
    if isinstance(keys, str):
        keys = [keys]
    elif keys is None:
        keys = ["T_mean"] + [f"{spc}_int" for spc in config.SPECIES]
    if np.isscalar(k) or k is None:
        k = slice(None, k)

    # Extract the data.
    features = {}
    with timed_block(f"Loading statistical features from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            if len(keys) == 1:
                return hf[keys[0]][k], hf["time"][k]
            else:
                return {key: hf[key][k] for key in keys}, hf["time"][k]


# Figure saving ===============================================================

def save_figure(figname):
    """Save the current matplotlib figure to the figures folder."""
    save_path = os.path.join(config.figures_path(), figname)
    # plt.show() # Uncomment to display figure before saving.
    with timed_block(f"Saving {save_path}"):
        plt.savefig(save_path, bbox_inches="tight", dpi=1200)
        plt.close(plt.gcf())
