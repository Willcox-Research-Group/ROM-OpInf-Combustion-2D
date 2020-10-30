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
    Q : (NUM_ROMVARS*DOF,trainsize) ndarray
        The lifted, scaled data.

    time_domain : (trainsize) ndarray
        The time domain corresponding to the lifted, scaled data.

    scales : (NUM_ROMVARS,2) ndarray
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
            if hf["scales"].shape != (config.NUM_ROMVARS, 2):
                raise RuntimeError(f"data set 'scales' has incorrect shape")

            # Load and return the data.
            return hf["data"][:,:], hf["time"][:], hf["scales"][:,:]


def load_basis(trainsize, r1, r2):
    """Load a POD basis and associated singular values.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used when the SVD was computed.

    r1 : int
        The number of left singular vectors to load for the first basis
        (pressure and velocities).

    r2 : int
        The number of left singular vectors to load for the second basis
        (temperature, specific volume, and chemical species).

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r1+r2) ndarray
        Complete block-structure POD basis.

    scales : (NUM_ROMVARS,2) ndarray
        The min/max factors used to scale the variables before projecting.
    """
    # Locate the data.
    data_path = _checkexists(config.basis_path(trainsize))

    # Extract the data.
    with timed_block(f"Loading POD basis from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shapes.
            rmax1 = hf["basis1/V"].shape[1]
            if r1 is not None and rmax1 < r1:
                raise ValueError(f"first basis only has {rmax1} columns")
            rmax2 = hf["basis1/V"].shape[1]
            if r2 is not None and rmax2 < r2:
                raise ValueError(f"T basis only has {rmax2} columns")

            # Load the data.
            V1 = hf["basis1/V"][:,:r1]
            V2 = hf["basis2/V"][:,:r2]
            scales = hf["scales"][:]

        # Put the two basis blocks together, preserving variable order.
        return _assemble_basis(V1, V2), scales


def _assemble_basis(V1, V2):
    """Piece the two bases together in a block structure, preserving
    the order of the variables as given in the configuration file, i.e.,

            [V1   0]
        V = [ 0  V2].

    Parameters
    ----------
    V1 : ((NUM_ROMVARS-1)*DOF,r1) ndarray
        POD basis for pressure and velocities.

    V2 : (DOF,r2) ndarray
        POD basis for temperature, specific volume, and chemical species.

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r1+r2) ndarray
        Complete block-structure POD basis.
    """
    # Check shapes.
    N = config.DOF * config.NUM_ROMVARS
    if V1.shape[0] + V2.shape[0] != N:
        raise RuntimeError("illegal basis size (row count off)")
    r1, r2 = V1.shape[1], V2.shape[1]
    n1 = V1.shape[0]

    # Assemble block-diagonal matrix.
    V = np.zeros((N,r1+r2))
    V[:n1,:r1] = V1
    V[n1:,r1:] = V2

    return V


def get_basis_size(trainsize):
    """Get the number of saved POD basis vectors for the given `trainsize`.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used when the SVD was computed.

    Returns
    -------
    r1 : int
        Number of left singular vectors that have been saved for the
        first basis.

    r2 : int
        Number of left singular vectors that have been saved for the
        second basis.
    """
    # Locate the data.
    data_path = _checkexists(config.basis_path(trainsize))

    # Extract the data.
    with h5py.File(data_path, 'r') as hf:
        return hf["basis1/V"].shape[1], hf["basis2/V"].shape[1]


def load_projected_data(trainsize, r1, r2):
    """Load snapshots that have been projected to a low-dimensional subspace.

    Parameters
    ----------
    trainsize : int
        The number of snapshots to load. This is also the number of
        snapshots that were used when the POD basis (SVD) was computed.

    r1 : int
        The number of retained POD modes used in the first projection.

    r2 : int
        The number of retained POD modes used in the second projection.

    Returns
    -------
    Q_ : (r1+r2,trainsize) ndarray
        The lifted, scaled, projected snapshots.

    Qdot_ : (r1+r2,trainsize) ndarray
        Velocity snapshots corresponding to Q_.

    time_domain : (trainsize) ndarray
        The time domain corresponding to the lifted, scaled data.
    """
    # Locate the data.
    data_path = _checkexists(config.projected_data_path(trainsize))

    # Extract the data.
    with timed_block(f"Loading projected training data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:

            # Check data shapes.
            R1, R2 = hf["rs"][:]
            if hf["data"].shape[0] != R1 + R2:
                raise RuntimeError(f"data sets 'data' and 'rs' not aligned")
            if R1 < r1:
                raise ValueError(f"first basis only has {R1} modes")
            if R2 < r2:
                raise ValueError(f"second basis only has {R2} modes")
            if hf["data"].shape[1] != trainsize:
                raise RuntimeError(f"data set 'data' has incorrect shape")
            if hf["ddt"].shape != hf["data"].shape:
                raise RuntimeError(f"data sets 'data' and 'ddt' not aligned")
            if hf["time"].shape != (trainsize,):
                raise RuntimeError(f"data set 'time' has incorrect shape")

            # Get the correct rows of the saved projection data.
            Q_ = np.row_stack([hf["data"][:r1], hf["data"][R1:R1+r2]])
            Qdot_ = np.row_stack([hf["ddt"][:r1], hf["ddt"][R1:R1+r2]])
            return Q_, Qdot_, hf["time"][:]


def load_rom(trainsize, r1, r2, reg):
    """Load a single trained ROM.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r1 : int
        Number of retained POD modes (left singular vectors) used to project
        the training data with the first basis.

    r2 : int
        Number of retained POD modes (left singular vectors) used to project
        the training data with the second basis.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROM.

    Returns
    -------
    rom : roi.InferredContinuousROM
        The trained reduced-order model.
    """
    # Locate the data.
    data_path = _checkexists(config.rom_path(trainsize, r1, r2, reg))

    # Extract the trained ROM.
    try:
        rom = roi.load_model(data_path)
    except FileNotFoundError as e:
        raise DataNotFoundError(f"could not locate ROM with {trainsize:d} "
                                f"training snapshots, r1={r1:d}, r2={r2:d}, "
                                f"and reg={reg:e}") from e
    # Check data shapes.
    if rom.r != r1 + r2:
        raise RuntimeError(f"rom.r = {rom.r} != {r1+r2}")

    rom.trainsize = trainsize
    rom.reg = reg
    return rom


def load_spatial_statistics(keys, k=None):
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
        * "T_mean" -> spatially averaged temperature
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
    with timed_block(f"Loading statistical features from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            if len(keys) == 1:
                return hf[f"space/{keys[0]}"][k], hf["t"][k]
            return {key: hf[f"space/{key}"][k] for key in keys}, hf["t"][k]


def load_temporal_statistics(keys):
    """Load statistical features of the lifted data, computed over the
    temporal domain at each spatial point.

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
        * "T_mean" -> time-averaged temperature
        * "vx_min" -> minimum x-velocity
        * "CH4_sum" -> methane molar concentration time integral

    Returns
    -------
    features : dict(str -> (N,) ndarray) or (N,) ndarray
        Dictionary of statistical feature arrays with keys `keys`.
        If only one key is given, return the actual array, not a dict.
    """
    # Locate the data.
    data_path = _checkexists(config.statistical_features_path())

    # Parse arguments.
    if isinstance(keys, str):
        keys = [keys]

    # Extract the data.
    with timed_block(f"Loading statistical features from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            if len(keys) == 1:
                return hf[f"time/{keys[0]}"][:]
            return {key: hf[f"time/{key}"][:] for key in keys}


# Figure saving ===============================================================

def save_figure(figname):
    """Save the current matplotlib figure to the figures folder."""
    save_path = os.path.join(config.figures_path(), figname)
    # plt.show() # Uncomment to display figure before saving.
    with timed_block(f"Saving {save_path}"):
        plt.savefig(save_path, bbox_inches="tight", dpi=1200)
        plt.close(plt.gcf())
