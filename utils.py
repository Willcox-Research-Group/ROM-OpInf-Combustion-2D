# utils.py
"""Utilities for logging, timing, loading, and saving."""
import os
import sys
import h5py
import time
import signal
import logging
import numpy as np
import matplotlib.pyplot as plt

try:
    import rom_operator_inference as opinf
except ModuleNotFoundError:
    print("\nrom_operator_inference module not installed",
          "(python3 -m pip install --user -r requirements.txt)\n")
    raise
if opinf.__version__ != "1.2.1":
    raise ModuleNotFoundError("rom-operator-inference version 1.2.1 required "
                              "(python3 -m pip install --user "
                              "-r requirements.txt)")

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
    if trainsize == "euler":
        log_filename = config.EULER_LOG
    else:
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

    >>> with timed_block("This is a test"):
    ...     # Code to be timed
    ...     time.sleep(2)
    ...
    This is a test...done in 2.00 s.

    >>> with timed_block("Another test", timelimit=3):
    ...     # Code to be timed and halted within the specified time limit.
    ...     i = 0
    ...     while True:
    ...         i += 1
    Another test...TIMED OUT after 3.00 s.
    """
    verbose = True

    @staticmethod
    def _signal_handler(signum, frame):
        raise TimeoutError("timed out!")

    @property
    def timelimit(self):
        """Time limit (in seconds) for the block to complete."""
        return self._timelimit

    def __init__(self, message, timelimit=None):
        """Store print/log message."""
        self.message = message
        self._end = '\n' if '\r' not in message else ''
        self._timelimit = timelimit

    def __enter__(self):
        """Print the message and record the current time."""
        if self.verbose:
            print(f"{self.message}...", end='', flush=True)
        self._tic = time.time()
        if self._timelimit is not None:
            signal.signal(signal.SIGALRM, self._signal_handler)
            signal.alarm(self._timelimit)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Calculate and report the elapsed time."""
        self._toc = time.time()
        if self._timelimit is not None:
            signal.alarm(0)
        elapsed = self._toc - self._tic
        if exc_type:    # Report an exception if present.
            if self._timelimit is not None and exc_type is TimeoutError:
                print(f"TIMED OUT after {elapsed:.2f} s.",
                      flush=True, end=self._end)
                logging.info(f"TIMED OUT after {elapsed:.2f} s.")
                return True
            print(f"{exc_type.__name__}: {exc_value}")
            logging.info(self.message.strip())
            logging.error(f"({exc_type.__name__}) {exc_value} "
                          f"(raised after {elapsed:.6f} s)")
        else:           # If no exception, report execution time.
            if self.verbose:
                print(f"done in {elapsed:.2f} s.", flush=True, end=self._end)
            logging.info(f"{self.message.strip()}...done in {elapsed:.6f} s.")
        self.elapsed = elapsed
        return


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
        Number of snapshots of scaled data to load. See step2a_transform.py.

    Returns
    -------
    Q : (NUM_ROMVARS*DOF,trainsize) ndarray
        Lifted, scaled, shifted data.
    time_domain : (trainsize) ndarray
        Time domain corresponding to the lifted, scaled data.
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot of the scaled training data.
    scales : (NUM_ROMVARS,) ndarray
        Factors used to scale the variables.
    """
    # Locate the data.
    data_path = _checkexists(config.scaled_data_path(trainsize))

    # Extract the data.
    with timed_block(f"Loading lifted, scaled snapshot data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shapes.
            if hf["data"].shape != (config.NUM_ROMVARS*config.DOF, trainsize):
                raise RuntimeError("data set 'data' has incorrect shape")
            if hf["time"].shape != (trainsize,):
                raise RuntimeError("data set 'time' has incorrect shape")
            if "mean" in hf:
                if hf["mean"].shape != (hf["data"].shape[0],):
                    raise RuntimeError("data set 'mean' has incorrect shape")
                mean = hf["mean"][:]
            else:
                mean = np.zeros(hf["data"].shape[0])
            if hf["scales"].shape != (config.NUM_ROMVARS,):
                raise RuntimeError("data set 'scales' has incorrect shape")

            # Load and return the data.
            return (hf["data"][:], hf["time"][:], mean, hf["scales"][:])


def load_basis(trainsize, r):
    """Load a POD basis and the associated scales.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used when the SVD was computed.
    r : int
        Number of left singular vectors to load.

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis of rank `r`, i.e., the first `r` left singular vectors of
        the training data.
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot that the training data was shifted by after scaling
        but before projection.
    scales : (NUM_ROMVARS,) ndarray
        Factors used to scale the variables before projecting.
    """
    # Locate the data.
    data_path = _checkexists(config.basis_path(trainsize))

    # Secret! Return list of full singular values.
    if r == -1:
        data_path = data_path.replace(config.BASIS_FILE, "svdvals.h5")
        with h5py.File(data_path, 'r') as hf:
            return hf["svdvals"][:]

    # Extract the data.
    with timed_block(f"Loading POD basis from {data_path}"):
        with h5py.File(data_path, 'r') as hf:
            # Check data shapes.
            rmax = hf["basis"].shape[1]
            if r is not None and rmax < r:
                raise ValueError(f"basis only has {rmax} columns")
            if "mean" in hf:
                if hf["mean"].shape != (hf["basis"].shape[0],):
                    raise RuntimeError("basis and mean snapshot not aligned!")
                mean = hf["mean"][:]
            else:
                mean = np.zeros(hf["basis"].shape[0])

            # Load the data.
            return hf["basis"][:,:r], mean, hf["scales"][:]


def load_projected_data(trainsize, r):
    """Load snapshots that have been projected to a low-dimensional subspace.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to load. This is also the number of
        snapshots that were used when the POD basis (SVD) was computed.
    r : int
        Number of retained POD modes used in the projection.

    Returns
    -------
    Q_ : (r,trainsize) ndarray
        Lifted, scaled, projected snapshots.
    Qdot_ : (r,trainsize) ndarray
        Velocity snapshots corresponding to Q_.
    time_domain : (trainsize) ndarray
        Time domain corresponding to the lifted, scaled data.
    """
    # Locate the data.
    data_path = _checkexists(config.projected_data_path(trainsize))

    # Extract the data.
    with timed_block(f"Loading projected training data from {data_path}"):
        with h5py.File(data_path, 'r') as hf:

            # Check data shapes.
            rmax = hf["data"].shape[0]
            if rmax < r:
                raise ValueError(f"basis only has {rmax} columns")
            if hf["data"].shape[1] != trainsize:
                raise RuntimeError("data set 'data' has incorrect shape")
            if hf["ddt"].shape != hf["data"].shape:
                raise RuntimeError("data sets 'data' and 'ddt' not aligned")
            if hf["time"].shape != (trainsize,):
                raise RuntimeError("data set 'time' has incorrect shape")

            # Get the correct rows of the saved projection data.
            return hf["data"][:r], hf["ddt"][:r], hf["time"][:]


def load_rom(trainsize, r, regs):
    """Load a single trained ROM.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.
    r : int
        Dimension of the ROM. Also the number of retained POD modes (left
        singular vectors) used to project the training data.
    regs : one, two, or three positive floats
        Regularization hyperparameters used in the Operator Inference
        least-squares problem for training the ROM.

    Returns
    -------
    rom : opinf.InferredContinuousROM
        The trained reduced-order model.
    """
    # Locate the data.
    data_path = _checkexists(config.rom_path(trainsize, r, regs))

    # Extract the trained ROM.
    try:
        rom = opinf.load_model(data_path)
    except FileNotFoundError as e:
        raise DataNotFoundError(f"could not locate ROM with {trainsize:d} "
                                f"training snapshots, r={r:d}, and "
                                f"{config.REGSTR(regs)}") from e
    # Check ROM dimension.
    if rom.r != r:
        raise RuntimeError(f"rom.r = {rom.r} != {r}")

    rom.trainsize, rom.regs = trainsize, regs
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
        plt.savefig(save_path, bbox_inches="tight", dpi=250)
        plt.close(plt.gcf())
