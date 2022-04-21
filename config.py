# config.py
"""Configuration file containing project directives for file and folder names,
constants for the geometry and chemistry of the problem, plot customizations,
and so forth.

New users should set the BASE_FOLDER variable to the location of the data,
preferably as an absolute path. Other global variables specify the naming
conventions for the various data files.
"""
import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt


# File structure --------------------------------------------------------------

# CHANGE THIS LINE vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

BASE_FOLDER = ""                            # Base folder for all data.

# CHANGE THIS LINE ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FIGURES_FOLDER = "figures"                  # Name of saved figures folder.
TECPLOT_FOLDER = "tecdata"                  # Name of Tecplot data folder.

GEMS_DATA_FILE = "gems.h5"                  # Name of GEMS data file.
SCALED_DATA_FILE = "data_scaled.h5"         # Name of scaled data files.
BASIS_FILE = "basis.h5"                     # Name of POD basis files.
PROJECTED_DATA_FILE = "data_projected.h5"   # Name of projected data files.
ROM_INDEX_FILE = "roms.json"                # Name of ROM index files.
FEATURES_FILE = "statistical_features.h5"   # Name of statistical feature file.
GRID_FILE = "grid.dat"                      # Name of Tecplot grid data file.
LOG_FILE = "log.log"                        # Name of log files.

TRN_PREFIX = "k"                            # Prefix for training size folders.
DIM_PREFIX = "r"                            # Prefix for ROM dimension folders.
ROM_PREFIX = "rom"                          # Prefix for trained ROM files.
REG_PREFIX = "reg"                          # Prefix for regularization values.


def TRNFMT(k):
    """String format for trianing sizes."""
    return f"{TRN_PREFIX}{k:05d}"


def DIMFMT(r):
    """String format for ROM dimensions."""
    return f"{DIM_PREFIX}{r:03d}"


def REGFMT(λs):
    """String format for the regularization parmeters."""
    if np.isscalar(λs) or len(λs) != 2 or any(λ < 0 for λ in λs):
        raise ValueError(f"invalid regularization parameters {λs}")
    return REG_PREFIX + "_".join(f"{λ:06.0f}" for λ in λs)


def REGSTR(λs):
    """[x,y,z] -> 'λ1=x, λ2=y, λ3=z'."""
    if np.isscalar(λs):
        return f"λ={λs:5e}"
    return ", ".join(f"λ{i+1}={λ:4e}" for i,λ in enumerate(λs))


# Domain geometry -------------------------------------------------------------

DOF = 38523                                 # Spatial degrees of freedom.

MONITOR_LOCATIONS = [
    36915,                                  # Index of monitor location 1.
    37886,                                  # Index of monitor location 2.
    36443,                                  # Index of monitor location 3.
    6141,                                   # Index of monitor location 4.
]

DT = 1e-7                                   # Temporal resolution of snapshots.

# GEMS and Learning Variables -------------------------------------------------

# Chemical species in the combustion reaction.
SPECIES = ["CH4", "O2", "H2O", "CO2"]

# Solution variables for the raw GEMS data.
GEMS_VARIABLES = ["p", "vx", "vy", "T"] + SPECIES

# Variables that the ROM learns from and makes predictions for.
ROM_VARIABLES = ["p", "vx", "vy", "T", "xi"] + SPECIES

NUM_SPECIES = len(SPECIES)                  # Number of chemical species.
NUM_GEMSVARS = len(GEMS_VARIABLES)          # Number of GEMS variables.
NUM_ROMVARS = len(ROM_VARIABLES)            # Number of learning variables.


# Chemistry and physics -------------------------------------------------------

MOLAR_MASSES = [16.04,                      # Molar mass of CH4 [kg/kmol].
                32.0,                       # Molar mass of O2  [kg/kmol].
                18.0,                       # Molar mass of H2O [kg/kmol].
                44.01]                      # Molar mass of CO2 [kg/kmol].

R_UNIVERSAL = 8.3144598                     # Univ. gas constant [J/(mol K)].


def U(t):
    """Input function for pressure oscillation."""
    return 1e6*(1 + 0.1*np.sin(np.pi*1e4*t))


# Matplotlib plot customization -----------------------------------------------

plt.rc("figure", dpi=300)                   # High-quality figures.
plt.rc("text", usetex=True)                 # Use LaTeX fonts.
plt.rc("font", family="serif")              # Serif axis labels.
plt.rc("legend", edgecolor="none",          # No borders around legends.
                 frameon=False)             # No legend backgrounds.
plt.rc("axes", linewidth=.5)                # Thinner plot borders.

# Names of the learning variables (for LaTeX fonts).
VARTITLES = {
    "p": "Pressure",
    "vx": "$x$-velocity",
    "vy": "$y$-velocity",
    "T": "Temperature",
    "xi": r"$\xi$",
    "CH4": "CH$_4$",
    "O2": "O$_2$",
    "H2O": "H$_2$O",
    "CO2": "CO$_2$",
}

# Units of the learning variables (for LaTeX fonts).
VARUNITS = {
    "p": "Pa",
    "vx": "m/s",
    "vy": "m/s",
    "T": "K",
    "xi": "m$^3$/kg",
    "CH4": "kmol/m$^3$",
    "O2": "kmol/m$^3$",
    "H2O": "kmol/m$^3$",
    "CO2": "kmol/m$^3$",
}

# Learning variable titles with units (for LaTeX fonts).
VARLABELS = {v: f"{VARTITLES[v]} [{VARUNITS[v]}]" for v in ROM_VARIABLES}

# Line plot customizations (keyword argments for plt.plot()).
GEMS_STYLE = dict(linestyle="-",            # Line style for GEMS time traces.
                  color="C1",               # Line color for GEMS time traces.
                  label="GEMS")             # Line label for GEMS time traces.
ROM_STYLE = dict(linestyle="--",            # Line style for ROM time traces.
                 color="C0",                # Line color for ROM time traces.
                 label="ROM")               # Line label for ROM time traces.


# =============================================================================
# DO NOT MODIFY ===============================================================


def _makefolder(*args):
    """Join arguments into a path to a folder. If the folder doesn't exist,
    make the folder as well. Return the resulting path.
    """
    folder = os.path.join(*args)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    return folder


def gems_data_path():
    """Return the path to the file containing the raw GEMS data."""
    return os.path.join(BASE_FOLDER, GEMS_DATA_FILE)


def log_path(trainsize=None):
    """Return the path to the logging file for all experiments with `trainsize`
    snapshots of training data. If `trainsize` is None, return the path to the
    base logging file in `BASE_FOLDER`.
    """
    folder = BASE_FOLDER if not trainsize else _makefolder(BASE_FOLDER,
                                                           TRNFMT(trainsize))
    return os.path.join(folder, LOG_FILE)


def scaled_data_path(trainsize):
    """Return the path to the file containing `trainsize` lifted, scaled
    snapshots.
    """
    folder = _makefolder(BASE_FOLDER, TRNFMT(trainsize))
    return os.path.join(folder, SCALED_DATA_FILE)


def basis_path(trainsize):
    """Return the path to the file containing the POD basis computed from
    `trainsize` lifted, scaled snapshots.
    """
    return os.path.join(BASE_FOLDER, TRNFMT(trainsize), BASIS_FILE)


def projected_data_path(trainsize):
    """Return the path to the file containing `trainsize` training snapshots,
    projected with a POD basis computed from `trainsize` high-fidelity
    snapshots.
    """
    return os.path.join(BASE_FOLDER, TRNFMT(trainsize), PROJECTED_DATA_FILE)


def rom_path(trainsize, r, regs, overwrite=False):
    """Return the path to a file containing an OpInf ROM.

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
    overwrite : bool
        If True, make a new ROM filename and overwrite any previous instances
        with the same trainsize, r, and regs in the ROM index.

    Returns
    -------
    filename : str
        Path to a file to save an OpInf ROM to or load an OpInf ROM from.
    """
    rlabel = DIMFMT(r)
    kfolder = _makefolder(BASE_FOLDER, TRNFMT(trainsize))
    rfolder = _makefolder(kfolder, rlabel)
    if np.isscalar(regs):
        regs = [regs]
    rregs = np.round(regs, 0)

    # Find (or create) ROM index JSON file.
    rom_json = os.path.join(kfolder, ROM_INDEX_FILE)
    if not os.path.isfile(rom_json):
        with open(rom_json, 'w') as outfile:
            json.dump({}, outfile)

    # Load and search ROM index file.
    with open(rom_json, 'r') as infile:
        rom_data = json.load(infile)
    if rlabel not in rom_data:
        rom_data[rlabel] = {}
    for filename, reglabel in rom_data[rlabel].items():
        if rregs.size == len(reglabel) and np.all(rregs == np.round(reglabel)):
            romfile = os.path.join(rfolder, filename)
            if not overwrite:
                return romfile
            else:
                rom_data[rlabel].pop(filename)
                if os.path.isfile(romfile):
                    os.remove(romfile)
                break

    # Add an entry to the index if the ROM was not found (or superseded).
    filename = time.strftime("%Y-%m-%d_%H:%M:%S") + ".h5"
    rom_data[rlabel][filename] = regs
    with open(rom_json, 'w') as outfile:
        json.dump(rom_data, outfile, indent=4)
    return os.path.join(rfolder, filename)


def statistical_features_path():
    """Return the path to the file containing statistical features computed
    from the lifted GEMS data over the spatial domain at each point in time.
    """
    return os.path.join(BASE_FOLDER, FEATURES_FILE)


def figures_path():
    """Return the path to the folder containing all results figures."""
    # return _makefolder(BASE_FOLDER, FIGURES_FOLDER)   # Figures live by data.
    return _makefolder(os.getcwd(), FIGURES_FOLDER)     # Figures live by code.


def tecplot_path():
    """Return the path to the folder containing files for use with Tecplot."""
    return _makefolder(BASE_FOLDER, TECPLOT_FOLDER)


def grid_data_path():
    """Return the path to the file containing the raw grid data."""
    return os.path.join(tecplot_path(), GRID_FILE)


def gems_snapshot_path(timeindex):
    """Return the path to the file containing a full-order GEMS snapshot
    at time index `timeindex`, for use with Tecplot.
    """
    folder = _makefolder(tecplot_path(), "gems")
    return os.path.join(folder, f"snapshot_{timeindex:05d}.dat")


def rom_snapshot_path(trainsize, num_modes, regs):
    """Return the path to the folder containing reconstructed snapshots
    derived from a ROM trained with `trainsize` snapshots, `num_modes` POD
    modes, and regularization parameters of `regs`, for use with Tecplot.
    """
    filename = f"{TRNFMT(trainsize)}_{DIMFMT(num_modes)}_{REGFMT(regs)}"
    return _makefolder(tecplot_path(), filename)


# Validation ------------------------------------------------------------------

# Check dictionary keys
for d,label in zip([VARTITLES, VARUNITS], ["VARTITLES", "VARUNITS"]):
    if sorted(d.keys()) != sorted(ROM_VARIABLES):
        raise KeyError(f"{label}.keys() != ROM_VARIABLES")

# Check that the base folder exists.
BASE_FOLDER = os.path.abspath(BASE_FOLDER)
if not os.path.exists(BASE_FOLDER):
    raise NotADirectoryError(BASE_FOLDER + " (set config.BASE_FOLDER)")


# Initialize default logger ---------------------------------------------------

_logger = logging.getLogger()
_handler = logging.FileHandler(log_path(), 'a')
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_handler.setLevel(logging.INFO)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
