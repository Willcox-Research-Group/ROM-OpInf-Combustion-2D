# config.py
"""Configuration file containing project directives for file and folder names,
constants for the geometry and chemistry of the problem, plot customizations,
and so forth.

New users should set the BASE_FOLDER variable to the location of the data,
preferably as an absolute path. Other global variables specify the naming
conventions for the various data files.
"""
import os
import re
import glob
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
PROJECTED_DATA_FILE = "data_projected.h5"   # Name of projected data files.
FEATURES_FILE = "statistical_features.h5"   # Name of statistical feature file.
GRID_FILE = "grid.dat"                      # Name of Tecplot grid data file.
LOG_FILE = "log.log"                        # Name of log files.

TRN_PREFIX = "k"                            # Prefix for training size folders.
DIM_PREFIX = "r"                            # Prefix for ROM dimension folders.
POD_PREFIX = "basis"                        # Prefix for POD basis files.
ROM_PREFIX = "rom"                          # Prefix for trained ROM files.
REG_PREFIX = "reg"                          # Prefix for regularization values.

TRNFMT = lambda k: f"{TRN_PREFIX}{k:05d}"   # String format for training sizes.
DIMFMT = lambda r: f"{DIM_PREFIX}{r:03d}"   # String format for ROM dimensions.
REGFMT = lambda λ: f"{REG_PREFIX}{λ:09.0f}" # String format for regularization.

# Domain geometry -------------------------------------------------------------

DOF = 38523                                 # Spatial degrees of freedom.

MONITOR_LOCATIONS = [36915,                 # Index of monitor location 1.
                     37886,                 # Index of monitor location 2.
                     36443,                 # Index of monitor location 3.
                      6141]                 # Index of monitor location 4.

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

# Chemistry constants ---------------------------------------------------------

MOLAR_MASSES = [16.04,                      # Molar mass of CH4 [kg/kmol].
                32.0,                       # Molar mass of O2  [kg/kmol].
                18.0,                       # Molar mass of H2O [kg/kmol].
                44.01]                      # Molar mass of CO2 [kg/kmol].

R_UNIVERSAL = 8.3144598                     # Univ. gas constant [J/(mol K)].

# Scaling information ---------------------------------------------------------

# Scale the learning variables to the following bounds.
SCALE_TO = np.array([[-1, 1],               # Pressure.
                     [-1, 1],               # x-velocity.
                     [-1, 1],               # y-velocity.
                     [-1, 1],               # Temperature.
                     [-1, 1],               # Specific volume.
                     [ 0, 1],               # CH4 molar concentration.
                     [ 0, 1],               #  O2 molar concentration.
                     [ 0, 1],               # H2O molar concentration.
                     [ 0, 1]],              # CO2 molar concentration.
                    dtype=np.float)

# ROM Structure ---------------------------------------------------------------

MODELFORM = "cAHB"                          # ROM operators to be inferred.

# Input function (Pressure oscillation) ---------------------------------------

U = lambda t: 1e6*(1 + 0.1*np.sin(np.pi*10000*t))

# Matplotlib plot customization --------------------------------------------------

plt.rc("figure", dpi=1200)                  # High-quality figures.
plt.rc("text", usetex=True)                 # Use LaTeX fonts.
plt.rc("font", family="serif")              # Crisp axis labels.
plt.rc("legend", edgecolor='none',          # No borders around legends.
                 frameon=False)             # No legend backgrounds.

# Names of the learning variables (for LaTeX fonts).
VARTITLES = {  "p": "Pressure",
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
VARUNITS = {   "p": "Pa",
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


def basis_path(trainsize, num_modes):
    """Return the path to the file containing the POD basis
    computed with `trainsize` snapshots and `num_modes` modes.
    """
    return os.path.join(BASE_FOLDER, TRNFMT(trainsize),
                        f"{POD_PREFIX}_{DIMFMT(num_modes)}.h5")


def smallest_basis_path(trainsize, num_modes):
    """Return the path to the file containing the *smallest* POD basis
    computed with `trainsize` snapshots and at least `num_modes` modes.
    """
    folder = os.path.join(BASE_FOLDER, TRNFMT(trainsize))
    files = glob.glob(os.path.join(folder, f"{POD_PREFIX}_*.h5"))
    if not files:
        raise FileNotFoundError(f"could not locate POD file with {trainsize} "
                                f"training snapshots in {folder}")
    pat = fr"{POD_PREFIX}_{DIM_PREFIX}(\d+)\.h5"
    rs = [int(re.findall(pat, s)[0]) for s in files]
    rs = [r for r in rs if r >= num_modes]
    if not rs:
        raise FileNotFoundError(f"could not locate POD file with {trainsize} "
                                f"training snapshots and at least {num_modes} "
                                f"retained POD modes in {folder}")
    return basis_path(trainsize, min(rs))


def projected_data_path(trainsize, num_modes):
    """Return the path to the file containing `trainsize` training snapshots,
    projected to a `num_modes`-dimensional space.
    """
    folder = _makefolder(BASE_FOLDER,
                         TRNFMT(trainsize), DIMFMT(num_modes))
    return os.path.join(folder, PROJECTED_DATA_FILE)


def rom_path(trainsize, num_modes, reg):
    """Return the path to the file containing a ROM trained from
    `trainsize` snapshots, projected to a `num_modes`-dimensional space,
    with regularization factor `reg`.
    """
    folder = os.path.join(BASE_FOLDER,
                          TRNFMT(trainsize), DIMFMT(num_modes))
    return os.path.join(folder, f"{ROM_PREFIX}_{REGFMT(reg)}.h5")


def statistical_features_path():
    """Return the path to the file containing statistical features computed
    from the lifted GEMS data over the spatial domain at each point in time.
    """
    return os.path.join(BASE_FOLDER, FEATURES_FILE)


def figures_path():
    """Return the path to the folder containing all results figures."""
    # return _makefolder(BASE_FOLDER, FIGURES_FOLDER)   # Put figures by data.
    return _makefolder(os.getcwd(), FIGURES_FOLDER)     # Put figures by code.


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


def rom_snapshot_path(trainsize, num_modes, reg):
    """Return the path to the folder containing reconstructed snapshots
    derived from a ROM trained with `trainsize` snapshots, `num_modes` POD
    modes, and a regularization factor of `reg`, for use with Tecplot.
    """
    return _makefolder(tecplot_path(),
                      f"{TRNFMT(trainsize)}_{DIMFMT(num_modes)}_{REGFMT(reg)}")


# Validation ------------------------------------------------------------------

if SCALE_TO.shape[0] != NUM_ROMVARS:
    raise ValueError(f"SCALE_TO must have NUM_ROMVARS={NUM_ROMVARS} rows")

# Check dictionary keys
for d,label in zip([VARTITLES, VARUNITS], ["VARTITLES", "VARUNITS"]):
    if sorted(d.keys()) != sorted(ROM_VARIABLES):
        raise KeyError(f"{label}.keys() != ROM_VARIABLES")

# Check that the base folder exists.
if not os.path.exists(BASE_FOLDER):
    raise NotADirectoryError(BASE_FOLDER + " (set config.BASE_FOLDER)")


# Initialize default logger ---------------------------------------------------

_logger = logging.getLogger()
_handler = logging.FileHandler(log_path(), 'a')
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_handler.setLevel(logging.INFO)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
