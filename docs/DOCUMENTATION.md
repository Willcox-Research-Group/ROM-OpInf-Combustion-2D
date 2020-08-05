# Documentation

This document walks through the steps for downloading and processing the data from [\[3\]](#references), model learning with the `rom_operator_inference` Python package [\[4\]](#references), and postprocessing/visualizing results.

## Contents

- [**File Summary**](#file-summary): list of code files with short descriptions.
- [**Setup**](#setup): how to download the data and the code.
- [**Instructions**](#instructions): how to use the code.
    1. [**Unpack**](#1-unpack): extract GEMS data.
    2. [**Preprocess**](#2-preprocess): prepare a set of training data.
    3. [**Train**](#3-train): learn reduced-order models (ROMs) from training data.
    4. [**Analyze**](#4-analyze): reconstruct and compare results of trained ROMs.
    5. [**Export**](#5-export): write Tecplot-friendly files for full-domain visualization.
- [**References**](#references)

## File Summary

### Utility Files

- [`config.py`](../config.py): Configuration file containing project directives for file and folder names, constants for the geometry and chemistry of the problem, plot customizations, and so forth.
- [`utils.py`](../utils.py): Utilities for logging and timing operations, loading and saving data, saving figures, and calculating ROM reconstruction errors.
- [`chemistry_conversions.py`](../chemistry_conversions.py): Chemistry-related data conversions.
- [`data_processing.py`](../data_processing.py): Tools for (un)lifting and (un)scaling data.

### Main Files

- [`step1_unpack.py`](../step1_unpack.py): Read raw `.tar` archives and compile snapshot data into a single HDF5 data set.
- [`step2_preprocess.py`](../step2_preprocess.py): Lift, scale, and project GEMS training data. The process can also be decoupled into the following routines.
    - [`step2a_lift.py`](../step2a_lift.py): Lift and scale GEMS training data.
    - [`step2b_basis.py`](../step2b_basis.py): Compute reduced POD bases from lifted, scaled training data.
    - [`step3c_project.py`](../step2c_project.py): Project lifted, scaled data onto low-dimensional subspaces.
- [`step3_train.py`](../step3_train.py): Learn ROMs from projected data via regularized Operator Inference.

<!-- - [`step4_analyze.py`](../step4_analyze.py): Use trained ROMs for prediction.
- [`step5_export.py`](../step5_export.py): Export data to a Tecplot-friendly format. -->

<!-- ### Other Files

- `log.log`: Logging files (created as needed). For experiments with a specified number of training snapshots _k_, a separate log file is created (e.g., _k = 10000_ activity is logged to `config.BASE_FOLDER/k10000/log.log`.
- [`requirements.txt`](../requirements.txt): Python package requirements for this repository. Use `pip3 install --user -r requirements.txt` to install the prerequisites.
 -->

## Setup

This section walks through downloading the code and the associated data files.
As an example, suppose the (small) code files are to be placed in a folder `~/Desktop/combustion/` and the (large) data files are to be placed in a folder `/storage/combustion/`.
The raw data is about 60 GB, so about 200 GB of disk storage are needed to comfortably work with the data.

### Download the Code

The preferred way to use and contribute to this repository is to [create a fork](https://guides.github.com/activities/forking/) of the source repository [Willcox-Research-Group/ROM-OpInf-Combustion-2D](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D), then [create a local clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) of the new fork.
On the command line, run
```shell
$ git clone https://<username>@github.com/<username>/ROM-OpInf-Combustion-2D.git ~/Desktop/combustion
```
where `<username>` is your [GitHub](https://github.com) username.

To keep the repository up to date, set the source repository as an [upstream remote](https://github.com/git-guides/git-remote):
```shell
$ cd ~/Desktop/combustion
$ git remote add upstream https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D.git
$ git config --local pull.rebase false
```

Then the following commands update the repository.
```shell
$ cd ~/Desktop/combustion       # Navigate to the folder.
$ git checkout master           # Switch to the main branch.
$ git pull upstream master      # Pull updates from the source repository.
```

### Download GEMS Data from Globus

A dataset [\[3\]](#references) of 30,000 high-fidelity solution snapshots produced by the General Equation and Mesh Solver (GEMS) is publically available and hosted by [Globus](https://www.globus.org/).
See [this page](https://deepblue.lib.umich.edu/data/user-guide#download-globus) and the [Globus docs](https://docs.globus.org/how-to/) for login instructions.
After logging in and locating the collection, the machine that will host the local copy of the data must be [set up as a globus endpoint](https://docs.globus.org/how-to/globus-connect-personal-linux).
The following shell commands summarize the process.
```shell
# Create the data folder if needed.
$ mkdir -p /storage/combustion

# Download the latest version of the Globus personal client (e.g., for a Linux machine).
$ wget https://downloads.globus.org/globus-connect-personal/linux/stable/globusconnectpersonal-latest.tgz
$ tar -vzxf globusconnectpersonal-latest.tgz     # Unpack the archive.
$ cd globusconnectpersonal-3.1.1                 # Enter the created folder.

# Set up the endpoint. This requires an auth code from Globus online.
$ ./globusconnectpersonal -setup --no-gui

# Start the endpoint server.
$ nohup ./globusconnectpersonal -start -restrict-paths rw/storage/combustion &
```
At this point, select files from the collection on Globus to transfer to the endpoint.
When the download process is finished, shut down the endpoint server with the following command.
```
$ ./globusconnectpersonal -stop
```

### Code Configuration

In [`config.py`](../config.py), set `BASE_FOLDER` to the directory where the processed data should be placed, preferably as an absolute path.
This can be the folder containing the raw `.tar` data files or another folder.
```python
# /Desktop/combustion/config.py

BASE_FOLDER = "/storage/combustion"
```
This folder must exist in order for the code to run.

Other variables in [`config.py`](../config.py) can be changed to customized


## Instructions

This section walks through the process of creating and analyzing reduced-order models (ROMs) for the GEMS combustion data.
We assume the code is placed in `~/Desktop/combustion`, the data is in `/storage/combustion`, and `BASE_FOLDER` in [`config.py`](../config.py) is set to `/storage/combustion`.

### 1. Unpack

The script [`step1_unpack.py`](../step1_unpack.py) reads the GEMS output directly from the `.tar` archives downloaded from Globus, gathers the data into a single data set, and saves it in [HDF5](https://www.h5py.org) format.
The process runs in parallel (via the [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) module) and takes several minutes.
After the process completes successfully, the `.tar` archives from Globus may be deleted.

```shell
# See usage details.
$ python3 step1_unpack.py --help

# Read the GEMS .tar files.
$ python3 step1_unpack.py /storage/combustion
```

The snapshot data can then be accessed with `utils.load_raw_data()` or the following code.

```python
>>> import h5py
>>> import config

>>> with h5py.File(config.gems_data_path(), 'r') as hf:
...     gems_data = hf["data"][:,:]     # The GEMS snapshots.
...     time_domain = hf["time"][:]     # The associated time domain.
```
Note carefully the `[:]` and `[:,:]`, which are required to load the actual data from the file.

Each column of `gems_data` is a snapshot, i.e., `gems_data[:,j]` is **q**<sub>g</sub>(_t_<sub>_j_</sub>).
The first `DOF = 38523` rows of data represent the first native variable, and so on.
The variables are, in order,
1. Pressure \[Pa\]
2. _x_-velocity \[m/s\]
3. _y_-velocity \[m/s\]
4. Temperature \[K\]
5. CH<sub>4</sub> (methane) Mass Fraction
6. O<sub>2</sub> (oxygen) Mass Fraction
7. H<sub>2</sub>O (water) Mass Fraction
8. CO<sub>2</sub> (carbon dioxide) Mass Fraction

See [PROBLEM.md](./PROBLEM.md) for more on the problem statement and the GEMS output.

### 2. Preprocess

The data must be preprocessed to make it more suitable for Operator Inference.
The script [`step2_preprocess.py`](../step2_preprocess.py) generates training data for reduced-order model learning in three steps:
1. Transform the GEMS variables **q**<sub>g</sub> to the learning variables **q**<sub>L</sub> and scale each
variable to the intervals defined by `config.SCALE_TO`.
2. Compute the POD basis (the dominant left singular vectors) of the lifted, scaled snapshot training data and save the basis and the corresponding singular values.
3. Project the lifted, scaled snapshot training data to the subspace spanned by
the columns of the POD basis V, compute velocity information for the projected
snapshots, and save the projected data.

These three steps can also performed separately by [step2a_lift.py](../step2a_lift.py), [step2b_basis.py](../step2b_basis.py), and [step2c_project.py](../step2c_project.py), respectively.

```shell
# See usage details.
$ python3 step2_preprocess.py --help

# Generate a training data from the first 10,000 GEMS snapshots with 44 POD modes.
$ python3 step2_preprocess.py 10000 44

# Equivalently, do the three steps separately.
$ python3 step2a_lift.py 10000          # Lift and scale 10,000 GEMS snapshots.
$ python3 step2b_basis.py 10000 44      # Compute the rank-44 POD basis.
$ python3 step2c_project.py 10000 44    # Project the lifted, scaled snapshots.
```

To access the resulting POD basis and the associated singular values, use `utils.load_basis()` or the following code.
```python
>>> import h5py
>>> import config

>>> with h5py.File(config.basis_path(), 'r') as hf:
...     V = hf["V"][:]              # The POD basis (left singular vectors).
...     svdvals = hf["svdvals"][:]  # The associated singular values.
```
To access the resulting projected data, use `utils.load_projected_data()` or the following.
```python
>>> import h5py
>>> import config

>>> with h5py.File(config.projected_data_path(), 'r') as hf:
...     X_ = hf["data"][:]          # The projected snapshots.
...     Xdot_ = hf["xdot"][:]       # The associated projected velocities.
...     times = hf["time"][:]       # The time domain for the snapshots.
...     scales = hf["scales"][:]    # Info on how the data was scaled.
```

### 3. Train

### 4. Analyze

### 5. Export

---

## References

- \[1\] [McQuarrie, S.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), _Data-driven reduced-order models via regularized operator inference for a single-injector combustion process_, to appear.

- \[2\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [Learning physics-based reduced-order models for a single-injector combustion process](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, Vol. 58:6, pp. 2658-2672, 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13.
([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2020ROMCombustion,
    title     = {Learning physics-based reduced-order models for a single-injector combustion process},
    author    = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal   = {AIAA Journal},
    volume    = {58},
    number    = {6},
    pages     = {2658--2672},
    year      = {2020},
    publisher = {American Institute of Aeronautics and Astronautics}
}</pre></details>

- \[3\] [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ) (2020). 2D Benchmark Reacting Flow Dataset for Reduced Order Modeling Exploration \[Data set\]. University of Michigan - Deep Blue. [https://doi.org/10.7302/jrdr-bj37](https://doi.org/10.7302/jrdr-bj37).

- \[4\] [ROM Operator Inference Python 3 package](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) ([pypi](https://pypi.org/project/rom-operator-inference/))
