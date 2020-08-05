# Documentation

This document walks through the steps for downloading and processing the data from [\[3\]](#references), model learning with the `rom_operator_inference` Python package [\[4\]](#references), and postprocessing/visualizing results.

## Contents

- [**File Summary**](#file-summary): list of code files with short descriptions.
- [**Setup**](#setup): how to download the data and the code.
- [**Instructions**](#instructions): how to use the code.
    1. [**Unpack**](#unpack): extract GEMS data.
    2. [**Preprocess**](#preprocess): prepare a set of training data.
    3. [**Train**](#train): learn reduced-order models (ROMs) from training data.
    4. [**Analyze**](#analyze): reconstruct and compare results of trained ROMs.
    5. [**Export**](#export): write Tecplot-friendly files for full-domain visualization.
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
    - [`step2a_lift.py`](../step2a_lift.py): Lift and scale raw GEMS training data.
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

### Download GEMS Data from Globus

A dataset of 60,000 high-fidelity solution snapshots produced by the General Equation and Mesh Solver (GEMS) is available at **TODO**.

### Download the Code

The preferred way to use and contribute to this repository is to [create a fork](https://guides.github.com/activities/forking/) of the source repository [Willcox-Research-Group/ROM-OpInf-Combustion-2D](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D), then [create a local clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) of your new fork with, for example,
```shell
$ git clone https://<username>@github.com/<username>/ROM-OpInf-Combustion-2D.git
```
where `<username>` is your [GitHub](https://github.com) username.

### Code Configuration

In `config.py`, set `BASE_FOLDER` to the directory where the data should be dumped, preferably as an absolute path (e.g., `/storage/GEMS/`).
```python
BASE_FOLDER = "/storage/GEMS"               # Base folder for all data.
```
This folder [must exist](https://en.wikipedia.org/wiki/Mkdir) in order for the code to run.

## Instructions

In this example we assume that `BASE_FOLDER` in `config.py` is set to `/storage/combustion2D`.

### 1. Unpack

The GEMS data files should already be placed in their own dedicated folder (see [**Setup**](#steup)).
In the command prompt, run `python3 step1_unpack.py <path/to/data>`, where `</path/to/data>` is the folder containing the raw data, preferably as an absolute path.
This script reads the files and compiles the data into a single [HDF5](https://www.h5py.org) file.
The process runs in parallel (via [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html)) and takes several minutes.

```shell
$ python3 step1_unpack.py /storage/GEMS/rawdata
```

After running `step1_unpack.py`, the shapshot matrix is contained in the HDF5 file `data_raw.h5`.
The raw data can be accessed with `utils.load_raw_data()` or the following code.

```python
>>> import h5py
>>> with h5py.File("/storage/GEMS/data_raw.h5", 'r') as hf:
...     raw_data = hf["data"][:]       # The raw, unscaled snapshots.
...     time_domain = hf["time"][:]    # The associated time domain.
```

Each column of `raw_data` is a snapshot, i.e., `raw_data[:,j]` is **q**<sub>g</sub>(_t_<sub>_j_</sub>).
The first `DOF = 38523` rows of data represent the first native variable, and so on.
The native variables in the GEMS data are, in order,
1. Pressure [Pa]
2. _x_-velocity [m/s]
3. _y_-velocity [m/s]
4. Temperature [K]
5. CH<sub>4</sub> (methane) Mass Fraction
6. O<sub>2</sub> (oxygen) Mass Fraction
7. H<sub>2</sub>O (water) Mass Fraction.
8. CO<sub>2</sub> (carbon dioxide) Mass Fraction

See [PROBLEM.md](./PROBLEM.md) for more on the problem setup.

### 2. Preprocess

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
