# Documentation

This document walks through the steps for downloading and processing the data from [\[3\]](#references), model learning with the `rom_operator_inference` Python package [\[4\]](#references), and postprocessing/visualizing results.

<!--

## File Summary

#### Utility Files

- [`config.py`](../config.py): Configuration file containing project directives for file and folder names, constants for the geometry and chemistry of the problem, plot customizations, and so forth.
- [`utils.py`](../utils.py): Utilities for logging and timing operations, loading and saving data, saving figures, and calculating ROM reconstruction errors.
- [`chemistry_conversions.py`](../chemistry_conversions.py): Chemistry-related data conversions.
- [`data_processing.py`](../data_processing.py): Tools for (un)lifting and (un)scaling data.

#### Main Files

- [`step1_unpack.py`](../step1_unpack.py): Read raw `.tar` archives and compile data into a single HDF5 file.
- [`step2_preprocess.py`](../step2_preprocess.py): Lift, scale, and project training data.
    - [`step2a_lift.py`](../step2a_lift.py): Lift and scale raw training data.
    - [`step2b_basis.py`](../step2b_basis.py): Compute reduced bases from the lifted, scaled training data.
    - [`step3c_project.py`](../step2c_project.py): Project lifted, scaled data onto low-dimensional subspaces.
- [`step3_train.py`](../step3_train.py): Learn ROMs from projected data via regularized Operator Inference.
- [`step4_analyze.py`](../step4_analyze.py): Use trained ROMs for prediction.
- [`step5_export.py`](../step5_export.py): Export data to a Tecplot-friendly format.

-->

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
