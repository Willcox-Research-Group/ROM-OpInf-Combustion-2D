[![License](https://img.shields.io/github/license/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](./LICENSE)
[![Top language](https://img.shields.io/github/languages/top/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/Willcox-Research-Group/ROM-OpInf-Combustion-2D)
[![Latest commit](https://img.shields.io/github/last-commit/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/commits/master)
[![Documentation](https://img.shields.io/badge/Documentation-WIKI-important)](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/wiki)
<!-- [![Issues](https://img.shields.io/github/issues/Willcox-Research-Group/ROM-OpInf-Combustion-2D)](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/issues) -->

# Reduced-order Modeling via Operator Inference for 2D Combustion

This repository is an extensive example of the non-intrusive, data-driven Operator Inference procedure for reduced-order modeling applied to a two-dimensional combustion problem.
It is the source code for [\[1\]](#references) (see the [jrsnz2021](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/tree/jrsnz2021) branch) and can be used to reproduce the results of [\[2\]](#references).

[**See the Wiki for details on the problem statement, instructions for using this repository, and visual results.**](https://github.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/wiki)

<p align="center">
    <img src="https://drive.google.com/uc?export=view&id=1TbIfBQW-YYXVydBFC0McJaSFjkvaeLvA" width="700">
</p>

---

**Contributors**:
[Shane A. McQuarrie](https://github.com/shanemcq18),
[Renee Swischuk](https://github.com/swischuk),
Parikshit Jain,
[Boris Kramer](http://kramer.ucsd.edu/),
[Karen Willcox](https://kiwi.oden.utexas.edu/)

## References

- \[1\] [McQuarrie, S. A.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [**Data-driven reduced-order models via regularized operator inference for a single-injector combustion process**](https://arxiv.org/abs/2008.02862).
arXiv:2008.02862.
([Download](https://arxiv.org/pdf/2008.02862.pdf))<details><summary>BibTeX</summary><pre>
@article{MHW2021regOpInfCombustion,
    title   = {Data-driven reduced-order models via regularized operator inference for a single-injector combustion process},
    author  = {McQuarrie, S. A. and Huang, C. and Willcox, K.},
    journal = {arXiv preprint arXiv:2008.02862},
    year    = {2021}
}</pre></details>

- \[2\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [**Learning physics-based reduced-order models for a single-injector combustion process**](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, Vol. 58:6, pp. 2658-2672, 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13.
([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2020romCombustion,
    title     = {Learning physics-based reduced-order models for a single-injector combustion process},
    author    = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal   = {AIAA Journal},
    volume    = {58},
    number    = {6},
    pages     = {2658--2672},
    year      = {2020},
    publisher = {American Institute of Aeronautics and Astronautics}
}</pre></details>

- \[3\] [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ) (2020). [[Updated] **2D Benchmark Reacting Flow Dataset for Reduced Order Modeling Exploration \[Data set\]**](https://doi.org/10.7302/nj7w-j319). University of Michigan - Deep Blue. https://doi.org/10.7302/nj7w-j319.

- \[4\] [**ROM Operator Inference Python 3 package**](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) ([pypi](https://pypi.org/project/rom-operator-inference/))
