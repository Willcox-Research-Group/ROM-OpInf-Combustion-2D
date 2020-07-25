# Single-Injector 2D Combustor

This document summarizes the problem statement.
See [\[2\]](#references) for full details.

## Computational Domain

We consider a two-dimensional representation of a combustion chamber with a single fuel injector.
The geometry is pictured below.

<p align="center"><img src="https://raw.githubusercontent.com/Willcox-Research-Group/ROM-OpInf-Combustion-2D/master/docs/figures/domain.svg"></p>

Methane (CH<sub>4</sub>) is introduced upsteam (from the left) through the small injector and oxygen (O<sub>2</sub>) is introduced upstream through the large injector.
The chemicals react in a single-step combustion process, producing water (H<sub>2</sub>O) and carbon dioxide (CO<sub>2</sub>).
The stoichiometric equation is CH<sub>4</sub> + 2O<sub>2</sub> -> 2H<sub>2</sub>O + CO<sub>2</sub>.
Reactants flow downstream (to the right) and out, i.e., the right end of the domain is where the flames are ejected.

## Variables

In addition to the chemical species, the variables of greatest interest for engineering purposes are temperature and pressure.
The dynamics also depend on the density of the mixture, the velocity of the fluid flow, and the total energy.

| Variable Name | Symbol | Code | Units |
|          ---: | :---:  | :--- | :---  |
| Pressure | _p_ | `p` | Pa | 
| _x_-velocity | _v_<sub>_x_</sub> | `vx` | m/s |
| _y_-velocity | _v_<sub>_y_</sub> | `vy` | m/s |
| Temperature | _T_ | `T` | K |
| Density | _ρ_ | `rho` | kg/m<sup>3</sup> |
| Specific volume | _ξ_ | `xi` | m<sup>3</sup>/kg |
| CH<sub>4</sub> mass fraction | _c_<sub>_1_</sub> | `c_CH4` | |
|  O<sub>2</sub> mass fraction | _c_<sub>_2_</sub> |  `c_O2` | |
| H<sub>2</sub>O mass fraction | _c_<sub>_3_</sub> | `c_H2O` | |
| CO<sub>2</sub> mass fraction | _c_<sub>_4_</sub> | `c_CO2` | |
| CH<sub>4</sub> molar concentration | _Y_<sub>_1_</sub> | `Y_CH4` | kmol/m<sup>3</sup> |
|  O<sub>2</sub> molar concentration | _Y_<sub>_2_</sub> |  `Y_O2` | kmol/m<sup>3</sup> |
| H<sub>2</sub>O molar concentration | _Y_<sub>_3_</sub> | `Y_H2O` | kmol/m<sup>3</sup> |
| CO<sub>2</sub> molar concentration | _Y_<sub>_4_</sub> | `Y_CO2` | kmol/m<sup>3</sup> |
| Total energy | _e_ | `e` | J/m<sup>3</sup> |

## Governing Dynamics

The boundary conditions are as follows.
- Top and bottom wall: no-slip conditions.
- Upstream at the inlets: constant mass flow (fuel / oxidizer being injected).
- Downstream at the outlet: non-reflecting while maintaining chamber pressure (reactants being ejected).

## Quantities of Interest

---

# References

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

