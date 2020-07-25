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

## State Variables

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
| CH<sub>4</sub> mass fraction | _Y_<sub>_1_</sub> | `Y_CH4` | |
|  O<sub>2</sub> mass fraction | _Y_<sub>_2_</sub> |  `Y_O2` | |
| H<sub>2</sub>O mass fraction | _Y_<sub>_3_</sub> | `Y_H2O` | |
| CO<sub>2</sub> mass fraction | _Y_<sub>_4_</sub> | `Y_CO2` | |
| CH<sub>4</sub> molar concentration | _c_<sub>_1_</sub> | `c_CH4` | kmol/m<sup>3</sup> |
|  O<sub>2</sub> molar concentration | _c_<sub>_2_</sub> |  `c_O2` | kmol/m<sup>3</sup> |
| H<sub>2</sub>O molar concentration | _c_<sub>_3_</sub> | `c_H2O` | kmol/m<sup>3</sup> |
| CO<sub>2</sub> molar concentration | _c_<sub>_4_</sub> | `c_CO2` | kmol/m<sup>3</sup> |
| Total energy | _e_ | `e` | J/m<sup>3</sup> |

The variables are grouped three ways:
- The _conservation variables_ **q**<sub>c</sub> = \[_p_, _ρv_<sub>_x_</sub>, _ρv_<sub>_y_</sub>, _ρe_, _Yρ_<sub>_1_</sub>, _Yρ_<sub>_2_</sub>, _Yρ_<sub>_3_</sub>, _Yρ_<sub>_4_</sub> \]<sup>T</sup> are the independent variables of the governing conservation equations.
- The _GEMS variables_ **q**<sub>GEMS</sub> = \[_p_, _v_<sub>_x_</sub>, _v_<sub>_y_</sub>, _T_, _Y_<sub>_1_</sub>, _Y_<sub>_2_</sub>, _Y_<sub>_3_</sub>, _Y_<sub>_4_</sub> \]<sup>T</sup> are the solution variables of the finite volume CFD code that discretizes the conservation equations.
- The _learning variables_ **q**<sub>ROM</sub> = \[_p_, _v_<sub>_x_</sub>, _v_<sub>_y_</sub>, _T_, _ξ_, _c_<sub>_1_</sub>, _c_<sub>_2_</sub>, _c_<sub>_3_</sub>, _c_<sub>_4_</sub> \]<sup>T</sup> are the variables used by Operator Inference to learn a ROM for the combustor.

## Governing Dynamics

The dynamics of the combustor are governed by the conservation equations for mass, momentum, energy, and species mass fraction, _∂_**q**<sub>c</sub> / _∂t_ + ∇.(**K** - **K**<sub>_v_</sub>) = **S**, where
- **K** is the inviscid flux,
- **K**<sub>_v_</sub> is the viscous flux, and
- **S** is the source term defined by the combustion reaction.

The boundary conditions are as follows.
- Top and bottom wall: no-slip conditions.
- Upstream at the inlets: constant mass flow (fuel / oxidizer being injected).
- Downstream at the outlet: non-reflecting while maintaining chamber pressure (reactants being ejected).

It is shown in [\[2\]](#references) that by transforming from the conservative variables **q**<sub>c</sub> to **q**<sub>ROM</sub>, the governing equations become nearly quadratic.
The strategy we pursue is to obtain high-dimensional simulation data, transform it to the variables **q**<sub>ROM</sub>, and learn a quadratic ROM from the transformed data with Operator Inference.

## High-dimensional Data: GEMS

The General Equation and Mesh Solver (GEMS), a finite-volume CFD code developed at Purdue University, was used to simulate the combustor conservation equations.
The computational domain is discretized into _n_<sub>_x_</sub> = 38523 cells and the solution is computed for the 8 variables **q**<sub>GEMS</sub>, resulting in _n_ = 38523 x 8 = 308184 entries for a full solution at a single time.
Data is available for 30000 time steps with step size 10<sup>-7</sup>s, i.e., 3 ms of simulation time.
Placing these 30000 data snapshots columnwise into a single matrix yields the 308184x30000 snapshot matrix **X** that constitutes available training data for a ROM.
See [DOCUMENTATION.md](./DOCUMENTATION.md) for download and processing instructions.

## Quantities of Interest

To judge the effectiveness of a learned ROM for this problem, we compare simulation output from the ROM to the output of GEMS in three main ways.
- _Time traces_ of the learning variable at specified locations in the computational domain, e.g., the pressure at the corner of the domain above the injector as a function of time.
- _Statistical features_ of the flow, e.g., the mean temperature over the entire spatial domain at a fixed point in time.
- _Coherent structures_ of the flow in two dimensions.

See [REPORT.md](./REPORT.md) for plots and analysis.

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

