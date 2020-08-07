# step2_preprocess.py
"""Generate training data for reduced-order model learning in three steps:

1. Transform the GEMS training data to the learning variables and scale each
variable to the intervals defined by config.SCALE_TO. Save the processed data
(see also step2a_transform.py).

2. Compute the POD basis (the dominant left singular vectors) of the lifted,
scaled snapshot training data and save the basis and the corresponding singular
values (see also step2b_basis.py).

3. Project the lifted, scaled snapshot training data to the subspace spanned by
the columns of the POD basis V, compute velocity information for the projected
snapshots, and save the projected data (see also step2c_project.py).

These three steps can be performed separately with
* step2a_transform.py,
* step2b_basis.py, and
* step2c_project.py, respectively.

Examples
--------
# Get training data from 10,000 snapshots and project it with 24 POD modes.
$ python3 step2_preprocess.py 10000 24

# Get training data from 15,000 snapshots and project it with r POD modes
# for every integer r from 25 through 50 (inclusive).
$ python3 step2_preprocess.py 15000 25 50 --moderange

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> X, t, scales = utils.load_scaled_data(trainsize)
>>> V, svdvals = utils.load_basis(trainsize, num_modes)
>>> X_, Xdot_, t, scales = utils.load_projected_data(trainsize, num_modes)

Command Line Arguments
----------------------
"""
import os
import h5py
import logging
import numpy as np

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc
import step2a_transform as step2a
import step2b_basis as step2b
import step2c_project as step2c


def main(trainsize, num_modes):
    """Lift and scale the GEMS simulation data; compute a POD basis of the
    lifted, scaled snapshot training data; project the lifted, scaled snapshot
    training data to the subspace spanned by the columns of the POD basis V,
    and compute velocity information for the projected snapshots.

    Save lifted/scaled snapshots, the POD basis, and the projected data.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to lift / scale / save.

    num_modes : int or list(int)
        The number of POD modes (left singular vectors) to use in the
        projection, which determines the dimension of the resulting ROM.
    """
    utils.reset_logger(trainsize)

    if np.isscalar(num_modes):
        num_modes = [int(num_modes)]

    # STEP 2A: Lift and scale the data ----------------------------------------
    try:
        # Attempt to load existing lifted, scaled data.
        X, time_domain, scales = utils.load_scaled_data(trainsize)

    except utils.DataNotFoundError:
        # Lift the GEMS data, then scale the lifted snapshots by variable.
        lifted_data, time_domain = step2a.load_and_lift_gems_data(trainsize)
        X, scales = step2a.scale_and_save_data(trainsize,
                                               lifted_data, time_domain)

    # STEP 2B: Get the POD basis from the lifted, scaled data -----------------
    try:
        # Attempt to load existing SVD data.
        V, _ = utils.load_basis(trainsize, max(num_modes))

    except utils.DataNotFoundError:
        # Compute and save the (randomized) SVD from the training data.
        V, _ = step2b.compute_and_save_pod_basis(trainsize,
                                                 max(num_modes), X, scales)

    # STEP 2C: Project data to the appropriate subspace -----------------------
    for r in num_modes:
        step2c.project_and_save_data(trainsize, r, X, time_domain, scales, V)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE MODES [...] [--moderange]"""
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")
    parser.add_argument("modes", type=int, nargs='+',
                        help="number of POD modes for projecting data")
    parser.add_argument("--moderange", action="store_true",
                        help="if two modes given, treat them as min, max"
                             " and project for each integer in [min, max]")

    # Do the main routine.
    args = parser.parse_args()
    if args.moderange and len(args.modes) == 2:
        args.modes = list(range(args.modes[0], args.modes[1]+1))
    main(args.trainsize, args.modes)
