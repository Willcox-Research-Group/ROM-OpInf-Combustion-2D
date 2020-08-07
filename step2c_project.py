# step2c_project.py
"""Project lifted, scaled snapshot training data to the subspace
spanned by the columns of the POD basis V; compute velocity
information for the projected snapshots; and save the projected data.

Examples
--------
# Project 10,000 preprocessed snapshots to a 24-dimensional subspace.
$ python3 step2c_project.py 10000 24

# Project 20,000 preprocessed snapshots to 24- and 29-dimensional subspaces.
$ python3 step2c_project.py 20000 24 29

# Project 30,000 preprocessed snapshots to an r-dimensional subspace
# for each integer r from 17 through 30 (inclusive).
$ python3 step2c_project.py 30000 17 30 --moderange

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> X_, Xdot_, t, scales = utils.load_projected_data(trainsize, num_modes)

Command Line Arguments
----------------------
"""
import h5py
import logging
import numpy as np

import rom_operator_inference as roi

import config
import utils


def project_and_save_data(trainsize, r, X, time_domain, scales, V):
    """Project preprocessed snapshots to a low-dimensional subspace.

    Parameters
    ----------
    trainsize : int
        Number of training snapshots to project.

    r : int
        Number of POD modes to use in the projection.

    X : (NUM_ROMVARS*DOF,trainsize) ndarray
        Preprocessed snapshot data to be projected.

    time_domain : (trainsize,) ndarray
        Time domain corresponding to the snapshots.

    scales : (NUM_ROMVARS,2) ndarray
        Info on how the snapshot data was scaled.

    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis of rank at least r.

    Returns
    -------
    X_ : (r,trainsize) ndarray
        Projected snapshots.

    Xdot_ : (r,trainsize) ndarray
        Time derivatives of projected snapshots.
    """
    # Verify that the time domain is uniformly spaced with spacing config.DT.
    dt = time_domain[1] - time_domain[0]
    if not np.allclose(np.diff(time_domain), dt):
        raise ValueError("time domain not uniformly spaced")
    if not np.isclose(dt, config.DT):
        raise ValueError("time domain spacing != config.DT")

    # Project the snapshot data.
    with utils.timed_block("Projecting snapshots to a "
                           f"{r:d}-dimensional linear subspace"):
        X_ = V[:,:r].T @ X

    # Compute time derivative data.
    with utils.timed_block("Approximating time derivatives "
                           "of projected snapshots"):
        Xdot_ = roi.pre.xdot_uniform(X_, dt, order=4)

    # Save the projected training data.
    save_path = config.projected_data_path(trainsize, r)
    with utils.timed_block(f"Saving projected data"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", data=X_)
            hf.create_dataset("xdot", data=Xdot_)
            hf.create_dataset("time", data=time_domain)
            hf.create_dataset("scales", data=scales)
    logging.info(f"Projected data saved to {save_path}.\n")

    return X_, Xdot_


def main(trainsize, num_modes):
    """Project lifted, scaled snapshot training data to the subspace spanned
    by the columns of the POD basis V; compute velocity information for the
    projected snapshots; and save the projected data.

    Parameters
    ----------
    trainsize : int
        The number of snapshots to use in the computation. There must exist
        a file of exactly `trainsize` lifted, scaled snapshots
        (see step2a_lift.py).

    num_modes : int or list(int)
        The number of POD modes (left singular vectors) to use in the
        projection, which determines the dimension of the resulting ROM.
        There must exist a file of at least `num_modes` left singular vectors
        computed from exactly `trainsize` lifted, scaled snapshots
        (see step2b_basis.py).
    """
    utils.reset_logger(trainsize)

    if np.isscalar(num_modes):
        num_modes = [int(num_modes)]

    # Load lifted, scaled snapshot data.
    X, time_domain, scales = utils.load_scaled_data(trainsize)

    # Load the POD basis.
    V, _ = utils.load_basis(trainsize, max(num_modes))

    # Project and save the data for each number of POD modes.
    for r in num_modes:
        project_and_save_data(trainsize, r, X, time_domain, scales, V)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE MODES [...]"""
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
