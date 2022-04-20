# step2c_project.py
"""Project the lifted, scaled snapshot training data to the low-dimensional
subspace spanned by the columns of the POD basis V, compute time derivative
information for the projected snapshots, and save the projected data.

Examples
--------
# Project 10,000 preprocessed snapshots.
$ python3 step2c_project.py 10000

# Project 20,000 preprocessed snapshots.
$ python3 step2c_project.py 20000

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes to use in the projection.
>>> Q_, Qdot_, t = utils.load_projected_data(trainsize, num_modes)

Command Line Arguments
----------------------
"""
import h5py
import logging
import numpy as np

import rom_operator_inference as opinf

import config
import utils


def project_and_save_data(Q, t, V):
    """Project preprocessed snapshots to a low-dimensional subspace.

    Parameters
    ----------
    Q : (NUM_ROMVARS*DOF,trainsize) ndarray
        Preprocessed snapshot data to be projected.
    t : (trainsize,) ndarray
        Time domain corresponding to the snapshots.
    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis of rank r.

    Returns
    -------
    Q_ : (r,trainsize) ndarray
        Projected snapshots.
    Qdot_ : (r,trainsize) ndarray
        Time derivatives of projected snapshots.
    """
    # Validate arguments.
    if Q.shape[1] != t.shape[0]:
        raise ValueError("training_data and time_domain not aligned")

    # Verify that the time domain is uniformly spaced with spacing config.DT.
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt):
        raise ValueError("t not uniformly spaced")
    if not np.isclose(dt, config.DT):
        raise ValueError("t spacing != config.DT")

    # Project the snapshot data.
    with utils.timed_block(f"Projecting snapshots to a {V.shape[1]:d}"
                           "-dimensional linear subspace"):
        Q_ = V.T @ Q

    # Compute time derivative data.
    with utils.timed_block("Approximating time derivatives "
                           "of projected snapshots"):
        Qdot_ = opinf.pre.xdot_uniform(Q_, dt, order=4)

    # Save the projected training data.
    save_path = config.projected_data_path(Q.shape[1])
    with utils.timed_block("Saving projected data"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", data=Q_)
            hf.create_dataset("ddt", data=Qdot_)
            hf.create_dataset("time", data=t)
    logging.info(f"Projected data saved to {save_path}.\n")

    return Q_, Qdot_


def main(trainsize):
    """Project lifted, scaled snapshot training data to the subspace spanned
    by the columns of the POD basis V; compute velocity information for the
    projected snapshots; and save the projected data.

    Parameters
    ----------
    trainsize : int
        The number of snapshots to use in the computation. There must
        exist a file of exactly `trainsize` lifted, scaled snapshots
        (see step2a_transform.py) and a basis for those snapshots
        (see step2b_basis.py).
    """
    utils.reset_logger(trainsize)

    # Load lifted, scaled snapshot data.
    scaled_data, time_domain, _, _ = utils.load_scaled_data(trainsize)

    # Load the POD basis.
    V, _, _ = utils.load_basis(trainsize, None)

    # Project and save the data.
    return project_and_save_data(scaled_data, time_domain, V)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE"""
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")

    # Do the main routine.
    args = parser.parse_args()
    main(args.trainsize)
