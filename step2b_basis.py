# step2b_basis.py
"""Compute the POD basis (the dominant left singular vectors) of the lifted,
scaled snapshot training data and save the basis and the corresponding
singular values.

Examples
--------
# Use 10,000 snapshots to compute a rank-50 POD basis.
$ python3 step2b_basis.py 10000 50

# Use 20,000 snapshots to compute rank-50 and rank-100 POD bases.
$ python3 step2b_basis.py 20000 50 100

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> V, svdvals = utils.load_basis(trainsize, num_modes)

Command Line Arguments
----------------------
"""
import h5py
import logging
import numpy as np

import rom_operator_inference as roi

import config
import utils


def compute_and_save_pod_basis(trainsize, num_modes, training_data, scales):
    """Compute and save the POD basis via a randomized SVD.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use in computing the basis.

    num_modes : list(int) or int
        Number of POD modes to compute.

    training_data : (NUM_ROMVARS*DOF,trainsize) ndarray
        Training snapshots to take the SVD of.

    scales : (NUM_ROMVARS,2) ndarray
        Info on how the snapshot data was scaled.

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis of rank r = max(num_modes).

    svdvals : (r,) ndarray
        Singular values corresponding to the POD modes.
    """
    if trainsize != training_data.shape[1]:
        raise ValueError("trainsize and training_data not aligned")

    if np.isscalar(num_modes):
        num_modes = [int(num_modes)]

    # Compute the randomized SVD from the training data.
    rmax = max(num_modes)
    with utils.timed_block(f"Computing {rmax}-component randomized SVD"):
        V, svdvals = roi.pre.pod_basis(training_data,
                                       r=rmax,
                                       mode="randomized",
                                       n_iter=15,
                                       random_state=42)
    # Save the POD basis.
    for r in num_modes:
        save_path = config.basis_path(trainsize, r)
        with utils.timed_block(f"Saving POD basis of rank {r}"):
            with h5py.File(save_path, 'w') as hf:
                hf.create_dataset("V", data=V[:,:r])
                hf.create_dataset("svdvals", data=svdvals[:r])
        logging.info(f"POD basis of rank {r} saved to {save_path}.\n")

    return V, svdvals


def main(trainsize, num_modes):
    """Compute the POD basis (dominant left singular values) of a set of
    lifted, scaled snapshot training data and save the basis and the
    corresponding singular values.

    Parameters
    ----------
    trainsize : int
        The number of snapshots to use in the computation. There must exist
        a file of exactly `trainsize` lifted, scaled snapshots
        (see step2a_lift.py).

    num_modes : int or list(int)
        The number of POD modes (left singular vectors) to retain.
    """
    utils.reset_logger(trainsize)

    # Load the first `trainsize` lifted, scaled snapshot data.
    training_data, _, scales = utils.load_scaled_data(trainsize)

    # Compute and save the (randomized) SVD from the training data.
    compute_and_save_pod_basis(trainsize, num_modes, training_data, scales)


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
                        help="number of left singular vectors/values to save")

    # Do the main routine.
    args = parser.parse_args()
    main(args.trainsize, args.modes)
