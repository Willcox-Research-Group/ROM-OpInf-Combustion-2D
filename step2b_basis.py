# step2b_basis.py
"""Compute the POD basis (the dominant left singular vectors) of the lifted,
scaled snapshot training data. Save the basis and information on how the
data was scaled.

Examples
--------
# Use 10,000 snapshots to compute a rank-50 POD basis.
$ python3 step2b_basis.py 10000 50

# Use 20,000 snapshots to compute a rank-100 POD basis.
$ python3 step2b_basis.py 20000 100

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> V, scales = utils.load_basis(trainsize, num_modes)

Command Line Arguments
----------------------
"""
import h5py
import logging
import scipy.linalg as la

import rom_operator_inference as opinf

import config
import utils


def compute_and_save_pod_basis(num_modes, training_data, qbar, scales):
    """Compute and save the POD basis via a randomized SVD.

    Parameters
    ----------
    num_modes : int
        Number of POD modes to compute.
    training_data : (NUM_ROMVARS*DOF,trainsize) ndarray
        Training snapshots to take the SVD of.
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot of the scaled training data.
    scales : (NUM_ROMVARS,) ndarray
        Info on how the snapshot data was scaled.

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis of rank r.
    """
    # Compute the randomized SVD from the training data.
    with utils.timed_block(f"Computing {num_modes}-component randomized SVD"):
        V, svdvals = opinf.pre.pod_basis(training_data, r=num_modes,
                                         mode="randomized",
                                         n_iter=15, random_state=42)

    # Save the POD basis.
    save_path = config.basis_path(training_data.shape[1])
    with utils.timed_block("Saving POD basis"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("basis", data=V)
            hf.create_dataset("svdvals", data=svdvals)
            hf.create_dataset("mean", data=qbar)
            hf.create_dataset("scales", data=scales)
    logging.info(f"POD bases of rank {num_modes} saved to {save_path}.\n")

    return V


def compute_and_save_all_svdvals(training_data):
    """Compute and save the singular values corresponding to the *full* POD
    basis for the training data.

    Parameters
    ----------
    training_data : (NUM_ROMVARS*DOF,trainsize) ndarray
        Training snapshots to take the SVD of.

    Returns
    -------
    svdvals : (trainsize,) ndarray
        Singular values for the full POD basis.
    """
    # Compute the DENSE SVD of the training data to get the singular values.
    with utils.timed_block("Computing *dense* SVD for singular values"):
        svdvals = la.svdvals(training_data,
                             overwrite_a=True, check_finite=False)

    # Save the POD basis.
    save_path = config.basis_path(training_data.shape[1])
    save_path = save_path.replace(config.BASIS_FILE, "svdvals.h5")
    with utils.timed_block("Saving singular values"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("svdvals", data=svdvals)
    logging.info(f"Singular values saved to {save_path}.\n")

    return svdvals


def main(trainsize, num_modes):
    """Compute the POD basis (dominant left singular values) of a set of
    lifted, scaled snapshot training data and save the basis and the
    corresponding singular values.

    WARNING: This will OVERWRITE any existing basis for this `trainsize`.

    Parameters
    ----------
    trainsize : int
        The number of snapshots to use in the computation. There must exist
        a file of exactly `trainsize` lifted, shifted, scaled snapshots
        (see step2a_transform.py).
    num_modes : int or list(int)
        The number of POD modes (left singular vectors) to retain.
    """
    utils.reset_logger(trainsize)

    # Load the first `trainsize` lifted, scaled snapshot data.
    training_data, _, qbar, scales = utils.load_scaled_data(trainsize)

    if num_modes == -1:
        # Secret mode! Compute all singular values (EXPENSIVE).
        return compute_and_save_all_svdvals(training_data)

    # Compute and save the (randomized) SVD from the training data.
    return compute_and_save_pod_basis(num_modes, training_data, qbar, scales)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE MODES"""
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")
    parser.add_argument("modes", type=int,
                        help="number of left singular vectors/values to save")

    # Do the main routine.
    args = parser.parse_args()
    main(args.trainsize, args.modes)
