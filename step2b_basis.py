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
import numpy as np

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc


def compute_and_save_pod_basis(num_modes, training_data, scales):
    """Compute and save the POD basis via a randomized SVD.

    Parameters
    ----------
    num_modes : int
        Number of POD modes to compute for each set of variables
        (everything-but-temperature and temperature-only).

    training_data : (NUM_ROMVARS*DOF,trainsize) ndarray
        Training snapshots to take the SVD of.

    scales : (NUM_ROMVARS,2) ndarray
        Info on how the snapshot data was scaled.

    Returns
    -------
    V : (NUM_ROMVARS*DOF,r) ndarray
        POD basis of rank r = 2*num_modes.
    """
    # Split the training data into T and non-T blocks.
    notT = np.row_stack([dproc.getvar(v, training_data)
                         for v in config.ROM_VARIABLES if v != "T"])
    T = dproc.getvar("T", training_data)

    # Compute the randomized SVD from the training data.
    with utils.timed_block(f"Computing {num_modes}-component rSVD (not T)"):
        V1, svdvals1 = roi.pre.pod_basis(notT, r=num_modes, mode="randomized",
                                         n_iter=15, random_state=42)

    with utils.timed_block(f"Computing {num_modes}-component rSVD (T)"):
        V2, svdvals2 = roi.pre.pod_basis(T, r=num_modes, mode="randomized",
                                         n_iter=15, random_state=42)
    # Save the POD basis.
    save_path = config.basis_path(training_data.shape[1])
    with utils.timed_block(f"Saving POD basis"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("notT/V", data=V1)
            hf.create_dataset("notT/svdvals", data=svdvals1)
            hf.create_dataset("T/V", data=V2)
            hf.create_dataset("T/svdvals", data=svdvals2)
            hf.create_dataset("scales", data=scales)
    logging.info(f"POD bases of rank {num_modes} saved to {save_path}.\n")

    return utils._assemble_basis(V1, V2)


def main(trainsize, num_modes):
    """Compute the POD basis (dominant left singular values) of a set of
    lifted, scaled snapshot training data and save the basis and the
    corresponding singular values.

    WARNING: This will OVERWRITE any existing basis for this `trainsize`.

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
    compute_and_save_pod_basis(num_modes, training_data, scales)


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
                        help="number of left singular vectors/values to save "
                             " for each subbasis")

    # Do the main routine.
    args = parser.parse_args()
    main(args.trainsize, args.modes)
