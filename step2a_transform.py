# step2a_transform.py
"""Transform the GEMS data to the learning variables and scale each variable
appropriately. Save the processed data.

Examples
--------
# Transform and save 10,000 snapshots.
$ python3 step2a_transform.py 10000

# Transform and save sets of 10,000, 20,000, and 30,000 snapshots.
$ python3 step2a_transform.py 10000 20000 30000

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> X, t, scales = utils.load_scaled_data(trainsize)

Command Line Arguments
----------------------
"""
import h5py
import logging
import numpy as np
import scipy.spatial as sp

import config
import utils
import data_processing as dproc


def load_and_lift_gems_data(trainsize):
    """Lift raw GEMS training snapshots (columnwise) to the learning variables.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to lift.

    Returns
    -------
    lifted_data : (NUM_ROMVARS*DOF,trainsize) ndarray
        The lifted snapshots.

    time_domain : (trainsize,) ndarray
        The time domain corresponding to the lifted snapshots.
    """
    # Load as many snapshots of GEMS training data as are needed.
    gems_data, time_domain = utils.load_gems_data(cols=trainsize)

    # Lift the training data to the learning variables.
    with utils.timed_block(f"Lifting {trainsize:d} GEMS snapshots"):
        lifted_data = dproc.lift(gems_data)

    return lifted_data, time_domain


def weights_time_decay(time_domain, sigma=2):
    """Construct weights based on the time (smaller time, greater weight):

    w_j = σ^(t_j / t_{k-1}),  j = 0, 1, ..., k - 1 = trainsize - 1.

    Parameters
    ----------
    time_domain : (trainsize,) ndarray
        Time domain corresponding to the training snapshots.

    sigma : float > 1
        Base of exponential.

    Returns
    -------
    w : (trainsize,) ndarray
        Snapshot weights.
    """
    t = time_domain - time_domain[0]
    return sigma**(-t/t[-1])


def weights_gaussian(training_data, sigma=1, k=None, kernelize=True):
    """Construct weights based on the Gaussian kernel (spatial importance):

    K(xi, xj) = exp(-||xi - xj||^2 / 2σ^2)

    Parameters
    ----------
    training_data : (n,k) ndarray
        Training snapshots, pre-processed except for mean shifting.

    sigma : float > 0
        Gaussian kernel spread hyperparameter.

    k : int > 0 or None
        Dimension of random projection to approximate distances.

    kernelize : bool
        If True, apply the Gaussian kernel. If False, use squared Euclidean
        distances (no kernel).
    """
    # If k is given, randomly project the data to r dimensions.
    if k is not None:
        M = np.random.standard_normal((training_data.shape[0], k))
        X = (M.T @ training_data).T
    else:
        X = training_data.T
        k = 1

    # Calculate the kernel matrix and the resulting weights.
    distances = sp.distance.pdist(X, "sqeuclidean") / k
    if kernelize:
        distances = np.exp(-distances/(2*sigma**2))
    K = sp.distance.squareform(distances)
    return np.mean(K, axis=1)


def scale_and_save_data(trainsize, lifted_data, time_domain, weights=None):
    """Scale lifted snapshots (by variable) and save the scaled snapshots.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to scale and save.

    lifted_data : (NUM_ROMVARS*DOF, k>trainsize) ndarray
        Lifted snapshots to scale and then save.

    time_domain : (k>trainsize,) ndarray
        The time domain corresponding to the lifted snapshots.

    weights : (trainsize,) ndarray or None
        If given, weight the mean shift and the resulting snapshots.

    Returns
    -------
    training_data : (NUM_ROMVARS*DOF, trainsize) ndarray
        Scaled, shifted snapshots to use as training data for the basis.

    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot of the scaled training data.

    scales : (NUM_ROMVARS,2) ndarray
        Info on how the snapshot data was scaled.
    """
    # Scale the learning variables to the bounds in config.SCALE_TO.
    with utils.timed_block(f"Scaling {trainsize:d} lifted snapshots"):
        training_data, scales = dproc.scale(lifted_data[:,:trainsize].copy())

    # Shift the scaled data by the mean snapshot.
    with utils.timed_block(f"Shifting {trainsize:d} scaled snapshots by mean"):
        if weights is None:
            qbar = np.mean(training_data, axis=1)           # Standard mean
            training_data -= qbar.reshape((-1,1))           # Shift by mean
        else:
            if isinstance(weights, str):
                if weights == "temporal":
                    weights = weights_time_decay(time_domain[:trainsize])
                elif weights == "Gaussian":
                    weights = weights_gaussian(training_data)
            weights /= np.sum(weights)                      # Normalize weights
            qbar = np.mean(training_data * weights, axis=1) # Weighted mean
            training_data -= qbar.reshape((-1,1))           # Shift by mean
            training_data *= weights                        # Weight snapshots

    # Save the lifted, scaled training data.
    save_path = config.scaled_data_path(trainsize)
    with utils.timed_block("Saving scaled, lifted training data"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", data=training_data)
            hf.create_dataset("time", data=time_domain[:trainsize])
            hf.create_dataset("mean", data=qbar)
            hf.create_dataset("scales", data=scales)
    logging.info(f"Processed data saved to {save_path}.\n")

    return training_data, qbar, scales


def main(trainsizes):
    """Lift and scale the GEMS simulation training data and save the results.

    Parameters
    ----------
    trainsizes : int or list(int)
        Number of snapshots to lift, scale, and save.
    """
    utils.reset_logger()

    if np.isscalar(trainsizes):
        trainsizes = [int(trainsizes)]

    # Lift the training data.
    lifted_data, time_domain = load_and_lift_gems_data(max(trainsizes))

    # Scale and save each subset of lifted data.
    for trainsize in trainsizes:
        utils.reset_logger(trainsize)
        scale_and_save_data(trainsize, lifted_data, time_domain)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE [...]"""
    parser.add_argument("trainsize", type=int, nargs='+',
                        help="number of snapshots to lift, scale, and save")

    # Do the main routine.
    args = parser.parse_args()
    main(args.trainsize)
