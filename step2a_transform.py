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


def scale_and_save_data(trainsize, lifted_data, time_domain):
    """Scale lifted snapshots (by variable) and save the scaled snapshots.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to scale and save.
    lifted_data : (NUM_ROMVARS*DOF, k>trainsize) ndarray
        Lifted snapshots to scale and then save.
    time_domain : (k>trainsize,) ndarray
        The time domain corresponding to the lifted snapshots.

    Returns
    -------
    training_data : (NUM_ROMVARS*DOF, trainsize) ndarray
        Scaled, shifted snapshots to use as training data for the basis.
    qbar : (NUM_ROMVARS*DOF,) ndarray
        Mean snapshot of the scaled training data.
    scales : (NUM_ROMVARS,2) ndarray
        Info on how the snapshot data was scaled.
    """
    # Shift the lifted data by the mean in the variables p, T, and xi.
    with utils.timed_block("Shifting snapshots by mean profile"):
        shifted_data = lifted_data[:,:trainsize].copy()
        qbar = np.zeros(shifted_data.shape[0])
        for var in ["p", "T", "xi"]:
            s = dproc._varslice(var, qbar.size)
            # qbar[s] = np.mean(shifted_data[s], axis=1)  # Shift by profile.
            qbar[s] = np.mean(shifted_data[s])            # Shift by scalar.
        shifted_data -= qbar.reshape((-1,1))

    # Scale the learning variables to [-1, 1] with MaxAbs scaling.
    with utils.timed_block(f"Scaling {trainsize:d} lifted, shifted snapshots"):
        training_data, scales = dproc.scale(shifted_data)

    # Save the lifted, shifted, scaled training data.
    save_path = config.scaled_data_path(trainsize)
    with utils.timed_block("Saving lifted, shifted, scaled training data"):
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
