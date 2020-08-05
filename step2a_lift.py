# step2a_lift.py
"""Transform the GEMS data to the learning variables and scale each variable
to the intervals defined by config.SCALE_TO. Save the processed data.

To access the resulting processed data, use utils.load_scaled_data()
or the following code.

>>> import h5py
>>> with h5py.File(<scaled_data_path>, 'r') as hf:
...     scaled_data = hf["data"][:] # The lifted, scaled snapshots.
...     times = hf["time"][:]       # The associated time domain.
...     scales = hf["scales"][:]    # Info on how the data is scaled.

The <scaled_data_path> can be obtained via config.scaled_data_path().

Examples
--------
# Transform and save 10,000 snapshots.
$ python3 step2a_lift.py 10000

# Transform and save sets of 10,000, 15,000, and 20,000 snapshots.
$ python3 step2a_lift.py 10000 15000 20000
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
    """
    # Scale the learning variables to the bounds in config.SCALE_TO.
    with utils.timed_block(f"Shifting {trainsize:d} lifted snapshots "
                           f"(by variable) to bounds in config.SCALE_TO"):
        scaled_data, scales = dproc.scale(lifted_data[:,:trainsize].copy())

    # Save the lifted, scaled training data.
    save_path = config.scaled_data_path(trainsize)
    with utils.timed_block("Saving scaled, lifted training data"):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("data", data=scaled_data)
            hf.create_dataset("time", data=time_domain[:trainsize])
            hf.create_dataset("scales", data=scales)
    logging.info(f"Scaled data saved as {save_path}.")

    return scaled_data, scales


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
