# step2_preprocess.py
"""Generate training data for reduced-order model learning in three steps:

1. Transform the GEMS training data to the learning variables and scale each
variable appropriately. Save the processed data (see also step2a_transform.py).

2. Compute the POD basis (the dominant left singular vectors) of the lifted,
scaled snapshot training data and save the basis (see also step2b_basis.py).

3. Project the lifted, scaled snapshot training data to the subspace spanned by
the columns of the POD basis V, compute time derivative information for the
projected snapshots, and save the projected data (see also step2c_project.py).

These three steps can be performed separately with
* step2a_transform.py,
* step2b_basis.py, and
* step2c_project.py, respectively.

Examples
--------
# Get training data from 10,000 snapshots and with a maximum of 50 POD modes.
$ python3 step2_preprocess.py 10000 50

# Get training data from 15,000 snapshots and with a maximum of 100 POD modes.
$ python3 step2_preprocess.py 15000 100

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> Q, t, scales = utils.load_scaled_data(trainsize)
>>> V, scales = utils.load_basis(trainsize, num_modes)
>>> Q_, Qdot_, t = utils.load_projected_data(trainsize, num_modes)

Command Line Arguments
----------------------
"""
import utils
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
    num_modes : int or None
        The number of POD modes (left singular vectors) to use in the
        projection. This is the upper bound for the size of ROMs that
        can be trained with this data set.
    """
    utils.reset_logger(trainsize)

    # STEP 2A: Lift and scale the data ----------------------------------------
    try:
        # Attempt to load existing lifted, scaled data.
        training_data, time, qbar, scales = utils.load_scaled_data(trainsize)

    except utils.DataNotFoundError:
        # Lift the GEMS data, then scale the lifted snapshots by variable.
        lifted_data, time = step2a.load_and_lift_gems_data(trainsize)
        training_data, qbar, scales = step2a.scale_and_save_data(trainsize,
                                                                 lifted_data,
                                                                 time)
        del lifted_data

    # STEP 2B: Get the POD basis from the lifted, scaled data -----------------
    try:
        # Attempt to load existing SVD data.
        basis, qbar, scales = utils.load_basis(trainsize, None)
        if basis.shape[1] < num_modes:
            raise utils.DataNotFoundError("not enough saved basis vectors")
        num_modes = basis.shape[1]      # Use larger basis size if available.

    except utils.DataNotFoundError:
        # Compute and save the (randomized) SVD from the training data.
        basis = step2b.compute_and_save_pod_basis(num_modes,
                                                  training_data, qbar, scales)

    # STEP 2C: Project data to the appropriate subspace -----------------------
    return step2c.project_and_save_data(training_data, time, basis)


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
                        help="number of POD modes for projecting data")

    # Do the main routine.
    args = parser.parse_args()
    main(args.trainsize, args.modes)
