# test_data_processing.py
"""Run basic tests for data_processing.py."""
import logging
import numpy as np

import config
import utils
import data_processing as dproc


def test_lift(testsize):
    """Read `testsize` random snapshots of GEMS data, lift them, and check
    that the before-and-after variables are consistent with each other.
    """
    # Load the unlifted, unscaled snapshot data.
    testindices = np.random.choice(30000, size=testsize, replace=False)
    testindices.sort()
    gems_data, t = utils.load_gems_data(cols=testindices)

    # Lift the training data to the learning variables.
    with utils.timed_block("Lifting training data to new coordinates"):
        lifted_data = dproc.lift(gems_data)

    # Check that the first three variables are the same.
    with utils.timed_block("Verifying first four variables"):
        for i in range(4):
            s = slice(i*config.DOF, (i+1)*config.DOF)
            assert np.allclose(lifted_data[s], gems_data[s])

    # Verify inverse lifting.
    with utils.timed_block("Verifying inverse lifting"):
        unlifted_data = dproc.unlift(lifted_data)
        assert np.allclose(unlifted_data, gems_data)

    return lifted_data


def test_getvar(lifted_data):
    """Test data_processing.getvar()."""
    with utils.timed_block("Verifying variable extraction"):
        for i,v in enumerate(config.ROM_VARIABLES):
            s = slice(i*config.DOF, (i+1)*config.DOF)
            assert np.all(dproc.getvar(v, lifted_data) == lifted_data[s])


def test_scalers(lifted_data):
    """Test data_processing.scale() and data_processing.unscale(),
    including checking that they are inverses.
    """
    # Shift the test data by the mean profile in a few variables.
    with utils.timed_block("Shifting snapshots by mean profile"):
        shifted_data = lifted_data
        qbar = np.zeros(shifted_data.shape[0])
        for var in ["p", "T", "xi"]:
            s = dproc._varslice(var, qbar.size)
            qbar[s] = np.mean(shifted_data[s], axis=1)
        shifted_data -= qbar.reshape((-1,1))

    # Scale the test data (learning the scaling simultaneously).
    with utils.timed_block("Scaling lifted test data"):
        scaled_data, scales = dproc.scale(shifted_data.copy())

    # Verify the scales and that the shift worked for each variable.
    with utils.timed_block("Verifying shift results with scales"):
        for i,v in enumerate(config.ROM_VARIABLES):
            s = slice(i*config.DOF, (i+1)*config.DOF)
            if v in ["p", "T", "xi"]:
                assert np.isclose(np.mean(shifted_data[s]), 0)
                assert np.isclose(np.mean(scaled_data[s]), 0)
            assert np.isclose(np.abs(scaled_data[s]).max(), 1)
            if v in config.SPECIES:
                assert np.isclose(scaled_data[s].min(), 0)

    # Redo the shift with the given scales and compare the results.
    with utils.timed_block("Verifying repeat shift with given scales"):
        scaled_data2, _ = dproc.scale(shifted_data.copy(), scales)
        assert np.allclose(scaled_data2, scaled_data)

    # Undo the shift and compare the results.
    with utils.timed_block("Verifying inverse scaling"):
        unscaled_data = dproc.unscale(scaled_data, scales)
        assert np.allclose(unscaled_data, scaled_data)

    # Check the inverse property for a subset of the variables.
    with utils.timed_block("Repeating experiment with nontrivial varindices"):
        variables = np.random.choice(config.ROM_VARIABLES,
                                     size=4, replace=False)
        subset = np.vstack([dproc.getvar(v, shifted_data) for v in variables])
        shifted_subset, _ = dproc.scale(subset.copy(), scales, variables)
        unshifted_subset = dproc.unscale(shifted_subset, scales, variables)
        assert np.allclose(unshifted_subset, subset)


def main(testsize):
    """Run all tests with `testsize` columns of data."""
    utils.reset_logger()
    lifted_data = test_lift(testsize)
    test_getvar(lifted_data)
    test_scalers(lifted_data)
    logging.info("ALL TESTS PASSED")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.usage = """ python3 test_data_processing.py -h
        python3 test_data_processing.py TESTSIZE"""
    parser.add_argument("testsize", type=int, nargs='?', default=500,
                        help="Number of test snapshots to use (default 500)")

    # Do the main routine.
    args = parser.parse_args()
    main(args.testsize)
