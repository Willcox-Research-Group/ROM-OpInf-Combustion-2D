# test_basis_projection.py

import numpy as np

import config
import utils


def test_load_basis(trainsize, r1, r2):
    V, scales = utils.load_basis(trainsize, r1, r2)
    assert V.shape == (config.DOF * config.NUM_ROMVARS, r1 + r2)
    assert scales.shape == (config.NUM_ROMVARS, 2)

    # Validate basis sparsity pattern.
    n1 = config.DOF * config.ROM_VARIABLES.index("T")
    n2 = n1 + config.DOF
    assert np.count_nonzero(V[:n1,r1:]) == 0
    assert np.count_nonzero(V[n1:n2,:r1]) == 0
    assert np.count_nonzero(V[n2:,r1:]) == 0


def test_load_projected_data(trainsize, r1, r2):

    Q, t, scales = utils.load_scaled_data(trainsize)
    V, scales2 = utils.load_basis(trainsize, r1, r2)
    assert scales.shape == scales2.shape == (config.NUM_ROMVARS, 2)
    assert np.all(scales == scales2)

    # Do the projection the old-fashioned way.
    Qa_ = V.T @ Q
    assert Qa_.shape == (r1 + r2, t.size)

    Qb_, _, _ = utils.load_projected_data(trainsize, r1, r2)
    assert Qb_.shape == (r1 + r2, t.size)
    assert np.allclose(Qa_, Qb_)


def main():
    test_load_basis(5000, 20, 30)
    test_load_projected_data(5000, 25, 25)
    test_load_projected_data(5000, 15, 30)
    test_load_projected_data(5000, 10, 5)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
