# step3_train.py
"""Use projected data to learn reduced-order models via Operator Inference
with optimal regularization parameter selection.

Examples
--------
## --single: train and save a ROM with given regularization parameters.

# Use 10,000 projected snapshots to learn a ROM of dimension r = 46
# with regularization parameters λ1 = 400, λ2 = 21000.
$ python3 step3_train.py --single 10000 --modes 24 --regularization 400 21000

## --gridsearch: train over a grid of reglarization parameter candidates,
                 saving only the best resulting ROM.

# Use 20,000 projected snapshots to learn a ROM of dimension 92 and save the
# one with the regularization resulting in the least training error and for
# which the integrated POD modes stay within 150% of the training data in
# magnitude for 60,000 time steps. For the regularization parameters, test
# each point in a 4x5 grid of [500, 9000] x [8000, 10000]
$ python3 step3_train.py --gridsearch 10000 --modes 92
                         --regularization 5e2 9e3 4 8e3 1e4 5
                         --testsize 60000 --margin 1.5

## --minimize: given initial guesses for the regularization parameters,
               use a Nelder-Mead search to train and save a ROM that is
               locally optimal in the regularization parameter space.

# Use 30,000 projected snapshots to learn a ROM of dimension 141 and save the
# one with the regularization resulting in the least training error and for
# which the integrated POD modes stay within 150% of the training data in
# magnitude for 60,000 time steps. For the regularization parameters, search
# starting from λ1 = 300, λ2 = 7000.
$ python3 step3_train.py --minimize 10000 --modes 141
                         --regularization 300 7000
                         --testsize 60000 --margin 1.5

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> regs = 1e4, 1e5         # Regularization parameters for Operator Inference.
>>> rom = utils.load_rom(trainsize, num_modes, reg)

Command Line Arguments
----------------------
"""
import h5py
import logging
import itertools
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils


_MAXFUN = 100               # Artificial ceiling for optimization routine.


# Subroutines =================================================================

def check_lstsq_size(trainsize, r):
    """Report the number of unknowns in the Operator Inference problem,
    compared to the number of snapshots. Ask user for confirmation before
    attempting to solve an underdetermined problem.
    """
    # Print info on the size of the system to be solved.
    d = roi.lstsq.lstsq_size(config.MODELFORM, r, m=1)
    message = f"{trainsize} snapshots, {r}x{d} DOFs ({r*d} total)"
    print(message)
    logging.info(message)

    # If the system is underdetermined, ask for confirmation before proceeding.
    if d > trainsize:
        message = "LSTSQ SYSTEM UNDERDETERMINED"
        logging.warning(message)
        if input(f"{message}! CONTINUE? [y/n] ") != "y":
            raise ValueError(message)
    return d


def check_regs(regs):
    """Assure that there are two positive regularization parameters."""
    if np.isscalar(regs) or len(regs) != 2:
        raise ValueError("two regularization parmameters required")
    if any(λ < 0 for λ in regs):
        raise ValueError("regularization parameters must be positive")
    return regs


def regularizer(r, d, λ1, λ2):
    """Return the regularizer that penalizes all operator elements by λ1,
    except for the quadratic operator elements, which are penalized by λ2.

    Parameters
    ----------
    r : int
        Dimension of the ROM.

    d : int
        Number of unknowns in a single least-squares problem, i.e., the
        number of elements in a single row of the operator matrix O.

    λ1 : float
        Regularization parameter for the non-quadratic operators.

    λ2 : float
        Regularization parameter for the quadratic operator.

    Returns
    -------
    diag(𝚪) : (d,) ndarray
        Diagonal entries of the dxd regularizer 𝚪.
    """
    diag𝚪 = np.full(d, λ1)
    diag𝚪[1+r:-1] = λ2
    return diag𝚪


def is_bounded(q_rom, B, message="bound exceeded"):
    """Return True if the absolute integrated POD coefficients lie within the
    given bound.

    Parameters
    ----------
    q_rom : (r,len(time_domain)) ndarray
        Integrated POD modes, i.e., the direct result of integrating a ROM.

    B : float > 0
        The bound that the integrated POD coefficients must satisfy.
    """
    if np.abs(q_rom).max() > B:
        print(message+"...", end='')
        logging.info(message)
        return False
    return True


def save_trained_rom(trainsize, r, regs, rom):
    """Save the trained ROM with the specified attributes.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r : int
        Dimension of the ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.

    regs : two positive floats
        Regularization parameters (non-quadratic, quadratic) used in the
        Operator Inference least-squares problem for training the ROM.

    rom : rom_operator_inference.InferredContinuousROM
        Actual trained ROM object. Must have a `save_model()` method.
    """
    save_path = config.rom_path(trainsize, r, regs)
    rom.save_model(save_path, save_basis=False, overwrite=True)
    logging.info(f"ROM saved to {save_path}")


# Main routines ===============================================================

def train_single(trainsize, r, regs):
    """Train and save ROMs with the given dimension and regularization.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM.

    r : int
        Dimension of the desired ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.

    regs : two positive floats
        Regularization parameters (non-quadratic, quadratic) to use in the
        Operator Inference least-squares problem for training the ROM.
    """
    utils.reset_logger(trainsize)

    # Validate inputs.
    d = check_lstsq_size(trainsize, r)
    λ1, λ2 = check_regs(regs)

    # Load training data.
    Q_, Qdot_, t = utils.load_projected_data(trainsize, r)
    U = config.U(t)

    # Train and save the ROM.
    with utils.timed_block(f"Training ROM with k={trainsize:d}, "
                           f"r={r:d}, λ1={λ1:.0f}, λ2={λ2:.0f}"):
        rom = roi.InferredContinuousROM(config.MODELFORM)
        rom.fit(None, Q_, Qdot_, U, P=regularizer(r, d, λ1, λ2),
                compute_extras=False, check_regularizer=False)
        save_trained_rom(trainsize, r, regs, rom)


def train_gridsearch(trainsize, r, regs, testsize=None, margin=1.5):
    """Train ROMs with the given dimension over a grid of potential
    regularization parameters, saving only the ROM with the least training
    error that satisfies a bound on the integrated POD coefficients.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM.

    r : int
        Dimension of the desired ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.

    regs : (float, float, int, float, float, int)
        Bounds and sizes for the grid of regularization parameters.
        Linear:    search in [regs[0], regs[1]] at regs[2] points.
        Quadratic: search in [regs[3], regs[4]] at regs[5] points.

    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.

    margin : float >= 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    d = check_lstsq_size(trainsize, r)
    if len(regs) != 6:
        raise ValueError("len(regs) != 6 (bounds / sizes for parameter grid")
    check_regs(regs[0:2])
    check_regs(regs[3:5])
    λ1grid = np.logspace(np.log10(regs[0]), np.log10(regs[1]), int(regs[2]))
    λ2grid = np.logspace(np.log10(regs[3]), np.log10(regs[4]), int(regs[5]))

    # Load training data.
    t = utils.load_time_domain(testsize)
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r)
    U = config.U(t[:trainsize])

    # Compute the bound to require for integrated POD modes.
    M = margin * np.abs(Q_).max()

    # Create a solver mapping regularization parameters to operators.
    print(f"TRAINING {λ1grid.size*λ2grid.size} ROMS")
    with utils.timed_block(f"Constructing least-squares solver, r={r:d}"):
        rom = roi.InferredContinuousROM(config.MODELFORM)
        solver = rom._construct_solver(None, Q_, Qdot_, U, np.ones(d),
                                       compute_extras=False,
                                       check_regularizer=False)

    # Test each regularization parameter.
    errors_pass = {}
    errors_fail = {}
    for λ1,λ2 in itertools.product(λ1grid, λ2grid):
        with utils.timed_block(f"Testing ROM with λ1={λ1:5e}, λ2={λ2:5e}"):
            # Train the ROM on all training snapshots.
            rom._evaluate_solver(solver, regularizer(r, d, λ1, λ2))

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            errors = errors_pass if is_bounded(q_rom, M) else errors_fail

            # Calculate integrated relative errors in the reduced space.
            if q_rom.shape[1] > trainsize:
                errors[(λ1,λ2)] = roi.post.Lp_error(Q_, q_rom[:,:trainsize],
                                                            t[:trainsize])[1]

    # Choose and save the ROM with the least error.
    if not errors_pass:
        message = f"NO STABLE ROMS for r={r:d}"
        print(message)
        logging.info(message)
        return

    err2reg = {err:reg for reg,err in errors_pass.items()}
    λ1,λ2 = err2reg[min(err2reg.keys())]
    with utils.timed_block(f"Best regularization for k={trainsize:d}, "
                           f"r={r:d}: λ1={λ1:.0f}, λ2={λ2:.0f}"):
        rom._evaluate_solver(solver, regularizer(r, d, λ1, λ2))
        save_trained_rom(trainsize, r, (λ1,λ2), rom)


def train_minimize(trainsize, r, regs, testsize=None, margin=1.5):
    """Train ROMs with the given dimension(s), saving only the ROM with
    the least training error that satisfies a bound on the integrated POD
    coefficients, using a search algorithm to choose the regularization
    parameter.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM.

    r : int
        Dimension of the desired ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.

    regs0 : two positive floats
        Initial guesses for the regularization parameters (non-quadratic,
        quadratic) to use in the Operator Inference least-squares problem
        for training the ROM.

    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.

    margin : float >= 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    d = check_lstsq_size(trainsize, r)
    log10regs = np.log10(check_regs(regs))

    # Load training data.
    t = utils.load_time_domain(testsize)
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r)
    U = config.U(t[:trainsize])

    # Compute the bound to require for integrated POD modes.
    B = margin * np.abs(Q_).max()

    # Create a solver mapping regularization parameters to operators.
    with utils.timed_block(f"Constructing least-squares solver, r={r:d}"):
        rom = roi.InferredContinuousROM(config.MODELFORM)
        solver = rom._construct_solver(None, Q_, Qdot_, U, np.ones(d),
                                       compute_extras=False,
                                       check_regularizer=False)

    # Test each regularization parameter.
    def training_error(log10regs):
        """Return the training error resulting from the regularization
        parameters λ1 = 10^log10regs[0], λ1 = 10^log10regs[1]. If the
        resulting model violates the POD bound, return "infinity".
        """
        λ1, λ2 = 10**log10regs

        # Train the ROM on all training snapshots.
        with utils.timed_block(f"Testing ROM with λ1={λ1:e}, λ2={λ2:e}"):
            rom._evaluate_solver(solver, regularizer(r, d, λ1, λ2))

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            if not is_bounded(q_rom, B):
                return _MAXFUN

            # Calculate integrated relative errors in the reduced space.
            return roi.post.Lp_error(Q_, q_rom[:,:trainsize], t[:trainsize])[1]

    opt_result = opt.minimize(training_error, log10regs, method="Nelder-Mead")
    if opt_result.success and opt_result.fun != _MAXFUN:
        λ1, λ2 = 10**opt_result.x
        with utils.timed_block(f"Best regularization for k={trainsize:d}, "
                               f"r={r:d}: λ1={λ1:.0f}, λ2={λ2:.0f}"):
            rom._evaluate_solver(solver, regularizer(r, d, λ1, λ2))
            save_trained_rom(trainsize, r, (λ1,λ2), rom)
    else:
        message = "Regularization search optimization FAILED"
        print(message)
        logging.info(message)


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} --single TRAINSIZE R REG1 REG2
        python3 {__file__} --gridsearch TRAINSIZE R REG1 ... REG6
                               --testsize TESTSIZE --margin TAU
        python3 {__file__} --minimize TRAINSIZE R REG1 REG2
                               --testsize TESTSIZE --margin TAU"""
    # Parser subcommands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", action="store_true",
                       help="train and save a single ROM")
    group.add_argument("--gridsearch", action="store_true",
                       help="train over a grid of regularization parameter "
                            "candidates, saving only the best resulting ROM")
    group.add_argument("--minimize", action="store_true",
                       help="given initial guesses for the regularization "
                            "parameters, use a Nelder-Mead search to train "
                            "and save a ROM that is locally optimal in the "
                            "regularization parameter space")

    # Positional arguments.
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")
    parser.add_argument("modes", type=int,
                        help="number of POD modes used to project the data")
    parser.add_argument("regularization", type=float, nargs='+',
                        help="regularization parameters for ROM training")

    # Other keyword arguments.
    parser.add_argument("--testsize", type=int, default=None,
                        help="number of time steps for which the trained ROM "
                             "must satisfy the POD bound")
    parser.add_argument("--margin", type=float, default=1.5,
                        help="percent that the POD coefficients of the "
                             "ROM simulation are allowed to deviate in "
                             "magnitude from the training data")

    # Parse arguments and do one of the main routines.
    args = parser.parse_args()
    if args.single:
        train_single(args.trainsize, args.modes, args.regularization)
    elif args.gridsearch:
        train_gridsearch(args.trainsize, args.modes, args.regularization,
                         args.testsize, args.margin)
    elif args.minimize:
        train_minimize(args.trainsize, args.modes, args.regularization,
                       args.testsize, args.margin)
