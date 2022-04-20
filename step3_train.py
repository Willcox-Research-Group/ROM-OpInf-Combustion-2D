# step3_train.py
"""Use projected data to learn reduced-order models via Tikhonov-regularized
Operator Inference with regularization hyperparameter selection.

Examples
--------
## --single: train and save a single ROM for a given λ1, λ2.

# Use 10,000 projected snapshots to learn a ROM of dimension r = 24
# with regularization hyperparameters λ1 = 400, λ2 = 21000.
$ python3 step3_train.py --single 10000 24 400 21000

## --gridsearch: train over a grid of candidates for λ1 and λ2, saving
                 only the stable ROM with least training error.

# Use 20,000 projected snapshots to learn a ROM of dimension r = 40 and save
# the one with the regularization resulting in the least training error and
# for which the integrated POD modes stay within 150% of the training data in
# magnitude for 60,000 time steps. For the regularization hyperparameters, test
# each point in the 4x5 logarithmically-spaced grid [500,9000]x[8000,10000]
$ python3 step3_train.py --gridsearch 10000 40 5e2 9e3 4 8e3 1e4 5
                         --testsize 60000 --margin 1.5

## --minimize: given initial guesses for λ1 and λ2, use Nelder-Mead search
               to train and save a ROM that is locally optimal in the
               regularization hyperparameter space.

# Use 10,000 projected snapshots to learn a ROM of dimension r = 30 and save
# the one with the regularization resulting in the least training error and
# for which the integrated POD modes stay within 150% of the training data in
# magnitude for 60,000 time steps. For the regularization hyperparameters,
# search starting from λ1 = 300, λ2 = 7000.
$ python3 step3_train.py --minimize 10000 30 300 7000
                         --testsize 60000 --margin 1.5

Indicating 3 regularization hyperparameters instead of 2 results in training a
cubic model.

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> regs = 1e4, 1e5         # OpInf regularization hyperparameters.
>>> rom = utils.load_rom(trainsize, num_modes, reg)

Command Line Arguments
----------------------
"""
import logging
import itertools
import numpy as np
import scipy.optimize as opt

import rom_operator_inference as opinf

import config
import utils


_MAXFUN = 100               # Artificial ceiling for optimization routine.


# Subroutines =================================================================

def get_modelform(regs):
    """Return the rom_operator_inference ROM modelform that is appropriate for
    the number of regularization parameters (fully quadratic or fully cubic).

    Parameters
    ----------
    regs : two or three non-negative floats
        Regularization hyperparameters for Operator Inference.

    Returns
    -------
    modelform : str
        'cAHB' for fully quadratic ROM; 'cAHGB' for fully cubic ROM.
    """
    if np.isscalar(regs) or len(regs) == 2:
        return "cAHB"
    elif len(regs) == 3:
        return "cAHGB"
    raise ValueError("expected 2 or 3 regularization hyperparameters")


def check_lstsq_size(trainsize, r, modelform="cAHB"):
    """Report the number of unknowns in the Operator Inference problem,
    compared to the number of snapshots. Ask user for confirmation before
    attempting to solve an underdetermined problem.
    """
    # Print info on the size of the system to be solved.
    d = opinf.lstsq.lstsq_size(modelform, r, m=1)
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
    """Assure there are the correct number of non-negative regularization
    hyperparameters.

    Parameters
    ----------
    regs : list/ndarray of two or three non-negative floats
        Regularization hyperparameters.
    """
    if np.isscalar(regs):
        regs = [regs]

    # Check number of values.
    nregs = len(regs)
    if nregs not in (2,3):
        raise ValueError(f"expected 2 or 3 hyperparameters, got {nregs}")

    # Check non-negativity.
    if any(λ < 0 for λ in regs):
        raise ValueError("regularization hyperparameters must be non-negative")

    return regs


def regularizer(r, λ1, λ2, λ3=None):
    """Return the regularizer that penalizes all operator elements by λ1,
    except for the quadratic operator elements, which are penalized by λ2.
    If λ3 is given, the entries of the cubic operator are penalized by λ3.

    Parameters
    ----------
    r : int
        Dimension of the ROM.
    λ1 : float
        Regularization hyperparameter for the non-quadratic operators.
    λ2 : float
        Regularization hyperparameter for the quadratic operator.
    λ2 : float or None
        Regularization hyperparameter for the cubic operator (if present).

    Returns
    -------
    diag(𝚪) : (d,) ndarray
        Diagonal entries of the dxd regularizer 𝚪.
    """
    r1 = 1 + r
    r2 = r1 + r*(r + 1)//2
    if λ3 is None:
        diag𝚪 = np.full(r2+1, λ1)
        diag𝚪[r1:-1] = λ2
    else:
        r3 = r2 + r*(r + 1)*(r + 2)//6
        diag𝚪 = np.full(r3+1, λ1)
        diag𝚪[r1:r2] = λ2
        diag𝚪[r2:-1] = λ3
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
    regs : two or three non-negative floats
        regularization hyperparameters (first-order, quadratic, cubic) used
        in the Operator Inference least-squares problem for training the ROM.
    rom : rom_operator_inference.InferredContinuousROM
        Actual trained ROM object. Must have a `save_model()` method.
    """
    save_path = config.rom_path(trainsize, r, regs)
    rom.save_model(save_path, save_basis=False, overwrite=True)
    logging.info(f"ROM saved to {save_path}")


# Main routines ===============================================================

def train_single(trainsize, r, regs):
    """Train and save a ROM with the given dimension and regularization
    hyperparameters.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM.
    r : int
        Dimension of the desired ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.
    regs : two or three non-negative floats
        Regularization hyperparameters (first-order, quadratic, cubic) to use
        in the Operator Inference least-squares problem for training the ROM.
    """
    utils.reset_logger(trainsize)

    # Validate inputs.
    modelform = get_modelform(regs)
    check_lstsq_size(trainsize, r, modelform)
    check_regs(regs)

    # Load training data.
    Q_, Qdot_, t = utils.load_projected_data(trainsize, r)
    U = config.U(t)

    # Train and save the ROM.
    with utils.timed_block(f"Training ROM with k={trainsize:d}, "
                           f"{config.REGSTR(regs)}"):
        rom = opinf.InferredContinuousROM(modelform)
        rom.fit(None, Q_, Qdot_, U, P=regularizer(r, *list(regs)))
        save_trained_rom(trainsize, r, regs, rom)


def train_gridsearch(trainsize, r, regs, testsize=None, margin=1.1):
    """Train ROMs with the given dimension over a grid of potential
    regularization hyperparameters, saving only the ROM with the least
    training error that satisfies a bound on the integrated POD coefficients.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM.
    r : int
        Dimension of the desired ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.
    regs : (float, float, int, float, float, int)
        Bounds and sizes for the grid of regularization hyperparameters.
        First-order: search in [regs[0], regs[1]] at regs[2] points.
        Quadratic:   search in [regs[3], regs[4]] at regs[5] points.
        Cubic:       search in [regs[6], regs[7]] at regs[8] points.
    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.
    margin : float ≥ 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).

    Returns
    -------
    regs : ndarray
        Regularization hyperparameter winners.
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    if len(regs) not in [6, 9]:
        raise ValueError("6 or 9 regs required (bounds / sizes of grids")
    grids = []
    for i in range(0, len(regs), 3):
        check_regs(regs[i:i+2])
        grids.append(np.logspace(np.log10(regs[i]),
                                 np.log10(regs[i+1]), int(regs[i+2])))
    modelform = get_modelform(grids)
    d = check_lstsq_size(trainsize, r, modelform)

    # Load training data.
    t = utils.load_time_domain(testsize)
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r)
    U = config.U(t[:trainsize])

    # Compute the bound to require for integrated POD modes.
    M = margin * np.abs(Q_).max()

    # Create a solver mapping regularization hyperparameters to operators.
    num_tests = np.prod([grid.size for grid in grids])
    print(f"TRAINING {num_tests} ROMS")
    with utils.timed_block(f"Constructing least-squares solver, r={r:d}"):
        rom = opinf.InferredContinuousROM(modelform)
        rom._construct_solver(None, Q_, Qdot_, U, np.ones(d))

    # Test each regularization hyperparameter.
    errors_pass = {}
    errors_fail = {}
    for i, regs in enumerate(itertools.product(*grids)):
        with utils.timed_block(f"({i+1:d}/{num_tests:d}) Testing ROM with "
                               f"{config.REGSTR(regs)}"):
            # Train the ROM on all training snapshots.
            rom._evaluate_solver(regularizer(r, *list(regs)))

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            errors = errors_pass if is_bounded(q_rom, M) else errors_fail

            # Calculate integrated relative errors in the reduced space.
            if q_rom.shape[1] > trainsize:
                errors[tuple(regs)] = opinf.post.Lp_error(Q_,
                                                          q_rom[:,:trainsize],
                                                          t[:trainsize])[1]

    # Choose and save the ROM with the least error.
    if not errors_pass:
        message = f"NO STABLE ROMS for r={r:d}"
        print(message)
        logging.info(message)
        return

    err2reg = {err:reg for reg,err in errors_pass.items()}
    regs = list(err2reg[min(err2reg.keys())])
    logging.info(f"Best regularization for k={trainsize:d}, r={r:d}: "
                 f"{config.REGSTR(regs)}")
    return regs


def train_minimize(trainsize, r, regs, testsize=None, margin=1.1):
    """Train ROMs with the given dimension(s), saving only the ROM with
    the least training error that satisfies a bound on the integrated POD
    coefficients, using a search algorithm to choose the regularization
    hyperparameters.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM.
    r : int
        Dimension of the desired ROM. Also the number of retained POD modes
        (left singular vectors) used to project the training data.
    regs : two positive floats
        Initial guesses for the regularization hyperparameters (non-quadratic,
        quadratic) to use in the Operator Inference least-squares problem
        for training the ROM.
    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.
    margin : float ≥ 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    modelform = get_modelform(regs)
    d = check_lstsq_size(trainsize, r, modelform)
    log10regs = np.log10(check_regs(regs))

    # Load training data.
    t = utils.load_time_domain(testsize)
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r)
    U = config.U(t[:trainsize])

    # Compute the bound to require for integrated POD modes.
    B = margin * np.abs(Q_).max()

    # Create a solver mapping regularization hyperparameters to operators.
    with utils.timed_block(f"Constructing least-squares solver, r={r:d}"):
        rom = opinf.InferredContinuousROM(modelform)
        rom._construct_solver(None, Q_, Qdot_, U, np.ones(d))

    # Test each regularization hyperparameter.
    def training_error(log10regs):
        """Return the training error resulting from the regularization
        parameters λ1 = 10^log10regs[0], λ1 = 10^log10regs[1]. If the
        resulting model violates the POD bound, return "infinity".
        """
        regs = list(10**log10regs)

        # Train the ROM on all training snapshots.
        with utils.timed_block(f"Testing ROM with {config.REGSTR(regs)}"):
            rom._evaluate_solver(regularizer(r, *regs))

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            if not is_bounded(q_rom, B):
                return _MAXFUN

            # Calculate integrated relative errors in the reduced space.
            return opinf.post.Lp_error(Q_,
                                       q_rom[:,:trainsize],
                                       t[:trainsize])[1]

    opt_result = opt.minimize(training_error, log10regs, method="Nelder-Mead")
    if opt_result.success and opt_result.fun != _MAXFUN:
        regs = list(10**opt_result.x)
        with utils.timed_block(f"Best regularization for k={trainsize:d}, "
                               f"r={r:d}: {config.REGSTR(regs)}"):
            rom._evaluate_solver(regularizer(r, *regs))
            save_trained_rom(trainsize, r, regs, rom)
    else:
        message = "Regularization search optimization FAILED"
        print(message)
        logging.info(message)


# First draft approach: single regularization hyperparameter, i.e., ===========
# equally penalize all entries of the ROM operators. ==========================
def _train_minimize_1D(trainsize, r, regs, testsize=None, margin=1.1):
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
    regs : two non-negative floats
        Bounds for the (single) regularization hyperparameter to use in the
        Operator Inference least-squares problem for training the ROM.
    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.
    margin : float ≥ 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    check_lstsq_size(trainsize, r, modelform="cAHB")
    log10regs = np.log10(regs)

    # Load training data.
    t = utils.load_time_domain(testsize)
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r)
    U = config.U(t[:trainsize])

    # Compute the bound to require for integrated POD modes.
    B = margin * np.abs(Q_).max()

    # Create a solver mapping regularization hyperparameters to operators.
    with utils.timed_block(f"Constructing least-squares solver, r={r:d}"):
        rom = opinf.InferredContinuousROM("cAHB")
        rom._construct_solver(None, Q_, Qdot_, U, 1)

    # Test each regularization hyperparameter.
    def training_error(log10reg):
        """Return the training error resulting from the regularization
        hyperparameters λ1 = λ2 = 10^log10reg. If the resulting model
        violates the POD bound, return "infinity".
        """
        λ = 10**log10reg

        # Train the ROM on all training snapshots.
        with utils.timed_block(f"Testing ROM with λ={λ:e}"):
            rom._evaluate_solver(λ)

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            if not is_bounded(q_rom, B):
                return _MAXFUN

            # Calculate integrated relative errors in the reduced space.
            return opinf.post.Lp_error(Q_,
                                       q_rom[:,:trainsize],
                                       t[:trainsize])[1]

    opt_result = opt.minimize_scalar(training_error,
                                     method="bounded", bounds=log10regs)
    if opt_result.success and opt_result.fun != _MAXFUN:
        λ = 10**opt_result.x
        with utils.timed_block(f"Best regularization for k={trainsize:d}, "
                               f"r={r:d}: λ={λ:.0f}"):
            rom._evaluate_solver(λ)
            save_trained_rom(trainsize, r, (λ,λ), rom)
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
        python3 {__file__} --single TRAINSIZE R REG1 REG2 [REG3]
        python3 {__file__} --gridsearch TRAINSIZE R REG1 ... REG6 [... REG9]
                               --testsize TESTSIZE --margin TAU
        python3 {__file__} --minimize TRAINSIZE R REG1 REG2 [REG3]
                               --testsize TESTSIZE --margin TAU"""
    # Parser subcommands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--single", action="store_true",
                       help="train and save a single ROM with regularization "
                            "hyperparameters REG1 (non-quadratic penalizer) "
                            "and REG2 (quadratic penalizer)")
    group.add_argument("--gridsearch", action="store_true",
                       help="train over the REG3xREG6 grid "
                            "[REG1,REG2]x[REG4,REG5] of regularization "
                            "hyperparameter candidates, followed by a "
                            "minimization-based search from the winner")
    group.add_argument("--minimize", action="store_true",
                       help="given initial guesses REG1 (non-quadratic  "
                            "penalizer) and REG2 (quadratic penalizer), use "
                            "Nelder-Mead search to train and save a ROM that "
                            "is locally optimal in the regularization "
                            "hyperparameter space")

    # Positional arguments.
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")
    parser.add_argument("modes", type=int,
                        help="number of POD modes used to project the data "
                             "(dimension of ROM to be learned)")
    parser.add_argument("regularization", type=float, nargs='+',
                        help="regularization hyperparameters for ROM training")

    # Other keyword arguments.
    parser.add_argument("--testsize", type=int, default=None,
                        help="number of time steps for which the trained ROM "
                             "must satisfy the POD bound")
    parser.add_argument("--margin", type=float, default=1.1,
                        help="factor by which the POD coefficients of the ROM "
                             "simulation are allowed to deviate in magnitude "
                             "from the training data (default 1.1)")

    # Parse arguments and do one of the main routines.
    args = parser.parse_args()
    if args.single:
        train_single(args.trainsize, args.modes, args.regularization)
    elif args.gridsearch:
        regs = train_gridsearch(args.trainsize, args.modes,
                                args.regularization,
                                args.testsize, args.margin)
        if regs is not None:
            train_minimize(args.trainsize, args.modes, regs,
                           args.testsize, args.margin)

    elif args.minimize:
        train_minimize(args.trainsize, args.modes, args.regularization,
                       args.testsize, args.margin)
