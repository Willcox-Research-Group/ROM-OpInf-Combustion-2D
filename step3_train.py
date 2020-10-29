# step3_train.py
"""Use projected data to learn reduced-order models via Operator Inference
with optimal regularization parameter selection.

Examples
--------
## --save-all: save each specified model.

# Use 10,000 projected snapshots to learn a ROM of dimension 24
# with a regularization parameter of 6 x 10^4.
$ python3 step3_train.py 10000 --save-all --modes 24 --regularization 6e4

# Use 20,000 projected snapshots to learn ROMs of dimension 24, 26, and 29,
# each with regularization parameters of 7 x 10^4.
$ python3 step3_train.py 15000 --save-all
                         --modes 24 26 29 --regularization 7e4

# Use 10,000 projected snapshots to learn ROMs of dimension 24
# with regularzation parameters of 6 x 10^4 and 7 x 10^4.
$ python3 step3_train.py 20000 --save-all
                         --modes 24 --regularization 6e4 7e4

# Use 10,000 projected snapshots to learn ROMs of dimension 17 through 30,
# each with regularization parameters of 6 x 10^4.
$ python3 step3_train.py 10000 --save-all
                         --modes 17 30 --moderange --regularization 6e4

## --gridsearch: save only the best model from the specified regularizations.

# Use 10,000 projected snapshots to learn a ROM of dimension 24 and save the
# one with the best regularization out of 5 x 10^4, 6 x 10^4, and 7 x 10^4
# in which the integrated POD modes stay within 150% of the training data
# in magnitude for 60,000 time steps.
$ python3 step3_train.py 10000 --gridsearch
                         --modes 24 --regularization 5e4 6e4 7e4
                         --testsize 60000 --margin 1.5

# Use 20,000 projected snapshots to learn a ROM of dimension 30 and save the
# one with the best regularization parameter out of a logarithmically spaced
# grid of 20 candidates from 10^4 to 10^5, requiring the integrated POD modes
# to stay within 200% of the training data in magnitude for 40,000 steps.
$ python3 step3_train.py 20000 --gridsearch
                         --modes 30 --regularization 1e4 1e5 --regrange 20
                         --testsize 40000 --margin 2

## --minimize: save only the best model from an interval of regularization
   parameters, using a one-dimensional bisection-type minimization search.

# Use 10,000 projected snapshots to learn a ROM of dimension 22 and save the
# one with the best regularization parameter out between 10^4 and 10^5
# in which the integrated POD modes stay within 150% of the training data
# in magnitude for 60,000 time steps.
$ python3 step3_train.py 10000 --minimize
                         --modes 22 --regularization 1e4 1e5
                         --testsize 60000 --margin 1.5

Loading Results
---------------
>>> import utils
>>> trainsize = 10000       # Number of snapshots used as training data.
>>> num_modes = 44          # Number of POD modes.
>>> reg = 1e4               # Regularization parameter for Operator Inference.
>>> rom = utils.load_rom(trainsize, num_modes, reg)

Command Line Arguments
----------------------
"""
import h5py
import logging
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils


_MAXFUN = 100               # Artificial ceiling for optimization routine.


# Subroutines =================================================================

def is_bounded(q_rom, B, message="ROM violates bound"):
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


def save_best_trained_rom(trainsize, r1, r2, reg, rom):
    """Save the trained ROM with the specified attributes.

    Parameters
    ----------
    trainsize : int
        Number of snapshots used to train the ROM.

    r1 : int
        Number of retained POD modes (left singular vectors) used to project
        the non-temperature training data.

    r2 : int
        Number of retained POD modes (left singular vectors) used to project
        the temperature training data.

    reg : float > 0
        Regularization parameter used in the training.

    rom : rom_operator_inference.InferredContinuousROM
        Actual trained ROM object. Must have a `save_model()` method.
    """
    save_path = config.rom_path(trainsize, r1, r2, reg)
    with utils.timed_block("Best regularization for "
                           f"r1={r1:d}, r2={r2:d}: {reg:.0f}"):
        rom.save_model(save_path, save_basis=False, overwrite=True)
    logging.info(f"ROM saved to {save_path}")


# Main routines ===============================================================

def train_and_save_all(trainsize, r1, r2, regs):
    """Train and save ROMs with the given dimension and regularization.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM(s).

    r1 : int
        Number of retained POD modes (left singular vectors) used to project
        the non-temperature training data.

    r2 : int
        Number of retained POD modes (left singular vectors) used to project
        the temperature training data.

    regs : float or list(float)
        regularization parameter(s) to use in the training.
    """
    utils.reset_logger(trainsize)

    logging.info(f"TRAINING {len(regs)} ROMS")

    # Load training data.
    Q_, Qdot_, time_domain = utils.load_projected_data(trainsize, r1, r2)

    # Evaluate inputs over the training time domain.
    U = config.U(time_domain)

    # Create a solver mapping regularization parameters to operators.
    with utils.timed_block("Constructing/factoring data matrix, "
                           f"r1={r1:d}, r2={r2:d}"):
        rom = roi.InferredContinuousROM(config.MODELFORM)
        Q_, Qdot_, U = rom._process_fit_arguments(None, Q_, Qdot_, U)
        solver = roi.lstsq.LstsqSolverL2(compute_extras=False)
        solver.fit(rom._construct_data_matrix(Q_, U), Qdot_.T)

    # Train and save each ROM.
    for reg in regs:
        with utils.timed_block(f"Training ROM with r={r:d}, reg={reg:e}"):
            rom._extract_operators(solver.predict(reg).T)
            rom.save_model(config.rom_path(trainsize, r1, r2, reg),
                           save_basis=False, overwrite=True)


def train_with_gridsearch(trainsize, r1, r2, regs,
                          testsize=None, margin=1.5):
    """Train ROMs with the given dimension(s) and regularization(s),
    saving only the ROM with the least training error that satisfies
    a bound on the integrated POD coefficients.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM(s).

    r1 : int
        Number of retained POD modes (left singular vectors) used to project
        the non-temperature training data.

    r2 : int
        Number of retained POD modes (left singular vectors) used to project
        the temperature training data.

    regs : float or list(float)
        regularization parameter(s) to use in the training.

    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.

    margin : float >= 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    if np.isscalar(regs):
        regs = [regs]

    # Load the full time domain and evaluate the input function.
    t = utils.load_time_domain(testsize)
    U = config.U(t)

    rom = roi.InferredContinuousROM(config.MODELFORM)
    logging.info(f"TRAINING {len(regs)} ROMS")

    # Load training data.
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r1, r2)

    # Compute the bound to require for integrated POD modes.
    M = margin * np.abs(Q_).max()

    # Create a solver mapping regularization parameters to operators.
    with utils.timed_block("Constructing/factoring data matrix, "
                           f"r1={r1:d}, r2={r2:d}"):
        Q_, Qdot_, Utrain = rom._process_fit_arguments(None, Q_, Qdot_,
                                                             U[:trainsize])
        solver = roi.lstsq.LstsqSolverL2(compute_extras=False)
        solver.fit(rom._construct_data_matrix(Q_, Utrain), Qdot_.T)

    # Test each regularization parameter.
    trained_roms = {}
    errors_pass = {}
    errors_fail = {}
    for reg in regs:

        # Train the ROM on all training snapshots.
        with utils.timed_block("Testing ROM with "
                               f"r1={r1:d}, r2={r2:d}, reg={reg:e}"):
            rom._extract_operators(solver.predict(reg).T)
            rom._construct_f_()
            trained_roms[reg] = rom

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            errors = errors_pass if is_bounded(q_rom, M) else errors_fail

            # Calculate integrated relative errors in the reduced space.
            if q_rom.shape[1] > trainsize:
                errors[reg] = roi.post.Lp_error(Q_, q_rom[:,:trainsize],
                                                          t[:trainsize])[1]

    # Choose and save the ROM with the least error.
    plt.semilogx(list(errors_fail.keys()), list(errors_fail.values()),
                 f"C0x", mew=1,
                 label=fr"$r_1 = {r1:d}, r_2 = {r2:d}$, bound violated")
    if not errors_pass:
        print(f"NO STABLE ROMS for r1={r1:d}, r2={r2:d}")
        return

    err2reg = {err:reg for reg,err in errors_pass.items()}
    best_reg = err2reg[min(err2reg.keys())]
    best_rom = trained_roms[best_reg]
    save_best_trained_rom(trainsize, r1, r2, best_reg, best_rom)

    plt.semilogx(list(errors_pass.keys()), list(errors_pass.values()),
                 f"C0*", mew=0,
                 label=fr"$r_1 = {r1:d}, r_2 = {r2:d}$, bound satisfied")
    plt.axvline(best_reg, lw=.5, color=f"C0")

    plt.legend()
    plt.xlabel(r"Regularization parameter $\lambda$")
    plt.ylabel(r"ROM relative error $\frac"
               r"{||\widehat{\mathbf{Q}} - \widetilde{\mathbf{Q}}_{:,:k}||}"
               r"{||\widehat{\mathbf{Q}}||}$")
    plt.ylim(0, 1)
    plt.xlim(min(regs), max(regs))
    plt.title(fr"$n_t = {trainsize}$")
    utils.save_figure(f"regsweep_nt{trainsize:05d}.pdf")


def train_with_minimization(trainsize, r1, r2, regs,
                            testsize=None, margin=1.5):
    """Train ROMs with the given dimension(s), saving only the ROM with
    the least training error that satisfies a bound on the integrated POD
    coefficients, using a search algorithm to choose the regularization
    parameter.

    Parameters
    ----------
    trainsize : int
        Number of snapshots to use to train the ROM(s).

    r1 : int
        Number of retained POD modes (left singular vectors) used to project
        the non-temperature training data.

    r2 : int
        Number of retained POD modes (left singular vectors) used to project
        the temperature training data.

    regs : [float, float]
        Regularization parameter(s) to use in the training.

    testsize : int
        Number of time steps for which a valid ROM must satisfy the POD bound.

    margin : float >= 1
        Amount that the integrated POD coefficients of a valid ROM are allowed
        to deviate in magnitude from the maximum magnitude of the training
        data Q, i.e., bound = margin * max(abs(Q)).
    """
    utils.reset_logger(trainsize)

    # Parse aguments.
    if np.isscalar(regs) or len(regs) != 2:
        raise ValueError("2 regularizations required (reg_low, reg_high)")
    bounds = np.log10(regs)

    # Load the full time domain and evaluate the input function.
    t = utils.load_time_domain(testsize)
    U = config.U(t)

    rom = roi.InferredContinuousROM(config.MODELFORM)

    # Load training data.
    Q_, Qdot_, _ = utils.load_projected_data(trainsize, r1, r2)

    # Compute the bound to require for integrated POD modes.
    B = margin * np.abs(Q_).max()

    # Create a solver mapping regularization parameters to operators.
    with utils.timed_block("Constructing/factoring data matrix, "
                           f"r1={r1:d}, r2={r2:d}"):
        Q_, Qdot_, Utrain = rom._process_fit_arguments(None, Q_, Qdot_,
                                                             U[:trainsize])
        solver = roi.lstsq.LstsqSolverL2(compute_extras=False)
        solver.fit(rom._construct_data_matrix(Q_, Utrain), Qdot_.T)

    # Test each regularization parameter.
    def training_error_from_rom(log10reg):
        reg = 10**log10reg

        # Train the ROM on all training snapshots.
        with utils.timed_block("Testing ROM with "
                               f"r1={r1:d}, r2={r2:d}, reg={reg:e}"):
            rom._extract_operators(solver.predict(reg).T)
            rom._construct_f_()

            # Simulate the ROM over the full domain.
            with np.warnings.catch_warnings():
                np.warnings.simplefilter("ignore")
                q_rom = rom.predict(Q_[:,0], t, config.U, method="RK45")

            # Check for boundedness of solution.
            if not is_bounded(q_rom, B):
                return _MAXFUN

            # Calculate integrated relative errors in the reduced space.
            return roi.post.Lp_error(Q_, q_rom[:,:trainsize], t[:trainsize])[1]

    opt_result = opt.minimize_scalar(training_error_from_rom,
                                     bounds=bounds, method="bounded")
    if opt_result.success and opt_result.fun != _MAXFUN:
        best_reg = 10 ** opt_result.x
        rom._extract_operators(solver.predict(best_reg).T)
        save_best_trained_rom(trainsize, r1, r2, best_reg, rom)
    else:
        print("Regularization search optimization FAILED "
              f"for r1={r1:d}, r2={r2:d}")


# =============================================================================
if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.usage = f""" python3 {__file__} --help
        python3 {__file__} TRAINSIZE --save-all
                               --modes R [...] [--moderange]
                               --regularization REG [...] [--regrange NREGS]
        python3 {__file__} TRAINSIZE --gridsearch
                               --modes R [...] [--moderange]
                               --regularization REG [...] [--regrange NREGS]
                               --testsize TESTSIZE --margin TAU
        python3 {__file__} TRAINSIZE --minimize
                               --modes R [...] [--moderange]
                               --regularization REGLOW REGHIGH
                               --testsize TESTSIZE --margin TAU"""
    parser.add_argument("trainsize", type=int,
                        help="number of snapshots in the training data")

    # Parser subcommands
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save-all", action="store_true",
                       help="train and save ROMs for each regulazation")
    group.add_argument("--gridsearch", action="store_true",
                       help="train with all regularization candidates "
                            "and save only the best ROM")
    group.add_argument("--minimize", action="store_true",
                       help="find the best regularization within an "
                            "interval via minimize scalar minimization")

    # Main positional arguments.
    parser.add_argument("-r1", type=int, required=True,
                        help="number of POD modes used to project "
                             "non-temperature data")
    parser.add_argument("-r2", type=int, required=True,
                        help="number of POD modes used to project "
                             "temperature data")
    parser.add_argument("-reg", "--regularization", type=float, nargs='+',
                        required=True,
                        help="regularization parameter(s) for ROM training "
                             "or, with --minimize, upper and lower bounds "
                             "for the regularization")
    parser.add_argument("--regrange", type=int, nargs='?', default=0,
                        help="if two regularizations given, treat them as min,"
                             " max and train with REGRANGE regularizations"
                             " logarithmically spaced in [min,max]")
    parser.add_argument("-tf", "--testsize", type=int, default=None,
                        help="number of time steps for which the trained ROM "
                             "must satisfy the POD bound")
    parser.add_argument("-tau", "--margin", type=float, default=1.5,
                        help="percent that the POD coefficients of the "
                             "ROM simulation are allowed to deviate in "
                             "magnitude from the training data")

    # Parse arguments and mode and regularization options.
    args = parser.parse_args()
    if args.regrange and len(args.regularization) == 2 and not args.minimize:
        args.regularization = np.logspace(np.log10(args.regularization[0]),
                                          np.log10(args.regularization[1]),
                                          args.regrange)
    regs = np.unique(np.array(args.regularization, dtype=int))

    # Do one of the main routines.
    if args.save_all:
        train_and_save_all(args.trainsize, args.r1, args.r2, regs)
    elif args.gridsearch:
        train_with_gridsearch(args.trainsize, args.r1, args.r2, regs,
                              args.testsize, args.margin)
    elif args.minimize:
        train_with_minimization(args.trainsize, args.r1, args.r2, regs,
                                args.testsize, args.margin)
