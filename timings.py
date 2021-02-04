# timings.py
"""Time Operator Inference training and prediction for this problem."""
import time
import numpy as np

import utils
import config
import step3_train as step3

import rom_operator_inference as roi


def main(roms, num_trials):
    """Run `num_trials` for each ROM and report average training and
    prediction times.

    Parameters
    ----------
    roms : list(3-tuple)
        List of ROM parameters (k,r,(λ1,λ2)).
    
    num_trials : int
        Number of times to run each experiment.
    """
    # Load full time domain.
    t = utils.load_time_domain(60000)

    # Do the tests.
    for trainsize, num_modes, reg in roms:
        X_, Xdot_, t_ = utils.load_projected_data(trainsize, num_modes)
        Us = config.U(t_)
        rom_label = f"k = {trainsize:d}, r = {num_modes:d}"
        d = step3.check_lstsq_size(trainsize, num_modes)
        P = step3.regularizer(num_modes, d, reg[0], reg[1])
        rom = roi.InferredContinuousROM(config.MODELFORM)
        with utils.timed_block(f"Running {num_trials} tests for {rom_label}"):
            traintimes, simtimes = [], []
            for _ in range(num_trials):
                # Time training.
                _start = time.time()
                rom.fit(None, X_, Xdot_, Us, P,
                        compute_extras=False, check_regularizer=False)
                _end = time.time()
                traintimes.append(_end - _start)

                # Time integration.
                _start = time.time()
                rom.predict(X_[:,0], t, config.U, method="RK45")
                _end = time.time()
                simtimes.append(_end - _start)

        print(rom_label,
              f"training:\t{np.mean(traintimes):.6f} ± {np.std(traintimes):.6f}s",
              f"integration:\t{np.mean(simtimes):.6f} ± {np.std(simtimes):.6f}s",
              sep="\n\t")


if __name__ == "__main__":
    ROMS = [
                (20000, 43, (316, 18199)),
             ]
    main(ROMS, 10)
