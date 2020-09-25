# bayes.py
"""Bayesian interpretation of Operator Inference for this problem.
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import rom_operator_inference as roi

import config
import utils
import data_processing as dproc


class OpInfPosterior:
    """Convenience class for sampling the posterior distribution, which
    is a different multivariate normal for each row of the operator matrix.
    """
    def __init__(self, means, covariances, modelform="cAHB"):
        """Construct the multivariate normal random variables."""
        assert len(means) == len(covariances)
        r = len(means)
        d = 1 + r + r*(r+1)//2 + 1
        assert means.shape == (r, d)
        self._d = d
        
        # self.components = [stats.multivariate_normal(mean, cov)
        #                     for mean, cov in zip(means, covariances)]
        # Initialize the random variables for each operator row.
        self.means = means
        self.covariances = covariances
        self._choleskies = [np.linalg.cholesky(cov) for cov in covariances]

        # Set other convenience attributes.
        self._modelform = modelform
        if self._modelform == "cAHB":
            self._indices = np.cumsum([1, r, r*(r+1)//2])
        else:
            raise NotImplementedError(f"modelform='{self._modelform}'")
    
    def rvs(self):
        """Do a single sample of the posterior distribution.

        Returns
        -------
        rom : roi.InferredContinuousROM
            A trained reduced-order model, representing a posterior draw.
        """
        # O = np.vstack([c.rvs() for c in self.components])
        O = np.vstack([mean + cho @ np.random.standard_normal(self._d)
                       for mean, cho in zip(self.means, self._choleskies)])
        c_, A_, Hc_, B_ = np.split(O, self._indices, axis=1)
        rom = roi.InferredContinuousROM(self._modelform)
        return rom._set_operators(None, c_=c_.flatten(), A_=A_, Hc_=Hc_, B_=B_)

    def predict(self, x0, t):
        """Draw a ROM from the posterior and simulate it from x0 over t."""
        return self.rvs().predict(x0, t, config.U, method="RK45")


def _data_matrix(trainsize, r):
    """Get the data matrix D for the given training size and
    ROM dimension.

    Returns
    -------
    D : (d(r,m),k) ndarray
        Data matrix for Operator Inference.

    R : (r,k) ndarray
        Right-hand side matrix for Operator Inference (time derivatives).
    """
    X_, Xdot_, t, _ = utils.load_projected_data(trainsize, r)
    U = config.U(t).reshape((1,-1))
    D = roi.InferredContinuousROM("cAHB")._construct_data_matrix(X_, U)
    return D, Xdot_


def _operator_matrix(trainsize, r, reg):
    """Extract the block matrix of learned operators O from the trained ROM.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r : int
        The dimension of the ROM. This is also the number of retained POD modes
        (left singular vectors) used to project the training data.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROM.

    Returns
    -------
    rom : roi.InferredContinuousROM
        The trained reduced-order model.

    O : (r,d) ndarray
        Operator matrix O = [c | A | Hc | B].
    """
    rom = utils.load_rom(trainsize, r, reg)
    return rom, np.column_stack((rom.c_, rom.A_, rom.Hc_, rom.B_))


def construct_posterior(trainsize, r, reg):
    """Construct the mean and covariance matrix for the posterior distribution,
    then create an object for sampling the posterior.

    Parameters
    ----------
    trainsize : int
        The number of snapshots used to train the ROM. This is also the number
        of snapshots that were used when the POD basis (SVD) was computed.

    r : int
        The dimension of the ROM. This is also the number of retained POD modes
        (left singular vectors) used to project the training data.

    reg : float
        The regularization factor used in the Operator Inference least-squares
        problem for training the ROM.

    Returns
    -------
    rom : roi.InferredContinuousROM
        The trained reduced-order model (mean of the posterior).

    post : OpInfPosterior
        Posterior distribution object with rvs() sampling method.
    """
    # Get the data matrices and the (previously) learned operators.
    d = 1 + r + r*(r+1)//2 + 1
    D,R = _data_matrix(trainsize, r)
    rom,O = _operator_matrix(trainsize, r, reg)
    assert D.shape == (trainsize, d)
    assert R.shape == (r, trainsize)
    assert O.shape == (r, d)

    with utils.timed_block("Computing posterior parameters"):
        # Construct the data Grammian and the regularization matrix.
        reg2 = reg**2
        S = la.inv(D.T @ D + reg2*np.eye(d))

        # Numerically symmetrize / sparsify Sigma for stability.
        Sigma_unscaled = (S + S.T) / 2
        Sigma_unscaled[np.abs(Sigma_unscaled) < 1e-16] = 0

        # Get each covariance matrix.
        covariances = []
        sigma2s = []
        for i in range(r):
            o, r = O[i,:], R[i,:]
            sig2 = (np.sum((D @ o - r)**2) + (reg**2)*np.sum(o**2)) / trainsize
            sigma2s.append(sig2)
            covariances.append(sig2*Sigma_unscaled)

    with utils.timed_block("Building posterior distribution"):
        return rom, OpInfPosterior(O, covariances)


def simulate_posterior(rom, post, ndraws=10, steps=None):
    """
    Parameters
    ----------
    rom : ...

    post : OpInfPosterior
    
    ndraws : int

    steps : int

    Returns
    -------
    x_rom_mean : (r,steps) ndarray
        TODO

    x_roms : list
        TODO

    scales : ndarray
        TODO
    """
    # Load the time domain and initial conditions.
    t = utils.load_time_domain(steps)
    X_, _, _, scales = utils.load_projected_data(rom.trainsize, rom.r)
    x0 = X_[:,0]

    # Simulate the mean ROM as a reference.
    with utils.timed_block(f"Simulating mean ROM"):
        x_rom_mean = rom.predict(x0, t, config.U, method="RK45")
    
    # Get ndraws simulation samples.
    x_roms = []
    i = 0
    while i < ndraws:
        with utils.timed_block(f"Simulating posterior draw ROM {i+1}"):
            x_rom = post.predict(x0, t)
            if x_rom.shape[1] == t.shape[0]:
                x_roms.append(x_rom)
                i += 1

    return x_rom_mean, x_roms, scales


def plot_mode_uncertainty(trainsize, mean, draws, modes=8):
    steps = mean.shape[1]
    t = utils.load_time_domain(steps)

    with utils.timed_block("Calculating sample deviations"):
        offsets = [draw - mean for draw in draws]
        deviations = np.std(offsets, axis=0)

    nrows = (modes//2) + 1 if modes % 2 else modes//2
    fig, axes = plt.subplots(nrows, 2)
    for i, ax in zip(range(modes), axes.flat):
        ax.plot(t, mean[i,:], 'C0-', lw=1)
        # for draw in draws:
        #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.2)
        ax.fill_between(t, mean[i,:]-deviations[i,:],
                           mean[i,:]+deviations[i,:], alpha=.5)
        ax.axvline(t[trainsize], color='k', lw=1)
        # ax.set_title(fr"POD mode ${i+1}$")
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
    fig.tight_layout()


def plot_pointtrace_uncertainty(trainsize, mean, draws, scales, var="p"):
    """

    Parameters
    ----------

    Returns
    -------
    """
    if var not in ["p", "vx", "vy", "T"]:
        raise NotImplementedError(f"var='{var}'")

    # Get the indicies for each variable.
    elems = np.atleast_1d(config.MONITOR_LOCATIONS)
    nelems = elems.size
    nrows = (nelems // 2) + (1 if nelems % 2 != 0 else 0)
    elems = elems + config.ROM_VARIABLES.index(var)*config.DOF

    # Load the true pressure traces and the time domain.
    traces_gems, t = utils.load_gems_data(rows=elems)
    steps = mean.shape[1]
    t = t[:steps]
    traces_gems = traces_gems[:,:steps]

    # Load the basis rows corresponding to the pressure traces.
    V, _ = utils.load_basis(trainsize, mean.shape[0])
    Velems = V[elems]

    # Reconstruct and rescale the simulation results.
    with utils.timed_block("Reconstructing simulation results"):
        traces_rom_mean = dproc.unscale(Velems @ mean, scales, var)
        traces_rom_draws = [dproc.unscale(Velems @ draw, scales, var)
                            for draw in draws]

    with utils.timed_block("Calculating sample deviations"):
        offsets = [draw - traces_rom_mean for draw in traces_rom_draws]
        deviations = np.std(offsets, axis=0)

    fig, axes = plt.subplots(nrows, 2, figsize=(9,6), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, traces_gems[i,:], lw=1, **config.GEMS_STYLE)
        ax.plot(t, traces_rom_mean[i,:], 'C0-', lw=1,
                label=r"ROM ($\mu$)")
        # for draw in traces_rom_draws:
        #     ax.plot(t, draw[i,:], 'C0-', lw=.5, alpha=.25)
        ax.fill_between(t, traces_rom_mean[i,:]-deviations[i,:],
                           traces_rom_mean[i,:]+deviations[i,:], alpha=.5)
        ax.axvline(t[trainsize], color='k', lw=1)
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks(np.arange(t[0], t[-1]+.001, .002))
        ax.set_title(f"Location ${i+1}$", fontsize=12)
        ax.locator_params(axis='y', nbins=2)
    for ax in axes[-1,:]:
        ax.set_xlabel("Time [s]", fontsize=12)
    for ax in axes[:,0]:
        ax.set_ylabel(config.VARLABELS[var], fontsize=12)

    # Single legend to the right of the subplots.
    fig.tight_layout(rect=[0, 0, .85, 1])
    leg = axes[0,0].legend(loc="center right", fontsize=14,
                           bbox_to_anchor=(1,.5),
                           bbox_transform=fig.transFigure)
    for line in leg.get_lines():
        line.set_linewidth(2)


def main(trainsize, r, reg, ndraws=10, steps=50000, modes=4):
    rom, post = construct_posterior(trainsize, r, reg)
    mean, draws, scales = simulate_posterior(rom, post, ndraws, steps)
    plot_mode_uncertainty(trainsize, mean, draws, modes)
    utils.save_figure("bayes_first4modes.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, scales, var="p")
    utils.save_figure("bayes_traces_pressure.pdf")
    plot_pointtrace_uncertainty(trainsize, mean, draws, scales, var="T")
    utils.save_figure("bayes_traces_temperature.pdf")


if __name__ == "__main__":
    main(20000, 44, 29638, 100, 50000, 4)
